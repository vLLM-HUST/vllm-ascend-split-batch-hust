# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure routing logic for the fi_sampling (flashinfer sampling) plugin.

No torch / vllm / triton import: every decision below is CPU-testable plain
logic (see ``tests/test_fi_sampling_route.py``).

Bench provenance (W2 package, reviewed 2026-09-09, V=152064 fp32, 910B2,
triton-ascend 3.2.1 -- ``flashinfer-migration/sampling/报告-W2-sampling-
migration-2026-09-09.md`` §4.8 "when it wins / when it loses"):

| workload                              | verdict                          |
|---------------------------------------|----------------------------------|
| any B, no top-k / top-p               | ours wins 1.12x (B=1) -> 12.83x (B=1024) |
| B >= 256, joint k=50 / p=0.95 (bv=4096) | ours wins 1.28x / 1.79x        |
| B = 1024, joint k=50 / p=0.95 (bv=1024) | tie (0.99x)                    |
| B <= 64, joint k=50 / p=0.95          | ours loses 1.33x - 3.83x         |
| any B, k = 1                          | ours loses 1.43x - 6.35x (route to argmax) |

Therefore the only *measured* wins are (a) the untruncated path at any batch
size and (b) the joint path at large batch with a medium/large k.  Everything
else -- including the k-only / p-only large-batch corners, which the reviewed
judgment table does not cover -- stays on the fork chain (conservative
default; the thresholds below are tunable through the plugin env knobs so the
e2e evidence can A/B them).

The five mandatory fallbacks (fork chain) come from the W2 review:

1. per-request generators: ``fi_sampling`` derives its seed from
   ``generator.initial_seed()`` only and never advances it (unlike the
   flashinfer contract), so any request carrying a persistent generator must
   use the fork chain;
2. ``processed_logits`` / ``processed_logprobs`` logprobs modes: the FI path
   cannot return post-truncation logits/logprobs;
3. ``VLLM_BATCH_INVARIANT``: batch-invariant sampling requires the vLLM
   native implementation;
4. ``enable_reduce_sample``: the fork's distributed (TP all-gather) sampling
   path owns the logits layout;
5. ``enable_async_exponential``: the fork's pre-computed exponential stream
   (``q``/event) is consumed by the fork chain only.
"""

from __future__ import annotations

# Route identifiers (kept as plain strings: the plugin module and the tests
# compare them without importing torch).
ROUTE_FORK = "fork"
ROUTE_ARGMAX = "argmax"
ROUTE_FI_API1 = "fi_api1"
ROUTE_FI_JOINT = "fi_joint"

# Fallback reasons (stable strings, surfaced in the single warning log).
FALLBACK_GENERATORS = "per_request_generators"
FALLBACK_LOGPROBS_MODE = "processed_logprobs_mode"
FALLBACK_BATCH_INVARIANT = "batch_invariant"
FALLBACK_REDUCE_SAMPLE = "reduce_sample"
FALLBACK_ASYNC_EXPONENTIAL = "async_exponential"

# Logprobs modes that need post-truncation logits/logprobs (fallback 2).
PROCESSED_LOGPROBS_MODES = ("processed_logits", "processed_logprobs")

# --- tunables, all sourced from the reviewed W2 bench -----------------------
#: joint kernel starts winning at B >= 256 (bench table B, block_v=4096).
FI_JOINT_MIN_BATCH = 256
#: k >= 32 keeps the rejection loop cheap enough (bench: k=50 wins at B>=256,
#: k=1 loses at every B -> argmax route).
FI_JOINT_MIN_K = 32
#: tuned block_v for the joint kernel (1.65x faster than 1024 at B=64, and the
#: configuration behind the 1.28x/1.79x wins at B=256/1024).
FI_JOINT_BLOCK_V = 4096
#: untruncated path is block_v-insensitive; keep the default configuration.
FI_API1_BLOCK_V = 1024
#: k == 1 is "approximate greedy": sample the argmax without the kernel.
ARGMAX_TOP_K = 1


def fallback_reason(
    *,
    logprobs_mode: str,
    has_generators: bool,
    batch_invariant: bool,
    reduce_sample: bool,
    async_exponential: bool,
) -> str | None:
    """Return the first fallback reason that forces the fork chain, else None.

    The order matches the reviewed fallback matrix; it is also the order the
    plugin reports in its single warning, so a hit is directly attributable.
    """
    if has_generators:
        return FALLBACK_GENERATORS
    if logprobs_mode in PROCESSED_LOGPROBS_MODES:
        return FALLBACK_LOGPROBS_MODE
    if batch_invariant:
        return FALLBACK_BATCH_INVARIANT
    if reduce_sample:
        return FALLBACK_REDUCE_SAMPLE
    if async_exponential:
        return FALLBACK_ASYNC_EXPONENTIAL
    return None


def decide_route(
    *,
    logprobs_mode: str,
    has_generators: bool,
    batch_invariant: bool,
    reduce_sample: bool,
    async_exponential: bool,
    batch_size: int,
    has_top_k: bool,
    has_top_p: bool,
    top_k_min: int | None = None,
    top_k_max: int | None = None,
    joint_min_k: int = FI_JOINT_MIN_K,
    joint_min_batch: int = FI_JOINT_MIN_BATCH,
    k1_argmax: bool = True,
) -> str:
    """Pick the sampling route for one ``forward_native`` call.

    ``top_k_min`` / ``top_k_max`` are the host-side extremes of the per-row
    top-k tensor (``None`` when no top-k applies); they are the only device
    values the routing needs, and the caller materializes them once per call.
    ``batch_size`` is the number of rows in ``logits``.  ``k1_argmax=False``
    sends the k=1 cell to the fork chain instead of argmax (tie-fidelity
    escape hatch, see :func:`k1_is_greedy`).
    """
    if (
        fallback_reason(
            logprobs_mode=logprobs_mode,
            has_generators=has_generators,
            batch_invariant=batch_invariant,
            reduce_sample=reduce_sample,
            async_exponential=async_exponential,
        )
        is not None
    ):
        return ROUTE_FORK

    if not has_top_k and not has_top_p:
        # Main win region: vLLM's default (no top-k/top-p) random sampling.
        return ROUTE_FI_API1

    if has_top_k and k1_is_greedy(top_k_min, top_k_max):
        return ROUTE_ARGMAX if k1_argmax else ROUTE_FORK

    if (
        batch_size >= joint_min_batch
        and has_top_k
        and has_top_p
        and top_k_min is not None
        and top_k_min >= joint_min_k
    ):
        # Measured win region (bench table B): joint k=50 / p=0.95, B >= 256,
        # block_v=4096.  k-only and p-only large-batch cells are NOT measured
        # against the fork chain and deliberately stay there (see module doc).
        return ROUTE_FI_JOINT

    return ROUTE_FORK


def k1_is_greedy(top_k_min: int | None, top_k_max: int | None) -> bool:
    """True when every row of the batch keeps only its maximum token(s)."""
    return top_k_min == ARGMAX_TOP_K and top_k_max == ARGMAX_TOP_K


def needs_top_k_read(*, batch_size: int, joint_min_batch: int) -> bool:
    """Whether the per-row top-k extremes can change the decision.

    The caller uses this to avoid a device->host read on cells that resolve
    without it.  The read costs a per-step host sync, which MEASURED as far
    more than its raw copy time: the first api1 pair (B=64, top_k=20 served,
    both legs routed to the fork chain) ran 120.77 ms vs 115.18 ms median TPOT
    (+4.9%) purely from one D2H per decode step
    (docs/evidence/w2b-fi-sampling/REPORT.md §4).  So the read is taken only
    when the joint cell -- the one measured e2e-relevant use of the value --
    is reachable:

    - ``batch_size >= joint_min_batch``: joint / k=1 cells may apply -> read;
    - otherwise: no FI cell can fire, so the call goes to the fork chain
      without any sync (a k=1 batch at small B is served by the fork chain;
      at that size sampling is <1% of TPOT, so short-circuiting it cannot pay
      for a per-step sync).
    """
    return batch_size >= joint_min_batch
