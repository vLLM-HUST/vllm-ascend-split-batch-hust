# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Startup micro-bench gate for cascade decode buckets (default-off, W2).

Cascade attention is a conditional optimization: on this host the e2e matrix
shows losses only in the small-prefix/small-batch corner (plan-exp W5: the
4.4k x B32 cell loses ~7% while every >=8k cell wins).  Rather than assuming
the winning region, this module MEASURES it once per process at capture time
and lets the replay paths consume the decision table.

Bench design (per uniform batch bucket N x shared-prefix bucket P):
    cascade cost = fa_fp32_stage1(flattened T=N over P)
                 + stage-2 FIA v2 (TND, per-request suffix)
                 + lse_merge
    full cost    = single FIA v2 over the total kv (production .out form)
A bucket enables cascade only when cascade is >= CASCADE_GATE_MARGIN (2%)
faster; unbenched buckets default to cascade-on (current behavior), so the
gate can only suppress the measured loss region, never the wins.

Ordering hazard (W0 report sec.3): after the first ``fa_fp32_stage1`` call in
a process, eager FIA v2 with kv >= ~8k faults on the aicore.  The bench is
therefore invoked BETWEEN the standard FULL captures and the twin captures,
and inside the bench every full-path probe runs BEFORE any cascade probe.

Consumption (both keyed on (num_tokens, prefix bucket), same verdict):
  - replay update pass: gate-off steps re-parameterize the STANDARD task
    groups (cascade key untouched);
  - ACLGraphWrapper variant selection: gate-off steps replay the standard
    graph instead of the twin.
Both fall back to the same code path as the documented twin-miss fail-open,
so a gated step is numerically the non-cascade path.

Env:
  VLLM_ASCEND_CASCADE_ADAPTIVE_GATE=1     enable benching + consumption
  VLLM_ASCEND_CASCADE_GATE_OVERRIDE=on|off  force every bucket (skips bench)
"""

from __future__ import annotations

import logging
import os
import threading

import torch
import torch_npu  # noqa: F401  (device init happens in the runner process)

logger = logging.getLogger(__name__)

ENV_GATE = "VLLM_ASCEND_CASCADE_ADAPTIVE_GATE"
ENV_OVERRIDE = "VLLM_ASCEND_CASCADE_GATE_OVERRIDE"

# Cascade must be at least this much faster to be chosen (noise hysteresis).
GATE_MARGIN = 0.02
# Representative per-request suffix for the probes (2 blocks of 128).
SUFFIX = 256

_lock = threading.Lock()
_decisions: dict = {}          # (num_tokens, prefix_bucket) -> bool
_prefix_buckets: list = []      # ascending shared-prefix bucket edges
_bench_duration_s = 0.0


def enabled() -> bool:
    return os.getenv(ENV_GATE) == "1"


def override() -> str | None:
    val = os.getenv(ENV_OVERRIDE)
    if val in ("on", "off"):
        return val
    return None


def reset_for_tests() -> None:
    """Clear all bench state (unit tests only)."""
    with _lock:
        _decisions.clear()
        _prefix_buckets.clear()
        globals()["_bench_duration_s"] = 0.0


def bench_duration() -> float:
    return _bench_duration_s


def decision_for(shared_len: int, num_tokens: int) -> bool:
    """Gate verdict for a step with this shared prefix and batch bucket.

    Override env wins; then the benched decision for the largest prefix
    bucket <= shared_len; unbenched combinations default to cascade-on.
    """
    ov = override()
    if ov is not None:
        return ov == "on"
    if not _decisions or shared_len <= 0:
        return True
    bucket = None
    for pb in _prefix_buckets:
        if pb <= shared_len:
            bucket = pb
        else:
            break
    if bucket is None:
        return True
    return _decisions.get((num_tokens, bucket), True)


# ----------------------------------------------------------------- benching


def _prefix_grid(min_prefix: int, max_model_len: int) -> list:
    """Shared-prefix buckets to bench: MIN_PREFIX x {1,2,4}, capped so a
    probed sequence (shared + suffix + generation) still fits the context."""
    cap = max_model_len - min_prefix
    grid = []
    for mult in (1, 2, 4):
        p = min_prefix * mult
        if p <= cap and p not in grid:
            grid.append(p)
    return grid


def _time_fn(fn, iters=10, warmup=3, repeats=2) -> float:
    best = None
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.npu.synchronize()
        us = start.elapsed_time(end) * 1000.0 / iters
        best = us if best is None else min(best, us)
    return best


def _bench_cell(
    num_tokens: int,
    shared: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    scale: float,
    block_size: int,
    device: str,
):
    """Bench one (N, shared) cell; returns (cascade_us, full_us)."""
    import ascend_kernel  # noqa: F401

    suffix_blocks = (SUFFIX + block_size - 1) // block_size
    total = shared + SUFFIX
    nblk_total = (total + block_size - 1) // block_size
    sb1 = shared // block_size

    def blocks_for(nbytes):
        free, _ = torch.npu.mem_get_info()
        if nbytes * 1.25 > free:
            raise MemoryError(
                f"cascade gate: {nbytes / 2**30:.1f}GiB probe pool exceeds "
                f"free HBM ({free / 2**30:.1f}GiB)"
            )

    # Per-request DISJOINT suffix blocks (no reuse: W0 hazard #3).
    blocks_for(2 * nblk_total * num_tokens * block_size * head_size * 2)
    k_pool = (torch.randn(nblk_total * num_tokens, block_size, num_kv_heads,
                          head_size, device=device) * 0.5).to(torch.bfloat16)
    v_pool = (torch.randn(nblk_total * num_tokens, block_size, num_kv_heads,
                          head_size, device=device) * 0.5).to(torch.bfloat16)
    k_pool = k_pool.reshape(nblk_total * num_tokens, block_size, -1).contiguous()
    v_pool = v_pool.reshape(nblk_total * num_tokens, block_size, -1).contiguous()
    q = (torch.randn(num_tokens, num_heads, head_size, device=device)
         * 0.5).to(torch.bfloat16)

    qlen_cum = list(range(1, num_tokens + 1))

    def cumsum(vals):
        out, acc = [], 0
        for v in vals:
            acc += v
            out.append(acc)
        return out

    # ---- full path (MUST run before any stage-1 call: W0 hazard #1) ----
    # Per-request DISJOINT blocks — block reuse in TND probes is an aicore
    # fault risk at ANY kv magnitude (W0 hazards #2/#3), and a faulting probe
    # kills the engine (uncatchable), so the bench must be fault-free by
    # construction.
    bt_full = torch.stack([
        torch.arange(nblk_total, dtype=torch.int32, device=device)
        + r * nblk_total for r in range(num_tokens)])
    ws_full = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
        query=q, key=k_pool, value=v_pool,
        block_table=bt_full,
        input_layout="TND", block_size=block_size,
        actual_seq_qlen=qlen_cum,
        actual_seq_kvlen=cumsum([total] * num_tokens),
        num_key_value_heads=num_kv_heads, softmax_scale=scale,
        num_query_heads=num_heads)
    out_full = torch.empty(num_tokens, num_heads, head_size,
                           dtype=torch.bfloat16, device=device)

    def full_call():
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=q, key=k_pool, value=v_pool,
            block_table=bt_full,
            input_layout="TND", block_size=block_size,
            actual_seq_qlen=qlen_cum,
            actual_seq_kvlen=cumsum([total] * num_tokens),
            num_key_value_heads=num_kv_heads, num_query_heads=num_heads,
            softmax_scale=scale, workspace=ws_full, out=(out_full,
                                                         torch.empty(
                                                             1, dtype=torch.bfloat16,
                                                             device=device)))
        return out_full

    t_full = _time_fn(full_call)

    # ---- cascade path (first fa_fp32_stage1 in this process happens here) ----
    bt_s1 = torch.arange(sb1, dtype=torch.int32, device=device).unsqueeze(0)
    bt_suf = torch.stack([
        torch.arange(suffix_blocks, dtype=torch.int32, device=device)
        + r * suffix_blocks for r in range(num_tokens)])
    suf_off = sb1 * num_tokens
    k_suf = k_pool[suf_off:suf_off + suffix_blocks * num_tokens]
    v_suf = v_pool[suf_off:suf_off + suffix_blocks * num_tokens]
    o2 = torch.empty(num_tokens, num_heads, head_size, dtype=torch.bfloat16,
                     device=device)
    l2 = torch.empty(1, dtype=torch.bfloat16, device=device)
    suf_lens = [SUFFIX] * num_tokens
    ws_s2 = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
        query=q, key=k_suf, value=v_suf,
        block_table=bt_suf, input_layout="TND", block_size=block_size,
        actual_seq_qlen=qlen_cum, actual_seq_kvlen=cumsum(suf_lens),
        num_key_value_heads=num_kv_heads, softmax_scale=scale,
        num_query_heads=num_heads)

    def cascade_call():
        # stage-1: flattened T=N over the SHARED prefix blocks (pool rows
        # [0, sb1) — shared across requests, matching bt_shared = bt[:1,:sb]).
        s1_out, s1_lse = torch.ops.npu.fa_fp32_stage1(
            q,
            k_pool[:sb1].view(sb1, block_size, num_kv_heads, head_size),
            v_pool[:sb1].view(sb1, block_size, num_kv_heads, head_size),
            bt_s1,
            torch.tensor([num_tokens], dtype=torch.int64, device=device),
            torch.tensor([shared], dtype=torch.int64, device=device),
            num_tokens)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=q, key=k_suf, value=v_suf,
            block_table=bt_suf, input_layout="TND", block_size=block_size,
            actual_seq_qlen=qlen_cum, actual_seq_kvlen=cumsum(suf_lens),
            num_key_value_heads=num_kv_heads, num_query_heads=num_heads,
            softmax_scale=scale, workspace=ws_s2, out=(o2, l2))
        torch.ops.npu.lse_merge(
            s1_out, o2.reshape(num_tokens, num_heads, head_size),
            s1_lse, l2.reshape(-1).float(), 2)

    t_cascade = _time_fn(cascade_call)

    return t_cascade, t_full


def bench_all(runner, batch_descriptors, block_size: int) -> float:
    """Bench every (uniform bucket, prefix bucket) cell; fill decisions.

    Called on the runner between the standard FULL captures and the cascade
    twin captures.  Returns the wall duration; any failure leaves the
    decision table empty (gate neutral = cascade-on everywhere).
    """
    global _bench_duration_s
    import time as _time

    start = _time.perf_counter()
    with _lock:
        _decisions.clear()
        _prefix_buckets.clear()

    if override() is not None:
        logger.info("cascade gate: override=%s skips bench", override())
        return 0.0

    cfg = runner.vllm_config.model_config
    parallel_config = runner.vllm_config.parallel_config
    num_heads = cfg.get_num_attention_heads(parallel_config)
    num_kv_heads = cfg.get_num_kv_heads(parallel_config)
    head_size = cfg.get_head_size()
    scale = 1.0 / (head_size ** 0.5)
    max_model_len = getattr(runner, "max_model_len", 0) or 1
    min_prefix = int(os.getenv("VLLM_ASCEND_CASCADE_MIN_PREFIX", "8192"))
    grid = _prefix_grid(min_prefix, max_model_len)
    if not grid or block_size <= 0:
        logger.info("cascade gate: no prefix buckets to bench (grid=%s)", grid)
        return 0.0

    buckets = sorted({
        getattr(d, "num_tokens", 0)
        for d in batch_descriptors
        if getattr(d, "uniform", False) and getattr(d, "num_tokens", 0) > 0
    })
    if not buckets:
        return 0.0

    device = "npu"
    decisions: dict = {}
    for num_tokens in buckets:
        for shared in grid:
            try:
                t_cas, t_full = _bench_cell(
                    num_tokens, shared, num_heads, num_kv_heads, head_size,
                    scale, block_size, device)
            except Exception as exc:  # noqa: BLE001
                # Recoverable op errors (tiling rejection, OOM) skip the cell
                # -> default on.  A device fault is NOT recoverable and will
                # take the engine down either way; the probe forms above are
                # the W0-validated fault-free combination (disjoint blocks,
                # full-path probes before the first stage-1 call).
                logger.info(
                    "cascade gate: skip (N=%s, P=%s): %s -> default on",
                    num_tokens, shared, exc)
                continue
            decisions[(num_tokens, shared)] = (
                t_cas <= t_full * (1.0 - GATE_MARGIN))
            logger.info(
                "cascade gate: N=%s P=%s cascade=%.0fus full=%.0fus -> %s",
                num_tokens, shared, t_cas, t_full,
                "on" if decisions[(num_tokens, shared)] else "OFF")

    with _lock:
        _decisions.update(decisions)
        _prefix_buckets[:] = sorted(grid)
        _bench_duration_s = _time.perf_counter() - start
    logger.info("cascade gate: %d cells benched in %.2fs", len(decisions),
                _bench_duration_s)
    return _bench_duration_s
