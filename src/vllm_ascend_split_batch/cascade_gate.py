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
Cells at L2-scale prefix buckets (<= SMALL_PREFIX_MAX) must clear a larger,
e2e-calibrated margin (SMALL_PREFIX_MARGIN); see the constant's comment for
the disjoint-table blind spot.

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
  VLLM_ASCEND_CASCADE_GATE_SMALLP_MARGIN=<0..1>  small-prefix bucket margin
  VLLM_ASCEND_CASCADE_GATE_SMALLP_MAXPREFIX=<N>  small-prefix bucket bound
"""

from __future__ import annotations

import logging
import os
import threading

import torch_npu  # noqa: F401  (device init happens in the runner process)

logger = logging.getLogger(__name__)

ENV_GATE = "VLLM_ASCEND_CASCADE_ADAPTIVE_GATE"
ENV_OVERRIDE = "VLLM_ASCEND_CASCADE_GATE_OVERRIDE"

# Cascade must be at least this much faster to be chosen (noise hysteresis).
GATE_MARGIN = 0.02
# e2e-calibrated small-prefix guard (v1-ev3 G6): the probes run per-request
# DISJOINT block tables (W0 hazards #2/#3), i.e. they measure the
# no-prefix-cache world, while the engine serves with enable_prefix_caching
# where the production FULL path reuses the shared KV blocks across requests.
# At L2-scale shared prefixes (4096 tokens ~ 16.8 MiB K+V at the tested 14B
# geometry) that reuse erases cascade's traffic advantage, leaving only the
# engine-path overheads -> measured e2e LOSS (+6.7% at (64,4096)) despite a
# +10.7% bench margin.  At >=8k the sign never flipped (4 cells, v1-ev3-rerun
# section 8).  A uniform margin cannot separate (64,4096)[bench +10.7% -> e2e
# loss] from (32,8192)[bench +10.9% -> e2e win], so the enhanced margin is
# keyed on the prefix bucket itself.  Fires only when MIN_PREFIX is lowered
# into the 4k region; the default MIN_PREFIX=8192 never benches such a bucket.
SMALL_PREFIX_MAX = 4096
SMALL_PREFIX_MARGIN = 0.25
ENV_SMALLP_MARGIN = "VLLM_ASCEND_CASCADE_GATE_SMALLP_MARGIN"
ENV_SMALLP_MAXPREFIX = "VLLM_ASCEND_CASCADE_GATE_SMALLP_MAXPREFIX"
# Representative per-request suffix for the probes (2 blocks of 128): it
# approximates the p420 shape at generation end (prompt tail ~105 + 128 gen),
# and an oversized suffix can only depress the bench margin (conservative
# direction), so do not shrink it without re-calibrating against e2e.
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


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        val = float(os.getenv(name, ""))
    except ValueError:
        return default
    return val if lo <= val <= hi else default


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

    Override env wins; then the benched decision for the SMALLEST prefix
    bucket >= shared_len (ceil).  Rationale: the runtime shared length is
    block-aligned prefix-cache coverage and runs slightly below the nominal
    prefix (a ~8.2k-token shared prefix reports 7936 = 62 blocks); flooring
    would route those steps into the smaller bucket's verdict and flag them
    OFF even where the e2e matrix says cascade wins.  Ceil errs toward the
    larger prefix, where cascade's advantage only grows.  Unbenched
    combinations default to cascade-on.
    """
    ov = override()
    if ov is not None:
        return ov == "on"
    if not _decisions or shared_len <= 0:
        return True
    bucket = None
    for pb in _prefix_buckets:
        if pb >= shared_len:
            bucket = pb
            break
    if bucket is None:
        # above the largest benched prefix: cascade keeps winning there
        return True
    return _decisions.get((num_tokens, bucket), True)


def _margin_for(shared: int) -> float:
    """Margin a benched cell must clear at this shared-prefix bucket."""
    try:
        small_max = int(os.getenv(ENV_SMALLP_MAXPREFIX, str(SMALL_PREFIX_MAX)))
    except ValueError:
        small_max = SMALL_PREFIX_MAX
    if small_max > 0 and shared <= small_max:
        return _env_float(ENV_SMALLP_MARGIN, SMALL_PREFIX_MARGIN, 0.0, 1.0)
    return GATE_MARGIN


def _verdict(cascade_us: float, full_us: float, shared: int) -> bool:
    """True when cascade clears the bucket margin (same <= as before)."""
    return cascade_us <= full_us * (1.0 - _margin_for(shared))


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


def bench_all(runner, batch_descriptors, block_size: int) -> float:
    """Bench every (uniform bucket, prefix bucket) cell; fill decisions.

    The probes run in an ISOLATED SUBPROCESS
    (``python3 -m vllm_ascend_split_batch.cascade_gate_self``): they exercise
    CANN FIA / custom-op corners with known aicore-fault modes (W0 report
    sec.3) and a faulting probe is uncatchable — in-engine it would kill the
    serving process, in the child it only degrades the gate to neutral.

    Returns the wall duration; any failure leaves the decision table empty
    (gate neutral = cascade-on everywhere).
    """
    global _bench_duration_s
    import json
    import subprocess
    import sys
    import tempfile
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
    max_model_len = getattr(runner, "max_model_len", 0) or 1
    min_prefix = int(os.getenv("VLLM_ASCEND_CASCADE_MIN_PREFIX", "8192"))
    grid = _prefix_grid(min_prefix, max_model_len)
    buckets = sorted({
        getattr(d, "num_tokens", 0)
        for d in batch_descriptors
        if getattr(d, "uniform", False) and getattr(d, "num_tokens", 0) > 0
    })
    if not grid or not buckets or block_size <= 0:
        print(f"[cas-gate] nothing to bench (grid={grid} buckets={buckets})",
              flush=True)
        return 0.0

    spec = {
        "prefix_buckets": grid,
        "batch_buckets": buckets,
        "block_size": block_size,
        "suffix": SUFFIX,
        "num_heads": cfg.get_num_attention_heads(parallel_config),
        "num_kv_heads": cfg.get_num_kv_heads(parallel_config),
        "head_size": cfg.get_head_size(),
        "scale": 1.0 / (cfg.get_head_size() ** 0.5),
    }
    tmpdir = tempfile.mkdtemp(prefix="cas-gate-")
    spec_path = os.path.join(tmpdir, "spec.json")
    out_path = os.path.join(tmpdir, "results.json")
    with open(spec_path, "w") as fh:
        json.dump(spec, fh)

    proc = subprocess.run(
        [sys.executable, "-m",
         "vllm_ascend_split_batch.cascade_gate_self", spec_path, out_path],
        capture_output=True, text=True, timeout=300,
        env=os.environ.copy())
    duration = _time.perf_counter() - start

    results = {}
    if proc.returncode != 0:
        print(f"[cas-gate] bench subprocess failed rc={proc.returncode}; "
              "gate neutral", flush=True)
        if proc.stderr:
            print("[cas-gate] child stderr tail: "
                  f"{proc.stderr.strip()[-400:]}", flush=True)
        return duration
    try:
        with open(out_path) as fh:
            results = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"[cas-gate] bench results unreadable ({exc}); gate neutral",
              flush=True)
        return duration

    for num_tokens in buckets:
        for shared in grid:
            cell = results.get(f"{num_tokens}:{shared}")
            if not isinstance(cell, dict) or "cascade" not in cell \
                    or "full" not in cell:
                continue
            margin = _margin_for(shared)
            verdict = cell["cascade"] <= cell["full"] * (1.0 - margin)
            _decisions[(num_tokens, shared)] = verdict
            print(f"[cas-gate] N={num_tokens} P={shared} "
                  f"cascade={cell['cascade']:.0f}us full={cell['full']:.0f}us "
                  f"margin={margin:.0%} "
                  f"-> {'on' if verdict else 'OFF'}", flush=True)

    with _lock:
        _prefix_buckets[:] = sorted(grid)
        _bench_duration_s = duration
    print(f"[cas-gate] {len(_decisions)} cells benched in {duration:.2f}s",
          flush=True)
    return duration
