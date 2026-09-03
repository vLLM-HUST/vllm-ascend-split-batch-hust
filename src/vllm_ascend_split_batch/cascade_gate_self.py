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

"""Subprocess entry point for the W2 cascade gate micro-bench.

Run as ``python3 -m vllm_ascend_split_batch.cascade_gate_self SPEC OUT``.

Isolated from the serving engine on purpose: the probes exercise CANN FIA /
custom-operator combinations with known aicore-fault corners (W0 report
sec.3), and a faulting probe is uncatchable — in-engine it would take the
engine down, here it only degrades the gate to "neutral" (the parent reads
no output file and defaults every bucket to cascade-on).

Two-pass ordering inside the child (W0 hazard #1: after the first
``fa_fp32_stage1`` call, eager FIA v2 with kv >= ~8k can fault in the same
process):
  pass 1 — every full-path probe (no stage-1 has run yet)
  pass 2 — every cascade probe (stage-1 + stage-2 + lse_merge; only small
           eager FIA afterwards, which is the proven-safe combination)

Block tables are per-request DISJOINT in every probe (W0 hazards #2/#3:
block reuse faults at any kv magnitude under cumulative lens).
"""

import json
import sys

import ascend_kernel  # noqa: F401
import torch
import torch_npu  # noqa: F401


def _cumsum(vals):
    out, acc = [], 0
    for v in vals:
        acc += v
        out.append(acc)
    return out


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


def _run_cell_full(spec, num_tokens, shared, k_pool, v_pool, q):
    """Single full-KV FIA v2 probe (production .out + cached workspace)."""
    block_size = spec["block_size"]
    suffix = spec["suffix"]
    total = shared + suffix
    nblk_total = (total + block_size - 1) // block_size
    qlen_cum = _cumsum(range(1, num_tokens + 1))
    kvlen_cum = _cumsum([total] * num_tokens)
    bt = torch.stack([
        torch.arange(nblk_total, dtype=torch.int32, device="npu")
        + r * nblk_total for r in range(num_tokens)])
    ws = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
        query=q, key=k_pool, value=v_pool, block_table=bt,
        input_layout="TND", block_size=block_size,
        actual_seq_qlen=qlen_cum, actual_seq_kvlen=kvlen_cum,
        num_key_value_heads=spec["num_kv_heads"],
        softmax_scale=spec["scale"], num_query_heads=spec["num_heads"])
    out = torch.empty(num_tokens, spec["num_heads"], spec["head_size"],
                      dtype=torch.bfloat16, device="npu")

    def call():
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=q, key=k_pool, value=v_pool, block_table=bt,
            input_layout="TND", block_size=block_size,
            actual_seq_qlen=qlen_cum, actual_seq_kvlen=kvlen_cum,
            num_key_value_heads=spec["num_kv_heads"],
            num_query_heads=spec["num_heads"], softmax_scale=spec["scale"],
            workspace=ws, out=(out, torch.empty(1, dtype=torch.bfloat16,
                                                device="npu")))
        return out

    return _time_fn(call)


def _run_cell_cascade(spec, num_tokens, shared, k_pool, v_pool, q):
    """Two-stage + merge probe (mirrors the twin's re-bind forms)."""
    block_size = spec["block_size"]
    suffix = spec["suffix"]
    sb1 = shared // block_size
    suffix_blocks = (suffix + block_size - 1) // block_size
    qlen_cum = _cumsum(range(1, num_tokens + 1))
    suf_lens = [suffix] * num_tokens
    bt_s1 = torch.arange(sb1, dtype=torch.int32, device="npu").unsqueeze(0)
    bt_suf = torch.stack([
        torch.arange(suffix_blocks, dtype=torch.int32, device="npu")
        + r * suffix_blocks for r in range(num_tokens)])
    suf_off = sb1 * num_tokens
    k_suf = k_pool[suf_off:suf_off + suffix_blocks * num_tokens]
    v_suf = v_pool[suf_off:suf_off + suffix_blocks * num_tokens]
    o2 = torch.empty(num_tokens, spec["num_heads"], spec["head_size"],
                     dtype=torch.bfloat16, device="npu")
    ws_s2 = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
        query=q, key=k_suf, value=v_suf, block_table=bt_suf,
        input_layout="TND", block_size=block_size,
        actual_seq_qlen=qlen_cum, actual_seq_kvlen=_cumsum(suf_lens),
        num_key_value_heads=spec["num_kv_heads"],
        softmax_scale=spec["scale"], num_query_heads=spec["num_heads"])

    def call():
        s1_out, s1_lse = torch.ops.npu.fa_fp32_stage1(
            q,
            k_pool[:sb1].view(sb1, block_size, spec["num_kv_heads"],
                              spec["head_size"]),
            v_pool[:sb1].view(sb1, block_size, spec["num_kv_heads"],
                              spec["head_size"]),
            bt_s1,
            torch.tensor([num_tokens], dtype=torch.int64, device="npu"),
            torch.tensor([shared], dtype=torch.int64, device="npu"),
            num_tokens)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=q, key=k_suf, value=v_suf, block_table=bt_suf,
            input_layout="TND", block_size=block_size,
            actual_seq_qlen=qlen_cum, actual_seq_kvlen=_cumsum(suf_lens),
            num_key_value_heads=spec["num_kv_heads"],
            num_query_heads=spec["num_heads"], softmax_scale=spec["scale"],
            workspace=ws_s2, out=(o2, torch.empty(1, dtype=torch.bfloat16,
                                                  device="npu")))
        torch.ops.npu.lse_merge(
            s1_out, o2.reshape(num_tokens, spec["num_heads"],
                               spec["head_size"]),
            s1_lse, torch.empty(num_tokens * spec["num_heads"],
                                dtype=torch.float32, device="npu"), 2)

    return _time_fn(call)


def main() -> int:
    spec_path, out_path = sys.argv[1], sys.argv[2]
    with open(spec_path) as fh:
        spec = json.load(fh)

    torch.npu.set_device(0)
    block_size = spec["block_size"]
    suffix = spec["suffix"]
    grid = spec["prefix_buckets"]
    buckets = spec["batch_buckets"]

    # Largest pool first-fit: allocate once for the biggest cell and slice
    # per cell (per-request disjoint rows always index within the pool).
    max_nblk = max((shared + suffix + block_size - 1) // block_size
                   for shared in grid)
    max_blocks = max_nblk * max(buckets)
    free, _total = torch.npu.mem_get_info()
    need = 2 * max_blocks * block_size * spec["head_size"] * 2
    if need * 1.25 > free:
        # Not enough HBM next to the serving engine: give up (gate neutral).
        with open(out_path, "w") as fh:
            json.dump({"error": f"insufficient HBM: need {need / 2**30:.1f}GiB, "
                                f"free {free / 2**30:.1f}GiB"}, fh)
        return 0

    k_pool = (torch.randn(max_blocks, block_size, spec["num_kv_heads"],
                          spec["head_size"], device="npu")
              * 0.5).to(torch.bfloat16)
    v_pool = (torch.randn(max_blocks, block_size, spec["num_kv_heads"],
                          spec["head_size"], device="npu")
              * 0.5).to(torch.bfloat16)
    k_pool = k_pool.reshape(max_blocks, block_size, -1).contiguous()
    v_pool = v_pool.reshape(max_blocks, block_size, -1).contiguous()
    q = (torch.randn(max(buckets), spec["num_heads"], spec["head_size"],
                     device="npu") * 0.5).to(torch.bfloat16)

    results = {}
    try:
        # pass 1: full path everywhere (clean — no stage-1 yet)
        for num_tokens in buckets:
            for shared in grid:
                key = f"{num_tokens}:{shared}"
                nblk_total = (shared + suffix + block_size - 1) // block_size
                k_view = k_pool[:nblk_total * num_tokens]
                v_view = v_pool[:nblk_total * num_tokens]
                q_view = q[:num_tokens]
                results[key] = {
                    "full": _run_cell_full(spec, num_tokens, shared,
                                           k_view, v_view, q_view)}
        # pass 2: cascade path (stage-1 from here on — small eager FIA only)
        for num_tokens in buckets:
            for shared in grid:
                key = f"{num_tokens}:{shared}"
                nblk_total = (shared + suffix + block_size - 1) // block_size
                k_view = k_pool[:nblk_total * num_tokens]
                v_view = v_pool[:nblk_total * num_tokens]
                q_view = q[:num_tokens]
                results[key]["cascade"] = _run_cell_cascade(
                    spec, num_tokens, shared, k_view, v_view, q_view)
    except Exception as exc:  # recoverable probe error: report and stop
        results["error"] = repr(exc)

    with open(out_path, "w") as fh:
        json.dump(results, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
