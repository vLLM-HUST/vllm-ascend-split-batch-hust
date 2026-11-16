#!/usr/bin/env python3
"""L0：补两个无法建模热点的 SOL 输入（GroupedMatmul / ChunkGatedDeltaRule）。

背景：整网 profile 里这两项合计占计算时间三成以上，但 `kernel_details` 只记
张量形状，不记每专家 token 数，也不记 chunk 算法参数，所以 roofline 的下界
算不出来。本基准在真实形状下把缺的两项测出来：

* ``GroupedMatmul``：用 topk 路由生成 ``group_list``，统计激活专家数，再按
  ``激活专家数 × K × N × dtype`` 算权重搬入下界，与设备时长对账；
* ``ChunkGatedDeltaRule``：扫 T，看时长是否随 T 线性；非算力/带宽项主导时，
  roofline 不适用，结论写 "受算法迭代与小步调度限制"。

峰值取自本机标定（``solcal-20261116/window.json``），不引用厂商标称值。
产物默认写 ``--out``，不落仓库。
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch_npu

PEAK_BW = 1088.7e9
PEAK_CUBE_BF16 = 290.6e12

# 与整网 profile 一致的 prefill 形状
E_DEFAULT = 256
TOPK_DEFAULT = 8
GMM_K = 2048
GMM_N = 512
GDN_HEADS = 8
GDN_DIM = 128


def timed(fn, iters: int, warmup: int) -> float:
    """Median device time in microseconds."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def group_list_from_topk(tokens: int, experts: int, topk: int, seed: int):
    """Return per-expert token counts and the cumsum the op expects."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=g)
    topk_ids = logits.topk(topk, dim=-1).indices.reshape(-1)
    counts = torch.bincount(topk_ids, minlength=experts).to(torch.int64)
    cumsum = torch.cumsum(counts, dim=0).to(torch.int64)
    return counts, cumsum.to("npu")


def bench_grouped_matmul(tokens: int, iters: int, warmup: int) -> dict:
    counts, cumsum = group_list_from_topk(tokens, E_DEFAULT, TOPK_DEFAULT, 42)
    active = int((counts > 0).sum().item())
    m_total = int(counts.sum().item())

    x = torch.randn(m_total, GMM_K, dtype=torch.bfloat16, device="npu")
    w = torch.randn(E_DEFAULT, GMM_K, GMM_N, dtype=torch.bfloat16, device="npu")

    def run():
        torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[w],
            split_item=2,
            group_type=0,
            group_list_type=0,
            group_list=cumsum,
        )

    us = timed(run, iters, warmup)

    weight_bytes = active * GMM_K * GMM_N * 2
    act_bytes = m_total * GMM_K * 2 + m_total * GMM_N * 2
    lower_us = (weight_bytes + act_bytes) / PEAK_BW * 1e6
    flops = 2.0 * m_total * GMM_K * GMM_N
    cube_us = flops / PEAK_CUBE_BF16 * 1e6
    return {
        "tokens": tokens,
        "experts": E_DEFAULT,
        "topk": TOPK_DEFAULT,
        "active_experts": active,
        "expanded_rows": m_total,
        "measured_us": round(us, 3),
        "weight_bytes": int(weight_bytes),
        "bandwidth_lower_us": round(lower_us, 3),
        "cube_lower_us": round(cube_us, 3),
        "sol_pct": round(100.0 * max(lower_us, cube_us) / us, 1),
        "implied_bw_gb_s": round((weight_bytes + act_bytes) / (us * 1e-6) / 1e9, 1),
    }


def bench_chunk_gdn(tokens: int, iters: int, warmup: int) -> dict:
    q = torch.randn(tokens, GDN_HEADS, GDN_DIM, dtype=torch.bfloat16, device="npu")
    k = torch.randn(tokens, GDN_HEADS, GDN_DIM, dtype=torch.bfloat16, device="npu")
    v = torch.randn(tokens, GDN_HEADS, GDN_DIM, dtype=torch.bfloat16, device="npu")
    beta = torch.rand(tokens, GDN_HEADS, dtype=torch.bfloat16, device="npu") + 0.1
    g = -torch.rand(tokens, GDN_HEADS, dtype=torch.float32, device="npu")
    state = torch.zeros(
        1, GDN_HEADS, GDN_DIM, GDN_DIM, dtype=torch.bfloat16, device="npu"
    )
    seq = torch.tensor([tokens], dtype=torch.int32, device="npu")
    scale = GDN_DIM**-0.5

    def run():
        torch_npu.npu_chunk_gated_delta_rule(
            q,
            k,
            v,
            beta=beta,
            initial_state=state,
            actual_seq_lengths=seq,
            scale=scale,
            g=g,
        )

    us = timed(run, iters, warmup)

    nbytes = (
        tokens * GDN_HEADS * GDN_DIM * 2 * 3  # q,k,v
        + tokens * GDN_HEADS * 4  # g
        + tokens * GDN_HEADS * 2  # beta
        + tokens * GDN_HEADS * GDN_DIM * 2  # o
        + GDN_HEADS * GDN_DIM * GDN_DIM * 2 * 2  # state in/out
    )
    lower_us = nbytes / PEAK_BW * 1e6
    return {
        "tokens": tokens,
        "measured_us": round(us, 3),
        "necessary_bytes": int(nbytes),
        "bandwidth_lower_us": round(lower_us, 3),
        "sol_pct": round(100.0 * lower_us / us, 1),
        "us_per_token": round(us / tokens, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(0)

    gmm = [bench_grouped_matmul(t, args.iters, args.warmup) for t in (8, 64, 256)]
    chunk = [
        bench_chunk_gdn(t, args.iters, args.warmup) for t in (256, 512, 1024, 1526)
    ]

    payload = {
        "schema": "sol-gap-bench/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.npu.get_device_name(0),
        "peaks": {
            "bandwidth_gb_s": PEAK_BW / 1e9,
            "cube_bf16_tflops": PEAK_CUBE_BF16 / 1e12,
            "source": "solcal-20261116/window.json",
        },
        "grouped_matmul": gmm,
        "chunk_gated_delta_rule": chunk,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(
        f"{'op':<24}{'T':>6}{'meas_us':>10}{'lower_us':>10}{'SOL%':>8}{'us/token':>10}"
    )
    for r in gmm:
        print(
            f"{'GroupedMatmul':<24}{r['tokens']:>6}{r['measured_us']:>10.1f}"
            f"{r['bandwidth_lower_us']:>10.1f}{r['sol_pct']:>8.1f}{'':>10}"
            f"  active_experts={r['active_experts']}"
        )
    for r in chunk:
        print(
            f"{'ChunkGatedDeltaRule':<24}{r['tokens']:>6}{r['measured_us']:>10.1f}"
            f"{r['bandwidth_lower_us']:>10.1f}{r['sol_pct']:>8.1f}{r['us_per_token']:>10.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
