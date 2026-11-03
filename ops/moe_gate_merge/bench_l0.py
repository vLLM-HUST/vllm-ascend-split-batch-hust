#!/usr/bin/env python3
"""L0 microbenchmark for moe_gate_merge (MoE router + shared gate merge)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401


def benchmark_moe_gate_merge(iters: int = 200) -> dict[str, float]:
    device = "npu:0"
    m_tokens = 32
    hidden = 2048
    n_router = 256
    n_shared = 1

    torch.manual_seed(42)
    x = torch.randn(m_tokens, hidden, dtype=torch.bfloat16, device=device)
    w_router = torch.randn(n_router, hidden, dtype=torch.bfloat16, device=device)
    w_shared = torch.randn(n_shared, hidden, dtype=torch.bfloat16, device=device)
    w_merged = torch.cat([w_router, w_shared], dim=0)

    # 1. Capture Separate MatMuls
    g_sep = torch.npu.NPUGraph()
    with torch.npu.graph(g_sep):
        _ = F.linear(x, w_router)
        _ = F.linear(x, w_shared)

    # 2. Capture Merged + Slice Contiguous
    g_merge_contig = torch.npu.NPUGraph()
    with torch.npu.graph(g_merge_contig):
        om = F.linear(x, w_merged)
        _ = om[:, :n_router].contiguous()
        _ = om[:, n_router:].contiguous()

    # Warmup
    for _ in range(20):
        g_sep.replay()
        g_merge_contig.replay()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        g_sep.replay()
    end.record()
    torch.npu.synchronize()
    sep_us = start.elapsed_time(end) * 1000.0 / iters

    start.record()
    for _ in range(iters):
        g_merge_contig.replay()
    end.record()
    torch.npu.synchronize()
    merge_us = start.elapsed_time(end) * 1000.0 / iters

    return {
        "baseline_us": sep_us,
        "optimized_us": merge_us,
        "speedup": sep_us / merge_us if merge_us > 0 else 0.0,
    }


if __name__ == "__main__":
    res = benchmark_moe_gate_merge()
    print("moe_gate_merge benchmark result:", res)
