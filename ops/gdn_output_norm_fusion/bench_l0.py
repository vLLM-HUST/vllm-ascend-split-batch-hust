#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.
"""L0 microbenchmark for GDN Part 3 Output Norm and Rearrange Fusion."""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401
import triton
from einops import rearrange

from vllm_ascend_split_batch.gdn_norm_gate_kernel import gdn_norm_gate_2d_kernel


def baseline_chain(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    z_shape_og: tuple[int, ...],
) -> torch.Tensor:
    c_2d = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z_2d = z.reshape(-1, z.shape[-1])
    var = c_2d.to(torch.float32).pow(2).mean(-1, keepdim=True)
    rsqrt = torch.rsqrt(var + 1e-6)
    c_normed = (c_2d.to(torch.float32) * rsqrt) * weight.to(torch.float32)
    c_2d = c_normed * torch.nn.functional.silu(z_2d.to(torch.float32))
    c_2d = c_2d.to(core_attn_out.dtype)
    c_3d = c_2d.reshape(z_shape_og)
    return rearrange(c_3d, "... h d -> ... (h d)")


def fused_output_norm(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    z_shape_og: tuple[int, ...],
) -> torch.Tensor:
    t, h, d = z_shape_og
    m = t * h
    n = d
    c_2d = core_attn_out.reshape(-1, d)
    z_2d = z.reshape(-1, d)
    out = torch.empty(
        (t, h * d), dtype=core_attn_out.dtype, device=core_attn_out.device
    )
    out_view = out.view(m, n)
    block_m = 32
    block_n = triton.next_power_of_2(n)
    grid = (triton.cdiv(m, block_m),)
    gdn_norm_gate_2d_kernel[grid](
        c_2d,
        z_2d,
        weight,
        out_view,
        c_2d.stride(0),
        c_2d.stride(1),
        z_2d.stride(0),
        z_2d.stride(1),
        out_view.stride(0),
        out_view.stride(1),
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        eps=1e-6,
    )
    return out


def benchmark(iters: int = 30) -> dict[str, float]:
    torch.manual_seed(42)
    device = torch.device("npu:0")
    t, h, d = 16, 4, 128
    z_shape = (t, h, d)
    core_attn_out = torch.randn(z_shape, dtype=torch.bfloat16, device=device)
    z = torch.randn(z_shape, dtype=torch.bfloat16, device=device)
    weight = torch.randn(d, dtype=torch.bfloat16, device=device)

    # L3 check
    ref = baseline_chain(core_attn_out, z, weight, z_shape)
    cand = fused_output_norm(core_attn_out, z, weight, z_shape)
    diff = (ref - cand).abs().max().item()
    assert diff == 0.0, f"L3 guard failed, diff={diff}"

    # Warmup
    for _ in range(5):
        _ = baseline_chain(core_attn_out, z, weight, z_shape)
        _ = fused_output_norm(core_attn_out, z, weight, z_shape)
    torch.npu.synchronize()

    repeat = 20
    g_base = torch.npu.NPUGraph()
    with torch.npu.graph(g_base):
        for _ in range(repeat):
            _ = baseline_chain(core_attn_out, z, weight, z_shape)

    g_cand = torch.npu.NPUGraph()
    with torch.npu.graph(g_cand):
        for _ in range(repeat):
            _ = fused_output_norm(core_attn_out, z, weight, z_shape)

    for _ in range(3):
        g_base.replay()
        g_cand.replay()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        g_base.replay()
    torch.npu.synchronize()
    t_base = ((time.perf_counter() - t0) * 1e6) / (iters * repeat)

    t0 = time.perf_counter()
    for _ in range(iters):
        g_cand.replay()
    torch.npu.synchronize()
    t_cand = ((time.perf_counter() - t0) * 1e6) / (iters * repeat)

    return {
        "baseline_us": t_base,
        "optimized_us": t_cand,
        "speedup": t_base / t_cand,
        "max_abs_diff": diff,
    }


if __name__ == "__main__":
    res = benchmark()
    print("gdn_output_norm_fusion result:", res)
