#!/usr/bin/env python3
"""L0 microbenchmark for Qwen Text Attention Split-QKV RoPE Fusion."""

from __future__ import annotations

import time

import torch
import torch_npu  # noqa: F401
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def benchmark_split_qkv_fuse(iters: int = 50) -> dict[str, float]:
    init_device_properties_triton()
    device = torch.device("npu:0")
    num_tokens = 16
    num_q_heads = 8
    num_kv_heads = 2
    head_size = 128
    rotary_dim = 64
    eps = 1e-6
    has_gate = True

    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    total_dim = q_size * 2 + kv_size * 2

    torch.manual_seed(42)
    qkv = torch.randn(num_tokens, total_dim, dtype=torch.bfloat16, device=device)
    positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    q_weight = torch.randn(head_size, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(head_size, dtype=torch.bfloat16, device=device)

    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(4096, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos_sin_cache = torch.cat(
        (freqs.cos().to(torch.bfloat16), freqs.sin().to(torch.bfloat16)), dim=-1
    )

    cos_sin_1d = cos_sin_cache[positions]
    cos_sin_3d = cos_sin_1d.unsqueeze(0).repeat(3, 1, 1).contiguous()

    # Stock graph
    g_stock = torch.npu.NPUGraph()
    with torch.npu.graph(g_stock):
        for _ in range(50):
            q_gate, k, v = qkv.split([q_size * 2, kv_size, kv_size], dim=-1)
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, num_q_heads, -1)
            q_stock, gate_stock = torch.chunk(q_gate, 2, dim=-1)
            q_stock = q_stock.reshape(*orig_shape, -1)
            gate_stock = gate_stock.reshape(*orig_shape, -1)
            q_norm, _ = torch_npu.npu_rms_norm(
                q_stock.view(-1, num_q_heads, head_size), 1.0 + q_weight, eps
            )
            k_norm, _ = torch_npu.npu_rms_norm(
                k.view(-1, num_kv_heads, head_size), 1.0 + k_weight, eps
            )
            q_roped, k_roped = torch_npu.npu_mrope(
                positions,
                q_norm.contiguous().view(num_tokens, -1),
                k_norm.contiguous().view(num_tokens, -1),
                cos_sin_cache,
                head_size,
                mrope_section=[0, 0, 0],
                rotary_mode="half",
                cache_mode="default",
            )

    # Fused graph
    g_fused = torch.npu.NPUGraph()
    with torch.npu.graph(g_fused):
        for _ in range(50):
            _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
                qkv=qkv,
                q_weight=1.0 + q_weight,
                k_weight=1.0 + k_weight,
                cos_sin=cos_sin_3d,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                eps=eps,
                mrope_section=[rotary_dim // 2, 0, 0],
                is_interleaved=False,
                rope_dim=rotary_dim,
                has_gate=has_gate,
            )

    for _ in range(10):
        g_stock.replay()
        g_fused.replay()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        g_stock.replay()
    torch.npu.synchronize()
    t_stock = (time.perf_counter() - t0) / (iters * 50) * 1e6

    t0 = time.perf_counter()
    for _ in range(iters):
        g_fused.replay()
    torch.npu.synchronize()
    t_fused = (time.perf_counter() - t0) / (iters * 50) * 1e6

    return {
        "baseline_us": t_stock,
        "optimized_us": t_fused,
        "speedup": t_stock / t_fused,
    }


if __name__ == "__main__":
    res = benchmark_split_qkv_fuse()
    print("split_qkv_fuse result:", res)
