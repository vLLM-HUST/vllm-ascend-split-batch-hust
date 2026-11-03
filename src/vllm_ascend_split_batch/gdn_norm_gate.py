# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_NORM_GATE_ENABLED = os.environ.get("VLLM_HUST_GDN_NORM_GATE", "0") == "1"


def gdn_norm_gate_fused(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Fused RMSNormGated (out = RMSNorm(x, weight) * silu(z)).

    Optimized for Qwen3.5 GDN Part 3 Output Projection under TP=2 Decode.
    Default-off guarded by VLLM_HUST_GDN_NORM_GATE=1.
    """
    if not _NORM_GATE_ENABLED:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + 1e-6)
        x_normed = (x.to(torch.float32) * rsqrt) * weight.to(torch.float32)
        out = x_normed * F.silu(z.to(torch.float32))
        return out.to(x.dtype)

    import triton

    from .gdn_norm_gate_kernel import gdn_norm_gate_2d_kernel

    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_M = 32
    BLOCK_N = triton.next_power_of_2(N)
    grid = (triton.cdiv(M, BLOCK_M),)

    gdn_norm_gate_2d_kernel[grid](
        x,
        z,
        weight,
        out,
        x.stride(0),
        x.stride(1),
        z.stride(0),
        z.stride(1),
        out.stride(0),
        out.stride(1),
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        eps=1e-6,
    )
    return out
