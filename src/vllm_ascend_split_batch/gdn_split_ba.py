# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import os

import torch

_SPLIT_BA_ENABLED = os.environ.get("VLLM_HUST_GDN_SPLIT_BA", "0") == "1"


def gdn_split_ba_fused(ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused split_ba for Qwen3.5 GDN Decode phase under TP=2.

    Splits [T, 2 * H_v] into contiguous (b, a) tensors [T, H_v].
    Default-off guarded by VLLM_HUST_GDN_SPLIT_BA=1.
    """
    if not _SPLIT_BA_ENABLED:
        b, a = ba.chunk(2, dim=-1)
        return b.contiguous(), a.contiguous()

    import triton

    from .gdn_split_ba_kernel import gdn_split_ba_2d_kernel

    M, TotalDim = ba.shape
    Half = TotalDim // 2
    b = torch.empty((M, Half), dtype=ba.dtype, device=ba.device)
    a = torch.empty((M, Half), dtype=ba.dtype, device=ba.device)

    BLOCK_M = 32
    BLOCK_N = Half
    grid = (triton.cdiv(M, BLOCK_M),)

    gdn_split_ba_2d_kernel[grid](
        ba,
        b,
        a,
        ba.stride(0),
        ba.stride(1),
        b.stride(0),
        b.stride(1),
        a.stride(0),
        a.stride(1),
        M=M,
        HALF=Half,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return b, a
