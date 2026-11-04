# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import os

_NORM_ENABLED = os.environ.get("VLLM_HUST_GDN_INPUT_NORM", "0") == "1"


def gdn_input_norm_fused(x, weight):
    """Fused RMSNorm for Qwen3.5 GDN in_proj decode phase [T=32, H=2048].

    Default-off guarded by VLLM_HUST_GDN_INPUT_NORM=1.
    """
    if not _NORM_ENABLED:
        import torch

        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(var + 1e-6)
        return (x.to(torch.float32) * rsqrt).to(x.dtype) * weight

    from .gdn_input_norm_kernel import rms_norm_kernel

    M, N = x.shape
    out = x.new_empty(x.shape)
    rms_norm_kernel[(M,)](
        x,
        weight,
        out,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        N=N,
        BLOCK_SIZE=N,
        eps=1e-6,
    )
    return out
