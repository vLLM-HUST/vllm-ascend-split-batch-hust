# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

"""Fused RMSNormGated wrapper.

No module in this package imports this file: the wrapper is a parked route, not a
live dispatch target. Its delivery state is owned by the ``gdn_norm_gate`` entry in
``DELIVERY_INDEX.json``, not by this docstring.

Shape contract of the enabled path: the kernel masks the row axis only, so the
reduced width must be exactly ``next_power_of_2(N)`` (i.e. ``N`` must be a power
of two). A padded column block is refused here instead of being launched.
"""

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

    The enabled path requires ``x.shape[1]`` to be a power of two (the kernel
    masks rows only); anything else raises ``ValueError`` instead of launching.
    """
    if not _NORM_GATE_ENABLED:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + 1e-6)
        x_normed = (x.to(torch.float32) * rsqrt) * weight.to(torch.float32)
        out = x_normed * F.silu(z.to(torch.float32))
        return out.to(x.dtype)

    import triton

    M, N = x.shape
    BLOCK_N = triton.next_power_of_2(N)
    if BLOCK_N != N:
        # The kernel masks rows only: its column range is ``tl.arange(0,
        # BLOCK_N)`` and the weight tile is loaded unmasked, so a padded column
        # block would read past the row and write past ``out``. Only an exact
        # power of two is in scope; extending it would need a column mask.
        raise ValueError(
            f"GDN fused norm-gate requires N to be a power of two (got N={N}, "
            f"BLOCK_N={BLOCK_N})"
        )

    from .gdn_norm_gate_kernel import gdn_norm_gate_2d_kernel

    out = torch.empty_like(x)
    BLOCK_M = 32
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
        M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        eps=1e-6,
    )
    return out
