# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def gdn_norm_gate_2d_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_zm,
    stride_zn,
    stride_outm,
    stride_outn,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    row_mask = row_offs < M
    x_ptrs = x_ptr + row_offs[:, None] * stride_xm + col_offs[None, :] * stride_xn
    z_ptrs = z_ptr + row_offs[:, None] * stride_zm + col_offs[None, :] * stride_zn
    out_ptrs = (
        out_ptr + row_offs[:, None] * stride_outm + col_offs[None, :] * stride_outn
    )
    w_ptrs = w_ptr + col_offs[None, :]

    mask_2d = row_mask[:, None]

    x = tl.load(x_ptrs, mask=mask_2d, other=0.0)
    z = tl.load(z_ptrs, mask=mask_2d, other=0.0)
    w = tl.load(w_ptrs)

    x_f32 = x.to(tl.float32)
    z_f32 = z.to(tl.float32)
    w_f32 = w.to(tl.float32)

    var = tl.sum(x_f32 * x_f32, axis=1, keep_dims=True) * (1.0 / N)
    rsqrt = tl.math.rsqrt(var + eps)
    x_normed = (x_f32 * rsqrt) * w_f32

    silu_z = z_f32 * tl.sigmoid(z_f32)
    out = x_normed * silu_z

    tl.store(out_ptrs, out.to(x.dtype), mask=mask_2d)
