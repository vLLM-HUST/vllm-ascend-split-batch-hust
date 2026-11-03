# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    stride_xm,
    stride_xn,
    stride_outm,
    stride_outn,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    eps: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + row_idx * stride_xm + cols * stride_xn)
    w = tl.load(weight_ptr + cols)

    x_f32 = x.to(tl.float32)
    w_f32 = w.to(tl.float32)

    sum_sq = tl.sum(x_f32 * x_f32, axis=0)
    var = sum_sq * (1.0 / N)
    rsqrt = tl.rsqrt(var + eps)

    # Cast to intermediate bfloat16 to match reference numerical semantics exactly
    out = (x_f32 * rsqrt).to(tl.bfloat16).to(tl.float32) * w_f32
    tl.store(out_ptr + row_idx * stride_outm + cols * stride_outn, out.to(tl.bfloat16))
