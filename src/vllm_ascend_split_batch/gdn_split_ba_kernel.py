# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def gdn_split_ba_2d_kernel(
    ba_ptr,
    b_ptr,
    a_ptr,
    stride_bam,
    stride_ban,
    stride_bm,
    stride_bn,
    stride_am,
    stride_an,
    M: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)
    row_mask = row_offs < M

    b_in_ptrs = ba_ptr + row_offs[:, None] * stride_bam + col_offs[None, :] * stride_ban
    a_in_ptrs = (
        ba_ptr
        + row_offs[:, None] * stride_bam
        + (col_offs[None, :] + HALF) * stride_ban
    )

    b_out_ptrs = b_ptr + row_offs[:, None] * stride_bm + col_offs[None, :] * stride_bn
    a_out_ptrs = a_ptr + row_offs[:, None] * stride_am + col_offs[None, :] * stride_an

    mask_2d = row_mask[:, None]

    val_b = tl.load(b_in_ptrs, mask=mask_2d, other=0.0)
    val_a = tl.load(a_in_ptrs, mask=mask_2d, other=0.0)

    tl.store(b_out_ptrs, val_b, mask=mask_2d)
    tl.store(a_out_ptrs, val_a, mask=mask_2d)
