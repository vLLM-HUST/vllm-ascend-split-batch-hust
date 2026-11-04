#!/usr/bin/env python3
"""L0 microbenchmark for gdn_split_ba (vllm_ascend/ops/gdn.py:210)."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "M",
        "stride_bam",
        "stride_bm",
        "stride_am",
    ]
)
def gdn_split_ba_2d_kernel(
    ba_ptr,
    b_ptr,
    a_ptr,
    stride_bam,
    stride_bm,
    stride_am,
    M,
    HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)
    row_mask = row_offs < M

    b_in_ptrs = ba_ptr + row_offs[:, None] * stride_bam + col_offs[None, :]
    a_in_ptrs = ba_ptr + row_offs[:, None] * stride_bam + (col_offs[None, :] + HALF)

    b_out_ptrs = b_ptr + row_offs[:, None] * stride_bm + col_offs[None, :]
    a_out_ptrs = a_ptr + row_offs[:, None] * stride_am + col_offs[None, :]

    mask_2d = row_mask[:, None]

    val_b = tl.load(b_in_ptrs, mask=mask_2d, other=0.0)
    val_a = tl.load(a_in_ptrs, mask=mask_2d, other=0.0)

    tl.store(b_out_ptrs, val_b, mask=mask_2d)
    tl.store(a_out_ptrs, val_a, mask=mask_2d)


def optimized_op(ba: torch.Tensor):
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
        b.stride(0),
        a.stride(0),
        M,
        HALF=Half,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return b, a


if __name__ == "__main__":
    print("Benchmarking gdn_split_ba...")
