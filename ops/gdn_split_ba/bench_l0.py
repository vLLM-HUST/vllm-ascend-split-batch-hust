#!/usr/bin/env python3
"""L0 microbenchmark for gdn_split_ba (vllm_ascend/ops/gdn.py:210)."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "stride_bam", "stride_bm", "stride_am"])
def split_ba_kernel(
    ba_ptr, b_ptr, a_ptr, M, stride_bam, stride_bm, stride_am, BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_k = tl.arange(0, 16)

    m_bam = offs_m[:, None] * stride_bam
    ptr_b_in = ba_ptr + m_bam + offs_k[None, :]
    ptr_a_in = ba_ptr + m_bam + (16 + offs_k[None, :])

    mask = mask_m[:, None]
    val_b = tl.load(ptr_b_in, mask=mask, other=0.0)
    val_a = tl.load(ptr_a_in, mask=mask, other=0.0)

    ptr_b_out = b_ptr + offs_m[:, None] * stride_bm + offs_k[None, :]
    ptr_a_out = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :]

    tl.store(ptr_b_out, val_b, mask=mask)
    tl.store(ptr_a_out, val_a, mask=mask)


def optimized_op(ba: torch.Tensor):
    M = ba.shape[0]
    out = torch.empty((2, M, 16), dtype=ba.dtype, device=ba.device)
    b = out[0]
    a = out[1]

    BLOCK_M = 64
    grid = ((M + 63) // 64, 1, 1)
    split_ba_kernel[grid](
        ba, b, a, M, ba.stride(0), 16, 16, BLOCK_M=BLOCK_M, num_warps=1
    )
    return b, a


if __name__ == "__main__":
    print("Benchmarking gdn_split_ba...")
