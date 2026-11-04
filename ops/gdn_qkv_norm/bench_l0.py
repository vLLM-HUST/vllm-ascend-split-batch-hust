#!/usr/bin/env python3
"""L0 microbenchmark for gdn_qkv_norm (vllm_ascend/ops/gdn.py:552-553,692-693)."""

import torch
import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["T", "stride_x_t", "stride_q_t", "stride_k_t", "stride_v_t"]
)
def _gdn_qkv_norm_kernel_2d(
    x_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    T,
    stride_x_t,
    stride_q_t,
    stride_k_t,
    stride_v_t,
    BLOCK_128: tl.constexpr = 128,
    BLOCK_256: tl.constexpr = 256,
    eps: tl.constexpr = 1e-6,
):
    h = tl.program_id(axis=0)
    t = tl.program_id(axis=1)

    cols = tl.arange(0, BLOCK_128)
    cols_v = tl.arange(0, BLOCK_256)

    x_row = x_ptr + t * stride_x_t
    q_row = q_ptr + t * stride_q_t
    k_row = k_ptr + t * stride_k_t
    v_row = v_ptr + t * stride_v_t

    q_val = tl.load(x_row + h * BLOCK_128 + cols)
    k_val = tl.load(x_row + 1024 + h * BLOCK_128 + cols)
    v_val = tl.load(x_row + 2048 + h * BLOCK_256 + cols_v)

    tl.store(v_row + h * BLOCK_256 + cols_v, v_val)

    q_f = q_val.to(tl.float32)
    k_f = k_val.to(tl.float32)

    q_var = tl.sum(q_f * q_f)
    k_var = tl.sum(k_f * k_f)

    q_norm = (q_f * tl.rsqrt(q_var + eps)).to(tl.bfloat16)
    k_norm = (k_f * tl.rsqrt(k_var + eps)).to(tl.bfloat16)

    tl.store(q_row + h * BLOCK_128 + cols, q_norm)
    tl.store(k_row + h * BLOCK_128 + cols, k_norm)


def optimized_op(x: torch.Tensor):
    T = x.shape[0]
    q_norm = torch.empty((1, T, 8, 128), dtype=x.dtype, device=x.device)
    k_norm = torch.empty((1, T, 8, 128), dtype=x.dtype, device=x.device)
    v = torch.empty((1, T, 16, 128), dtype=x.dtype, device=x.device)

    grid = (8, T)
    _gdn_qkv_norm_kernel_2d[grid](
        x,
        q_norm,
        k_norm,
        v,
        T,
        x.stride(0),
        q_norm.stride(1),
        k_norm.stride(1),
        v.stride(1),
    )
    return q_norm, k_norm, v


if __name__ == "__main__":
    print("Benchmarking gdn_qkv_norm...")
