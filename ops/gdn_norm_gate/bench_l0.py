#!/usr/bin/env python3
"""L0 microbenchmark for gdn_norm_gate (vllm_ascend/ops/gdn.py:255)."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "stride_xm", "stride_zm", "stride_outm"])
def gdn_norm_gate_2d_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    out_ptr,
    M,
    stride_xm,
    stride_zm,
    stride_outm,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    mask = rm[:, None] < M

    w = tl.load(w_ptr + rn)
    w_f = w.to(tl.float32)[None, :]

    x_ptrs = x_ptr + rm[:, None] * stride_xm + rn[None, :]
    z_ptrs = z_ptr + rm[:, None] * stride_zm + rn[None, :]
    out_ptrs = out_ptr + rm[:, None] * stride_outm + rn[None, :]

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    z = tl.load(z_ptrs, mask=mask, other=0.0)

    x_f = x.to(tl.float32)
    z_f = z.to(tl.float32)

    var = tl.sum(x_f * x_f, axis=1, keep_dims=True) * (1.0 / BLOCK_N)
    rsqrt = tl.rsqrt(var + eps)
    x_normed = x_f * rsqrt * w_f
    silu_z = z_f * tl.sigmoid(z_f)
    out = (x_normed * silu_z).to(tl.bfloat16)

    tl.store(out_ptrs, out, mask=mask)


def optimized_op(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    grid = (triton.cdiv(M, 32),)
    gdn_norm_gate_2d_kernel[grid](
        x,
        z,
        weight,
        out,
        M,
        x.stride(0),
        z.stride(0),
        out.stride(0),
        eps=1e-6,
        BLOCK_M=32,
        BLOCK_N=128,
    )
    return out


if __name__ == "__main__":
    print("Benchmarking gdn_norm_gate...")
