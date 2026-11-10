#!/usr/bin/env python3
"""L0 microbenchmark for ffn_activation.

Callsite: vllm_ascend/ops/fused_moe/fused_moe.py:ffn_activation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Numerically stable SiLU in FP32
    gate_clamped = tl.maximum(gate, -88.0)
    sig = 1.0 / (1.0 + tl.exp(-gate_clamped))
    silu_gate_fp32 = gate * sig

    # Round to input dtype (FP16/BF16) to match unfused PyTorch intermediate
    silu_gate = silu_gate_fp32.to(gate_ptr.dtype.element_ty).to(tl.float32)
    res = silu_gate * up

    tl.store(out_ptr + offsets, res.to(out_ptr.dtype.element_ty), mask=mask)


def optimized_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not gate.is_contiguous():
        gate = gate.contiguous()
    if not up.is_contiguous():
        up = up.contiguous()

    out = torch.empty_like(gate)
    total_elements = gate.numel()
    BLOCK_SIZE = 2048
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _silu_mul_kernel[grid](
        gate,
        up,
        out,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


if __name__ == "__main__":
    print("Benchmarking ffn_activation...")
