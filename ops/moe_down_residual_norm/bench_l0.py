#!/usr/bin/env python3
"""L0 microbenchmark for moe_down_residual_norm.

Callsite: vllm_ascend/ops/fused_moe/shared_experts.py:down_residual_norm.
"""

import torch


def optimized_op(x, down_w, residual, norm_w):
    # x: [32, 256]
    # down_w: [2048, 256]
    # residual: [32, 2048]
    # norm_w: [2048]
    # h = x @ down_w.T + residual
    h = torch.matmul(x, down_w.t()) + residual
    var = torch.mean(h * h, dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(var + 1e-6)
    out = (h * rsqrt) * norm_w
    return out


if __name__ == "__main__":
    print("Benchmarking moe_down_residual_norm...")
