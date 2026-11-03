#!/usr/bin/env python3
"""Canonical shape facts and weight labels for Qwen3.5-35B-A3B (TP=2).

Per PURPOSE.md and docs/operator-benchmark-standard.md:
Comparability is strictly determined by:
1. Operator Shape Facts (M, K, N, Top-K, Phase, Callsite)
2. Weight Labels (Model, Precision, Weight Format)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Qwen35ShapeContext:
    model_name: str = "Qwen3.5-35B-A3B"
    precision: str = "torch.bfloat16"
    tp_size: int = 2
    full_hidden_size: int = 4096
    tp_hidden_size: int = 2048  # Local hidden dimension on each rank under TP=2
    num_experts: int = 256
    top_k_experts: int = 8
    gdn_state_dim: int = 64
    gdn_conv_kernel: int = 4


CANONICAL_CASES: dict[str, dict[str, Any]] = {
    # 1. MoE Shared-Expert Broadcast Multiply (Decode phase)
    "moe_shared_epilogue_decode": {
        "callsite": "vllm_ascend/ops/fused_moe/shared_experts.py:part2",
        "phase": "decode",
        "gate_shape": (32, 1),
        "shared_out_shape": (32, 2048),
        "dtype": "bfloat16",
        "desc": "Decode broadcast multiply (T=32, H=2048)",
    },
    # 2. GDN Input Projection Merge / Norm
    "gdn_input_norm_decode": {
        "callsite": "vllm_ascend/ops/gdn.py:in_proj_norm",
        "phase": "decode",
        "input_shape": (32, 2048),
        "dtype": "bfloat16",
        "desc": "GDN in_proj norm over batch tokens (T=32, H=2048)",
    },
    # 3. Prefill GEMM / Epilogue (Batch Prefill)
    "moe_shared_epilogue_prefill": {
        "callsite": "vllm_ascend/ops/fused_moe/shared_experts.py:part2",
        "phase": "prefill",
        "gate_shape": (512, 1),
        "shared_out_shape": (512, 2048),
        "dtype": "bfloat16",
        "desc": "Prefill broadcast multiply (T=512, H=2048)",
    },
    # 4. GDN QKV Unpack and L2Norm Fusion (Qwen3.5 Linear Attention)
    "gdn_qkv_norm": {
        "callsite": "src/vllm_ascend_split_batch/gdn_qkv_norm.py:unpack_norm",
        "phase": "decode",
        "input_shape": (16, 4096),
        "dtype": "bfloat16",
        "desc": "Fused QKV unpack and L2Norm (T=16, C=4096)",
    },
}
