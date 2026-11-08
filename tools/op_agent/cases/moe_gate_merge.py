# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class MoeGateMergeCase(OpCase):
    name = "moe_gate_merge"
    callsite = "vllm/model_executor/models/qwen3_next.py:186-192"
    phase = "decode"
    M: int = 32
    K: int = 2048
    N_router: int = 256
    N_shared: int = 1
    shape_buckets = (1, 8, 16, 32)
    dynamic_shape_probes = (7, 24)

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "M": self.M,
            "K": self.K,
            "N_router": self.N_router,
            "N_shared": self.N_shared,
            "phase": self.phase,
        }

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        M, K, N_router, N_shared = self.M, self.K, self.N_router, self.N_shared

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                x = torch.randn(M, K, dtype=torch.bfloat16).npu(0)
                router_w = torch.randn(N_router, K, dtype=torch.bfloat16).npu(0)
                shared_w = torch.randn(N_shared, K, dtype=torch.bfloat16).npu(0)
                return x, router_w, shared_w
            else:
                np.random.seed(42)
                x = np.random.randn(M, K).astype(np.float32)
                router_w = np.random.randn(N_router, K).astype(np.float32)
                shared_w = np.random.randn(N_shared, K).astype(np.float32)
                return x, router_w, shared_w

        return input_gen

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(x, router_w, shared_w):
            if hasattr(x, "matmul"):
                import torch

                router_logits = torch.matmul(x, router_w.t())
                shared_gate = torch.matmul(x, shared_w.t())
                return router_logits, shared_gate
            else:
                router_logits = np.matmul(x, router_w.T)
                shared_gate = np.matmul(x, shared_w.T)
                return router_logits, shared_gate

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
import torch

def optimized_op(x, router_w, shared_w):
    router_logits = torch.matmul(x, router_w.t())
    shared_gate = torch.matmul(x, shared_w.t())
    return router_logits, shared_gate
"""

    def get_task_prompt(self) -> str:
        return (
            "Optimize MoE Router Gate + Shared Gate linear projection. Baseline "
            "performs two separate GEMMs: x @ router_w.T [M, 256] and "
            "x @ shared_w.T [M, 1]. Merging the weights into [257, 2048] "
            "executes a single GEMM, but unaligned slicing (stride 257) introduces "
            "overhead. Explore aligned padding (e.g. pad to 272/288) or fused Triton "
            "GEMM to achieve net speedup."
        )
