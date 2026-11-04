# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class GdnQkvNormCase(OpCase):
    name = "gdn_qkv_norm"
    callsite = "vllm_ascend/ops/gdn.py:480,619-620"
    phase = "pure decode"
    T: int = 16
    row_stride: int = 4096
    q_dim: int = 1024
    k_dim: int = 1024
    v_dim: int = 2048
    head_dim: int = 128

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "T": self.T,
            "row_stride": self.row_stride,
            "q": self.q_dim,
            "k": self.k_dim,
            "v": self.v_dim,
            "head_dim": self.head_dim,
            "phase": self.phase,
        }

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        T, row_stride = self.T, self.row_stride

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                x = torch.randn(T, row_stride, dtype=torch.bfloat16).npu(0)
                return (x,)
            else:
                np.random.seed(42)
                x = np.random.randn(T, row_stride).astype(np.float32)
                return (x,)

        return input_gen

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(x):
            if hasattr(x, "float"):
                import torch

                q = x[:, :1024].contiguous().view(1, 16, 8, 128)
                k = x[:, 1024:2048].contiguous().view(1, 16, 8, 128)
                v = x[:, 2048:].contiguous().view(1, 16, 16, 128)
                q_f = q.float()
                q_norm = (
                    q_f * torch.rsqrt((q_f * q_f).sum(-1, keepdim=True) + 1e-6)
                ).to(x.dtype)
                k_f = k.float()
                k_norm = (
                    k_f * torch.rsqrt((k_f * k_f).sum(-1, keepdim=True) + 1e-6)
                ).to(x.dtype)
                return q_norm, k_norm, v
            else:
                q = x[:, :1024].reshape(1, 16, 8, 128)
                k = x[:, 1024:2048].reshape(1, 16, 8, 128)
                v = x[:, 2048:].reshape(1, 16, 16, 128)
                q_norm = q / np.sqrt(np.sum(q**2, axis=-1, keepdims=True) + 1e-6)
                k_norm = k / np.sqrt(np.sum(k**2, axis=-1, keepdims=True) + 1e-6)
                return q_norm, k_norm, v

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
import torch

def optimized_op(x):
    if hasattr(x, "float"):
        q = x[:, :1024].contiguous().view(1, 16, 8, 128)
        k = x[:, 1024:2048].contiguous().view(1, 16, 8, 128)
        v = x[:, 2048:].contiguous().view(1, 16, 16, 128)
        q_f = q.float()
        q_norm = (
            q_f * torch.rsqrt((q_f * q_f).sum(-1, keepdim=True) + 1e-6)
        ).to(x.dtype)
        k_f = k.float()
        k_norm = (
            k_f * torch.rsqrt((k_f * k_f).sum(-1, keepdim=True) + 1e-6)
        ).to(x.dtype)
        return q_norm, k_norm, v
    else:
        import numpy as np
        q = x[:, :1024].reshape(1, 16, 8, 128)
        k = x[:, 1024:2048].reshape(1, 16, 8, 128)
        v = x[:, 2048:].reshape(1, 16, 16, 128)
        q_norm = q / np.sqrt(np.sum(q**2, axis=-1, keepdims=True) + 1e-6)
        k_norm = k / np.sqrt(np.sum(k**2, axis=-1, keepdims=True) + 1e-6)
        return q_norm, k_norm, v
"""

    def get_task_prompt(self) -> str:
        shape_str = f"x=[{self.T}, {self.row_stride}]"
        return (
            f"Optimize {self.name} operator.\n"
            f"Input signature: `optimized_op(x)` with {shape_str} bf16.\n"
            "Unpack layout:\n"
            "  q = x[:, :1024] -> view(1, 16, 8, 128), L2 norm on dim=128\n"
            "  k = x[:, 1024:2048] -> view(1, 16, 8, 128), L2 norm on dim=128\n"
            "  v = x[:, 2048:] -> view(1, 16, 16, 128), unchanged\n"
            "Returns: (q_norm, k_norm, v)\n\n"
            "Steps to optimize:\n"
            "1. Write an optimized Triton kernel using `write_candidate_file`.\n"
            "2. Evaluate it with `evaluate_kernel(file_path=...)`.\n"
            "3. If L3 passes and L0 speedup is achieved, call `final_answer`.\n"
            "4. Otherwise inspect error and iterate."
        )
