# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class GdnInputNormDecodeCase(OpCase):
    name = "gdn_input_norm_decode"
    callsite = "vllm_ascend/ops/gdn.py:in_proj_norm"
    phase = "decode"
    T: int = 32
    H: int = 2048

    def get_shape_facts(self) -> dict[str, Any]:
        return {"T": self.T, "Hidden": self.H, "phase": self.phase}

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        T, H = self.T, self.H

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                x = torch.randn(T, H, dtype=torch.bfloat16).npu(0)
                weight = torch.randn(H, dtype=torch.bfloat16).npu(0)
                return x, weight
            else:
                np.random.seed(42)
                x = np.random.randn(T, H).astype(np.float32)
                weight = np.random.randn(H).astype(np.float32)
                return x, weight

        return input_gen

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(x, weight):
            if hasattr(x, "pow"):
                import torch

                variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                rsqrt = torch.rsqrt(variance + 1e-6)
                return (x.to(torch.float32) * rsqrt).to(x.dtype) * weight
            else:
                variance = np.mean(np.square(x), axis=-1, keepdims=True)
                return (x / np.sqrt(variance + 1e-6)) * weight

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
import torch

def optimized_op(x, weight):
    if hasattr(x, "pow"):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + 1e-6)
        return (x.to(torch.float32) * rsqrt).to(x.dtype) * weight
    else:
        import numpy as np
        var = np.mean(np.square(x), axis=-1, keepdims=True)
        return (x / np.sqrt(var + 1e-6)) * weight
"""

    def get_task_prompt(self) -> str:
        return (
            f"Optimize {self.name} operator.\n"
            f"Input signature: `optimized_op(x, weight)` with x=[{self.T}, {self.H}], "
            f"weight=[{self.H}] bf16.\n"
            "Canonical RMSNorm formula:\n"
            "  variance = x.pow(2).mean(-1, keepdim=True)\n"
            "  return (x * torch.rsqrt(variance + 1e-6)) * weight\n\n"
            "Steps to optimize:\n"
            "1. Write an optimized Triton kernel using `write_candidate_file`.\n"
            "2. Evaluate it with `evaluate_kernel(file_path=...)`.\n"
            "3. If L3 passes and L0 speedup is achieved, call `final_answer`.\n"
            "4. Otherwise inspect error and iterate."
        )
