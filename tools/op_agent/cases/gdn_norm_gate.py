# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class GdnNormGateCase(OpCase):
    name = "gdn_norm_gate"
    callsite = "vllm_ascend/ops/gdn.py:255"
    phase = "decode"
    Rows: int = 512
    Dim: int = 128

    def get_shape_facts(self) -> dict[str, Any]:
        return {"Rows": self.Rows, "Dim": self.Dim, "phase": self.phase}

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        Rows, Dim = self.Rows, self.Dim

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                x = torch.randn(Rows, Dim, dtype=torch.bfloat16).npu(0)
                z = torch.randn(Rows, Dim, dtype=torch.bfloat16).npu(0)
                weight = torch.randn(Dim, dtype=torch.bfloat16).npu(0)
                return x, z, weight
            else:
                np.random.seed(42)
                x = np.random.randn(Rows, Dim).astype(np.float32)
                z = np.random.randn(Rows, Dim).astype(np.float32)
                weight = np.random.randn(Dim).astype(np.float32)
                return x, z, weight

        return input_gen

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(x, z, weight):
            if hasattr(x, "pow"):
                import torch
                import torch.nn.functional as F

                variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                rsqrt = torch.rsqrt(variance + 1e-6)
                x_normed = (x.to(torch.float32) * rsqrt) * weight.to(torch.float32)
                out = x_normed * F.silu(z.to(torch.float32))
                return out.to(x.dtype)
            else:
                variance = np.mean(np.square(x), axis=-1, keepdims=True)
                x_normed = (x / np.sqrt(variance + 1e-6)) * weight
                silu_z = z / (1.0 + np.exp(-z))
                return x_normed * silu_z

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
import torch
import torch.nn.functional as F

def optimized_op(x, z, weight):
    if hasattr(x, "pow"):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + 1e-6)
        x_normed = (x.to(torch.float32) * rsqrt) * weight.to(torch.float32)
        out = x_normed * F.silu(z.to(torch.float32))
        return out.to(x.dtype)
    else:
        import numpy as np
        var = np.mean(np.square(x), axis=-1, keepdims=True)
        x_normed = (x / np.sqrt(var + 1e-6)) * weight
        silu_z = z / (1.0 + np.exp(-z))
        return x_normed * silu_z
"""

    def get_task_prompt(self) -> str:
        sig = f"x=[{self.Rows}, {self.Dim}], z=[{self.Rows}, {self.Dim}]"
        return (
            f"Optimize {self.name} operator.\n"
            f"Input signature: `optimized_op(x, z, weight)` with {sig} bf16.\n"
            "Canonical Gated RMSNorm formula:\n"
            "  variance = x.pow(2).mean(-1, keepdim=True)\n"
            "  x_normed = (x * torch.rsqrt(variance + 1e-6)) * weight\n"
            "  out = x_normed * F.silu(z)\n\n"
            "Steps to optimize:\n"
            "1. Write an optimized Triton kernel using `write_candidate_file`.\n"
            "2. Evaluate it with `evaluate_kernel(file_path=...)`.\n"
            "3. If L3 passes and L0 speedup is achieved, call `final_answer`.\n"
            "4. Otherwise inspect error and iterate."
        )
