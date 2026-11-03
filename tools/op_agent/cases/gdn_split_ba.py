# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class GdnSplitBaCase(OpCase):
    name = "gdn_split_ba"
    callsite = "vllm_ascend/ops/gdn.py:210"
    phase = "decode"
    T: int = 32
    TotalDim: int = 32

    def get_shape_facts(self) -> dict[str, Any]:
        return {"T": self.T, "TotalDim": self.TotalDim, "phase": self.phase}

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        T, TotalDim = self.T, self.TotalDim

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                ba = torch.randn(T, TotalDim, dtype=torch.bfloat16).npu(0)
                return (ba,)
            else:
                np.random.seed(42)
                ba = np.random.randn(T, TotalDim).astype(np.float32)
                return (ba,)

        return input_gen

    def get_baseline_fn(self) -> Callable[..., Any]:
        HalfDim = self.TotalDim // 2

        def baseline_op(ba):
            if hasattr(ba, "chunk"):
                b, a = ba.chunk(2, dim=-1)
                return b.contiguous(), a.contiguous()
            else:
                b = ba[:, :HalfDim]
                a = ba[:, HalfDim:]
                return b, a

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
import torch

def optimized_op(ba):
    if hasattr(ba, "chunk"):
        b, a = ba.chunk(2, dim=-1)
        return b.contiguous(), a.contiguous()
    else:
        half = ba.shape[-1] // 2
        return ba[:, :half], ba[:, half:]
"""

    def get_task_prompt(self) -> str:
        shape_str = f"ba=[{self.T}, {self.TotalDim}]"
        return (
            f"Optimize {self.name} operator.\n"
            f"Input signature: `optimized_op(ba)` with {shape_str} bf16.\n"
            "Formula:\n"
            "  b, a = ba.chunk(2, dim=-1)\n"
            "  return b.contiguous(), a.contiguous()\n\n"
            "Steps to optimize:\n"
            "1. Write an optimized Triton kernel using `write_candidate_file`.\n"
            "2. Evaluate it with `evaluate_kernel(file_path=...)`.\n"
            "3. If L3 passes and L0 speedup is achieved, call `final_answer`.\n"
            "4. Otherwise inspect error and iterate."
        )
