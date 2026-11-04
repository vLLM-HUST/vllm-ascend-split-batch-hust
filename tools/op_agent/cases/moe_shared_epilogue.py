# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class MoeSharedEpilogueDecodeCase(OpCase):
    name = "moe_shared_epilogue_decode"
    callsite = "vllm_ascend/ops/fused_moe/shared_experts.py:part2"
    phase = "decode"
    M: int = 32
    H: int = 2048
    max_active_rows: int | None = 32
    shape_buckets = (1, 8, 16, 32)
    dynamic_shape_probes = (7, 23)

    def get_shape_facts(self) -> dict[str, Any]:
        return {"M": self.M, "Hidden": self.H, "phase": self.phase}

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        M, H = self.M, self.H

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                gate = torch.randn(M, 1, dtype=torch.bfloat16).npu(0)
                out = torch.randn(M, H, dtype=torch.bfloat16).npu(0)
                return gate, out
            else:
                np.random.seed(42)
                gate = np.random.randn(M, 1).astype(np.float32)
                out = np.random.randn(M, H).astype(np.float32)
                return gate, out

        return input_gen

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(gate, out):
            return gate * out

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
def optimized_op(gate, out):
    return gate * out
"""

    def get_task_prompt(self) -> str:
        return (
            f"Optimize {self.name} operator.\n"
            f"Input signature: `optimized_op(gate, out)` with gate=[{self.M}, 1] bf16, "
            f"out=[{self.M}, {self.H}] bf16.\n"
            "Formula: gate * out\n\n"
            "Steps to optimize:\n"
            "1. Write an optimized Triton kernel using `write_candidate_file`.\n"
            "2. Evaluate it with `evaluate_kernel(file_path=...)`.\n"
            "3. If L3 passes and L0 speedup is achieved, call `final_answer`.\n"
            "4. Otherwise inspect error and iterate."
        )
