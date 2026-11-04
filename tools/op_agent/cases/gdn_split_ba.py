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
    shape_buckets = (1, 4, 8, 16, 32)
    dynamic_shape_probes = (7, 97, 266)
    recompile_budget_ms = 250.0

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "T": self.T,
            "TotalDim": self.TotalDim,
            "phase": self.phase,
            "shape_buckets": list(self.shape_buckets),
            "dynamic_shape_probes": list(self.dynamic_shape_probes),
        }

    def get_bucket_input_gen_fn(
        self, is_tp2_ready: bool
    ) -> Callable[[int], tuple[Any, ...]]:
        TotalDim = self.TotalDim

        def gen_for_tokens(batch_tokens: int) -> tuple[Any, ...]:
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42 + batch_tokens)
                ba = torch.randn(batch_tokens, TotalDim, dtype=torch.bfloat16).npu(0)
                return (ba,)
            else:
                rng = np.random.default_rng(42 + batch_tokens)
                ba = rng.standard_normal((batch_tokens, TotalDim)).astype(np.float32)
                return (ba,)

        return gen_for_tokens

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        bucket_gen = self.get_bucket_input_gen_fn(is_tp2_ready)
        return lambda: bucket_gen(self.T)

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
        shape_str = f"ba=[T, {self.TotalDim}]"
        return (
            f"Optimize {self.name} operator.\n"
            f"Input signature: `optimized_op(ba)` with dynamic {shape_str} bf16.\n"
            "Formula:\n"
            "  b, a = ba.chunk(2, dim=-1)\n"
            "  return b.contiguous(), a.contiguous()\n\n"
            "Critical Constraints:\n"
            "1. Do NOT mark dynamic row count `M = ba.shape[0]` or row strides as "
            "`tl.constexpr`! Use `@triton.jit(do_not_specialize=['M', 'stride_bam', "
            "'stride_bm', 'stride_am'])` so unseen token counts `T` (e.g. 7, 97, "
            "266) do not trigger Triton JIT recompilation.\n"
            "2. Write candidate via `write_candidate_file`, evaluate with "
            "`evaluate_kernel(file_path=...)`, and call `final_answer` on speedup."
        )
