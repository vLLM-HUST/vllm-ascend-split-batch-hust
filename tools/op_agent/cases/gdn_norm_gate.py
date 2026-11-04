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
    num_v_heads: int = 16
    shape_buckets = (1, 4, 8, 16, 32)
    dynamic_shape_probes = (7, 97, 266)
    recompile_budget_ms = 250.0

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "Rows": self.Rows,
            "Dim": self.Dim,
            "num_v_heads": self.num_v_heads,
            "phase": self.phase,
            "shape_buckets": list(self.shape_buckets),
            "dynamic_shape_probes": list(self.dynamic_shape_probes),
        }

    def get_bucket_input_gen_fn(
        self, is_tp2_ready: bool
    ) -> Callable[[int], tuple[Any, ...]]:
        Dim = self.Dim
        heads = self.num_v_heads

        def gen_for_tokens(batch_tokens: int) -> tuple[Any, ...]:
            rows = batch_tokens * heads
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42 + batch_tokens)
                x = torch.randn(rows, Dim, dtype=torch.bfloat16).npu(0)
                z = torch.randn(rows, Dim, dtype=torch.bfloat16).npu(0)
                weight = torch.randn(Dim, dtype=torch.bfloat16).npu(0)
                return x, z, weight
            else:
                rng = np.random.default_rng(42 + batch_tokens)
                x = rng.standard_normal((rows, Dim)).astype(np.float32)
                z = rng.standard_normal((rows, Dim)).astype(np.float32)
                weight = rng.standard_normal((Dim,)).astype(np.float32)
                return x, z, weight

        return gen_for_tokens

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        bucket_gen = self.get_bucket_input_gen_fn(is_tp2_ready)
        return lambda: bucket_gen(self.Rows // self.num_v_heads)

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
        sig = f"x=[M, {self.Dim}], z=[M, {self.Dim}]"
        return (
            f"Optimize {self.name} operator.\n"
            f"Input signature: `optimized_op(x, z, weight)` with dynamic {sig} bf16 "
            f"(where `M = x.shape[0] = T * {self.num_v_heads}`).\n"
            "Canonical Gated RMSNorm formula (in fp32 intermediate precision):\n"
            "  variance = x.float().pow(2).mean(-1, keepdim=True)\n"
            "  x_normed = (x.float() * torch.rsqrt(variance + 1e-6)) * weight.float()\n"
            "  out = (x_normed * F.silu(z.float())).to(x.dtype)\n\n"
            "Critical Constraints:\n"
            "1. Do NOT mark dynamic row count `M` or row strides as `tl.constexpr`! "
            "Use `@triton.jit(do_not_specialize=['M', 'stride_xm', 'stride_zm', "
            "'stride_outm'])` so unseen token counts `T` (e.g. 7, 97, 266) do not "
            "trigger Triton JIT recompilation.\n"
            "2. Write candidate via `write_candidate_file`, evaluate with "
            "`evaluate_kernel(file_path=...)`, and call `final_answer` on speedup."
        )
