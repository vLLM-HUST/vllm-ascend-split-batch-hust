# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class MoeDownResidualNormCase(OpCase):
    """Moe Down GEMM + Residual + RMSNorm on-chip loop (Guide Pattern 1 / D1).

    Replaces discrete 'Down MatMul -> Add -> RMSNorm' steps with on-chip
    accumulator reuse, eliminating intermediate HBM activations.
    """

    name = "moe_down_residual_norm"
    callsite = "vllm_ascend/ops/fused_moe/shared_experts.py:down_residual_norm"
    phase = "decode"
    M: int = 32
    K: int = 256
    N: int = 2048
    shape_buckets = (1, 8, 16, 32)
    dynamic_shape_probes = (7, 23)
    recompile_budget_ms = 250.0

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "M": self.M,
            "K": self.K,
            "N": self.N,
            "phase": self.phase,
            "hbm_rounds_unfused": 6,
            "hbm_rounds_fused": 2,
            "activation_saving_ratio": 0.667,
            "shape_buckets": list(self.shape_buckets),
            "dynamic_shape_probes": list(self.dynamic_shape_probes),
        }

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        M, K, N = self.M, self.K, self.N

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                x = torch.randn(M, K, dtype=torch.bfloat16).npu(0)
                down_w = torch.randn(N, K, dtype=torch.bfloat16).npu(0)
                residual = torch.randn(M, N, dtype=torch.bfloat16).npu(0)
                norm_w = torch.randn(N, dtype=torch.bfloat16).npu(0)
                return x, down_w, residual, norm_w
            else:
                rng = np.random.default_rng(42)
                x = rng.standard_normal((M, K)).astype(np.float32)
                down_w = rng.standard_normal((N, K)).astype(np.float32)
                residual = rng.standard_normal((M, N)).astype(np.float32)
                norm_w = rng.standard_normal(N).astype(np.float32)
                return x, down_w, residual, norm_w

        return input_gen

    def get_bucket_input_gen_fn(
        self, is_tp2_ready: bool
    ) -> Callable[[int], tuple[Any, ...]]:
        K, N = self.K, self.N

        def gen_for_tokens(batch_tokens: int) -> tuple[Any, ...]:
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42 + batch_tokens)
                x = torch.randn(batch_tokens, K, dtype=torch.bfloat16).npu(0)
                down_w = torch.randn(N, K, dtype=torch.bfloat16).npu(0)
                residual = torch.randn(batch_tokens, N, dtype=torch.bfloat16).npu(0)
                norm_w = torch.randn(N, dtype=torch.bfloat16).npu(0)
                return x, down_w, residual, norm_w
            else:
                rng = np.random.default_rng(42 + batch_tokens)
                x = rng.standard_normal((batch_tokens, K)).astype(np.float32)
                down_w = rng.standard_normal((N, K)).astype(np.float32)
                residual = rng.standard_normal((batch_tokens, N)).astype(np.float32)
                norm_w = rng.standard_normal(N).astype(np.float32)
                return x, down_w, residual, norm_w

        return gen_for_tokens

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(x, down_w, residual, norm_w):
            if hasattr(x, "device") and str(x.device).startswith("npu"):
                import torch
                import torch.nn.functional as F

                # 1. Down GEMM
                gemm_out = F.linear(x, down_w)
                # 2. Residual Add
                added = gemm_out + residual
                # 3. RMSNorm
                var = added.pow(2).mean(-1, keepdim=True)
                return added * torch.rsqrt(var + 1e-6) * norm_w
            else:
                gemm_out = x @ down_w.T
                added = gemm_out + residual
                var = np.mean(added**2, axis=-1, keepdims=True)
                return added * (1.0 / np.sqrt(var + 1e-6)) * norm_w

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
def optimized_op(x, down_w, residual, norm_w):
    if hasattr(x, "matmul"):
        import torch
        import torch.nn.functional as F

        gemm_out = F.linear(x, down_w)
        added = gemm_out + residual
        var = added.pow(2).mean(-1, keepdim=True)
        return added * torch.rsqrt(var + 1e-6) * norm_w
    else:
        import numpy as np

        gemm_out = x @ down_w.T
        added = gemm_out + residual
        var = np.mean(added**2, axis=-1, keepdims=True)
        return added * (1.0 / np.sqrt(var + 1e-6)) * norm_w
""".strip()

    def get_task_prompt(self) -> str:
        signatures = (
            "Exact tensor signatures (bf16, keep shapes and argument order):\n"
            f"  x: [{self.M}, {self.K}]\n"
            f"  down_w: [{self.N}, {self.K}]\n"
            f"  residual: [{self.M}, {self.N}]  <-- not [{self.M}, {self.K}]\n"
            f"  norm_w: [{self.N}]\n"
            f"  returns: [{self.M}, {self.N}]"
        )
        semantics = (
            f"Reference semantics: h = x @ down_w.T + residual is "
            f"[{self.M}, {self.N}]; out = h * rsqrt(mean(h*h, -1, keepdim=True) "
            "+ 1e-6) * norm_w."
        )
        guidance = (
            "Keep the entry point named optimized_op with the same argument order. "
            "Accumulate GEMM tiles in UB, add residual in-place, and apply RMSNorm "
            "before writing the result once to HBM. Submit one small single-purpose "
            "code block per step so each evaluation finishes within the time limit."
        )
        return "\n".join(
            (
                "Implement on-chip closed-loop fusion for Down GEMM + "
                "Residual Add + RMSNorm.",
                signatures,
                semantics,
                guidance,
            )
        )
