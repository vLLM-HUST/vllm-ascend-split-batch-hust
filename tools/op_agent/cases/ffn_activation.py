# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class FfnActivationCase(OpCase):
    """FFN Gate/Up SwiGLU / GeGLU activation fusion with FTZ numerical guard (Guide D4).

    Fuses elementwise activation (SiLU/GELU) with gate multiply on device, eliminating
    intermediate activation tensor materialization between FFN projections.
    """

    name = "ffn_activation"
    callsite = "vllm_ascend/ops/fused_moe/fused_moe.py:ffn_activation"
    phase = "decode"
    M: int = 32
    H_ffn: int = 1024
    shape_buckets = (1, 8, 16, 32)
    dynamic_shape_probes = (7, 23)
    recompile_budget_ms = 250.0

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "M": self.M,
            "Hidden_ffn": self.H_ffn,
            "phase": self.phase,
            "cosine_threshold": 0.99999,
            "shape_buckets": list(self.shape_buckets),
            "dynamic_shape_probes": list(self.dynamic_shape_probes),
        }

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        M, H = self.M, self.H_ffn

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                gate = torch.randn(M, H, dtype=torch.bfloat16).npu(0)
                up = torch.randn(M, H, dtype=torch.bfloat16).npu(0)
                return gate, up
            else:
                rng = np.random.default_rng(42)
                gate = rng.standard_normal((M, H)).astype(np.float32)
                up = rng.standard_normal((M, H)).astype(np.float32)
                return gate, up

        return input_gen

    def get_bucket_input_gen_fn(
        self, is_tp2_ready: bool
    ) -> Callable[[int], tuple[Any, ...]]:
        H = self.H_ffn

        def gen_for_tokens(batch_tokens: int) -> tuple[Any, ...]:
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42 + batch_tokens)
                gate = torch.randn(batch_tokens, H, dtype=torch.bfloat16).npu(0)
                up = torch.randn(batch_tokens, H, dtype=torch.bfloat16).npu(0)
                return gate, up
            else:
                rng = np.random.default_rng(42 + batch_tokens)
                gate = rng.standard_normal((batch_tokens, H)).astype(np.float32)
                up = rng.standard_normal((batch_tokens, H)).astype(np.float32)
                return gate, up

        return gen_for_tokens

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(gate, up):
            if hasattr(gate, "device") and str(gate.device).startswith("npu"):
                import torch.nn.functional as F

                return F.silu(gate) * up
            else:
                silu_gate = gate / (1.0 + np.exp(-gate))
                return silu_gate * up

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
def optimized_op(gate, up):
    if hasattr(gate, "matmul"):
        import torch.nn.functional as F

        return F.silu(gate) * up
    else:
        import numpy as np

        silu_gate = gate / (1.0 + np.exp(-gate))
        return silu_gate * up
""".strip()

    def get_task_prompt(self) -> str:
        return (
            f"Fuse FFN elementwise activation (SiLU(gate) * up, shape Mx{self.H_ffn}) "
            "into a single kernel. Guard against subnormal truncation (FTZ)."
        )
