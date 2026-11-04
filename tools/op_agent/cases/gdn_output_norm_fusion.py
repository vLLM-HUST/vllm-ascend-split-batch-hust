# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class GdnOutputNormFusionCase(OpCase):
    """GDN Part 3 Output Norm + Rearrange fusion with realistic strided `z` view.

    Models two critical production constraints of Qwen3.5-35B-A3B (TP=2):
    1. `z` is sliced from `mixed_qkvz[:, 2048:].reshape(T, 16, 128)` and is
       non-contiguous (`stride=(6144, 128, 1)`), only unit-stride on last dim.
    2. Step token count `T` varies dynamically across chunked prefill and
       `sharegpt` long-tail traffic (`dynamic_shape_probes`), so Triton kernels
       must mark `M` and row strides `do_not_specialize` to avoid recompilation.
    """

    name = "gdn_output_norm_fusion"
    callsite = "vllm_ascend/ops/gdn.py:327"
    phase = "decode"
    default_tokens: int = 16
    num_v_heads: int = 16
    head_v_dim: int = 128
    qkv_prefix_cols: int = 2048
    shape_buckets = (1, 4, 8, 16, 32)
    dynamic_shape_probes = (7, 97, 266)
    recompile_budget_ms = 250.0

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "batch_tokens": self.default_tokens,
            "num_v_heads": self.num_v_heads,
            "head_v_dim": self.head_v_dim,
            "z_layout": "non_contiguous_column_slice_from_mixed_qkvz",
            "z_stride_pattern": "(6144, 128, 1)",
            "phase": self.phase,
            "shape_buckets": list(self.shape_buckets),
            "dynamic_shape_probes": list(self.dynamic_shape_probes),
        }

    def get_bucket_input_gen_fn(
        self, is_tp2_ready: bool
    ) -> Callable[[int], tuple[Any, ...]]:
        H, D = self.num_v_heads, self.head_v_dim
        prefix = self.qkv_prefix_cols
        total_cols = prefix + H * D * 2  # 2048 + 4096 = 6144

        def gen_for_tokens(batch_tokens: int) -> tuple[Any, ...]:
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42 + batch_tokens)
                core_out = torch.randn(batch_tokens, H, D, dtype=torch.bfloat16).npu(0)
                mixed_qkvz = torch.randn(
                    batch_tokens, total_cols, dtype=torch.bfloat16
                ).npu(0)
                # Reproduce exact non-contiguous slice view from Qwen3.5 gdn.py
                z = mixed_qkvz[:, prefix : prefix + H * D].reshape(batch_tokens, H, D)
                weight = torch.randn(D, dtype=torch.float32).npu(0)
                return core_out, z, weight
            else:
                rng = np.random.default_rng(42 + batch_tokens)
                core_out = rng.standard_normal((batch_tokens, H, D)).astype(np.float32)
                mixed_qkvz = rng.standard_normal((batch_tokens, total_cols)).astype(
                    np.float32
                )
                z = mixed_qkvz[:, prefix : prefix + H * D].reshape(batch_tokens, H, D)
                weight = rng.standard_normal((D,)).astype(np.float32)
                return core_out, z, weight

        return gen_for_tokens

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        bucket_gen = self.get_bucket_input_gen_fn(is_tp2_ready)
        return lambda: bucket_gen(self.default_tokens)

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(core_out, z, weight):
            T, H, D = z.shape
            if hasattr(core_out, "pow"):
                import torch
                import torch.nn.functional as F

                x_2d = core_out.reshape(-1, D).to(torch.float32)
                z_2d = z.reshape(-1, D).to(torch.float32)
                var = x_2d.pow(2).mean(-1, keepdim=True)
                normed = (x_2d * torch.rsqrt(var + 1e-6)) * weight.to(torch.float32)
                gated = (normed * F.silu(z_2d)).to(core_out.dtype)
                return gated.reshape(T, H * D)
            else:
                x_2d = core_out.reshape(-1, D)
                z_2d = z.reshape(-1, D)
                var = np.mean(np.square(x_2d), axis=-1, keepdims=True)
                normed = (x_2d / np.sqrt(var + 1e-6)) * weight
                silu_z = z_2d / (1.0 + np.exp(-z_2d))
                return (normed * silu_z).reshape(T, H * D)

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
def optimized_op(core_out, z, weight):
    T, H, D = z.shape
    if hasattr(core_out, "pow"):
        import torch
        import torch.nn.functional as F

        x_2d = core_out.reshape(-1, D).to(torch.float32)
        z_2d = z.reshape(-1, D).to(torch.float32)
        var = x_2d.pow(2).mean(-1, keepdim=True)
        normed = (x_2d * torch.rsqrt(var + 1e-6)) * weight.to(torch.float32)
        gated = (normed * F.silu(z_2d)).to(core_out.dtype)
        return gated.reshape(T, H * D)
    else:
        import numpy as np

        x_2d = core_out.reshape(-1, D)
        z_2d = z.reshape(-1, D)
        var = np.mean(np.square(x_2d), axis=-1, keepdims=True)
        normed = (x_2d / np.sqrt(var + 1e-6)) * weight
        silu_z = z_2d / (1.0 + np.exp(-z_2d))
        return (normed * silu_z).reshape(T, H * D)
"""

    def get_task_prompt(self) -> str:
        return (
            f"Optimize {self.name} operator at {self.callsite}.\n"
            "Signature: `optimized_op(core_out, z, weight)` -> 2D tensor `[T, H*D]`.\n"
            "Math spec (per-head RMSNormGated, D=128, H=16, eps=1e-6):\n"
            "  - `x_2d = core_out.reshape(T * H, D).to(torch.float32)`\n"
            "  - `z_2d = z.reshape(T * H, D).to(torch.float32)`\n"
            "  - `var = x_2d.pow(2).mean(dim=-1, keepdim=True)` (per 128-dim head!)\n"
            "  - `normed = (x_2d * torch.rsqrt(var + 1e-6)) * weight.float()`\n"
            "  - `gated = (normed * F.silu(z_2d)).to(core_out.dtype)`\n"
            "  - return `gated.reshape(T, H * D)`\n"
            "Constraints:\n"
            "1. `core_out` is `[T, 16, 128]` bf16 contiguous; `z` is `[T, 16, 128]` "
            "bf16 NON-CONTIGUOUS slice view (`z.stride()==(6144, 128, 1)`). "
            "Index 3D `(t_idx, h_idx)` with `z.stride(0)` and `z.stride(1)` in the "
            "Triton kernel while reducing each 128-dim head separately.\n"
            "2. Mark dynamic dimensions (`T` or `M`, `eps`, and row strides) with "
            "`do_not_specialize` in `@triton.jit` so unseen token counts `T` do not "
            "trigger recompilation.\n"
            "3. Allocate contiguous `[T, H*D]` output directly to eliminate "
            "intermediate reshape and rearrange views."
        )
