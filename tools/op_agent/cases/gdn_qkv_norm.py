# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class GdnQkvNormCase(OpCase):
    name = "gdn_qkv_norm"
    callsite = "vllm_ascend/ops/gdn.py:552-553,692-693"
    phase = "pure decode"
    T: int = 16
    row_stride: int = 4096
    q_dim: int = 1024
    k_dim: int = 1024
    v_dim: int = 2048
    head_dim: int = 128
    shape_buckets = (1, 4, 8, 16, 32)
    dynamic_shape_probes = (7, 97, 266)
    recompile_budget_ms = 250.0

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "T": self.T,
            "row_stride": self.row_stride,
            "q": self.q_dim,
            "k": self.k_dim,
            "v": self.v_dim,
            "head_dim": self.head_dim,
            "phase": self.phase,
            "shape_buckets": list(self.shape_buckets),
            "dynamic_shape_probes": list(self.dynamic_shape_probes),
        }

    def get_bucket_input_gen_fn(
        self, is_tp2_ready: bool
    ) -> Callable[[int], tuple[Any, ...]]:
        row_stride = self.row_stride

        def gen_for_tokens(batch_tokens: int) -> tuple[Any, ...]:
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42 + batch_tokens)
                x = torch.randn(batch_tokens, row_stride, dtype=torch.bfloat16).npu(0)
                return (x,)
            else:
                rng = np.random.default_rng(42 + batch_tokens)
                x = rng.standard_normal((batch_tokens, row_stride)).astype(np.float32)
                return (x,)

        return gen_for_tokens

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        bucket_gen = self.get_bucket_input_gen_fn(is_tp2_ready)
        return lambda: bucket_gen(self.T)

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(x):
            seq_len = x.shape[0]
            if hasattr(x, "float"):
                import torch

                q_dim, k_dim, v_dim = 1024, 1024, 2048
                query, key, value = torch.split(x, [q_dim, k_dim, v_dim], dim=-1)
                fused = torch.cat(
                    [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
                )
                q_size = seq_len * q_dim
                k_size = seq_len * k_dim
                q = fused[0:q_size].view(1, seq_len, 8, 128)
                k = fused[q_size : q_size + k_size].view(1, seq_len, 8, 128)
                v = fused[q_size + k_size :].view(1, seq_len, 16, 128)
                try:
                    from vllm.third_party.flash_linear_attention.ops.l2norm import (
                        l2norm_fwd,
                    )

                    return l2norm_fwd(q), l2norm_fwd(k), v
                except Exception:
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
                q = x[:, :1024].reshape(1, seq_len, 8, 128)
                k = x[:, 1024:2048].reshape(1, seq_len, 8, 128)
                v = x[:, 2048:].reshape(1, seq_len, 16, 128)
                q_norm = q / np.sqrt(np.sum(q**2, axis=-1, keepdims=True) + 1e-6)
                k_norm = k / np.sqrt(np.sum(k**2, axis=-1, keepdims=True) + 1e-6)
                return q_norm, k_norm, v

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
def optimized_op(x):
    seq_len = x.shape[0]
    if hasattr(x, "float"):
        import torch

        q_dim, k_dim, v_dim = 1024, 1024, 2048
        query, key, value = torch.split(x, [q_dim, k_dim, v_dim], dim=-1)
        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )
        q_size = seq_len * q_dim
        k_size = seq_len * k_dim
        q = fused[0:q_size].view(1, seq_len, 8, 128)
        k = fused[q_size : q_size + k_size].view(1, seq_len, 8, 128)
        v = fused[q_size + k_size :].view(1, seq_len, 16, 128)
        try:
            from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

            return l2norm_fwd(q), l2norm_fwd(k), v
        except Exception:
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
        q = x[:, :1024].reshape(1, seq_len, 8, 128)
        k = x[:, 1024:2048].reshape(1, seq_len, 8, 128)
        v = x[:, 2048:].reshape(1, seq_len, 16, 128)
        q_norm = q / np.sqrt(np.sum(q**2, axis=-1, keepdims=True) + 1e-6)
        k_norm = k / np.sqrt(np.sum(k**2, axis=-1, keepdims=True) + 1e-6)
        return q_norm, k_norm, v
"""

    def get_task_prompt(self) -> str:
        return (
            f"Optimize {self.name} operator at {self.callsite}.\n"
            "Signature: `optimized_op(x)` with dynamic shape `[T, 4096]` bf16.\n"
            "Mathematical specification (for arbitrary dynamic `T = x.shape[0]`):\n"
            "  - `q = x[:, :1024].reshape(1, T, 8, 128)` L2-normalized along dim=128: "
            "`q_f = q.float(); q_norm = (q_f * rsqrt(sum(q_f*q_f, -1, True) + 1e-6))`\n"
            "  - `k = x[:, 1024:2048].reshape(1, T, 8, 128)` L2-normalized on dim=128: "
            "`k_f = k.float(); k_norm = (k_f * rsqrt(sum(k_f*k_f, -1, True) + 1e-6))`\n"
            "  - `v = x[:, 2048:].contiguous().view(1, T, 16, 128)` unpacked copy.\n"
            "Returns `(q_norm.to(x.dtype), k_norm.to(x.dtype), v)` as contiguous "
            "shapes `(1, T, 8, 128)`, `(1, T, 8, 128)`, `(1, T, 16, 128)`.\n\n"
            "Critical Constraints:\n"
            "1. Do NOT hardcode `T=16`; use `T = x.shape[0]` dynamically.\n"
            "2. Mark dynamic dimensions (`T`, strides) with `do_not_specialize` in "
            "`@triton.jit` so unseen token counts `T` do not trigger recompilation.\n"
            "3. Fuse unpack, L2 norm of Q and K, and V copy into one Triton kernel.\n"
            "4. Save candidate via `write_candidate_file`, evaluate with "
            "`evaluate_kernel(file_path=...)`, and call `final_answer` on speedup."
        )
