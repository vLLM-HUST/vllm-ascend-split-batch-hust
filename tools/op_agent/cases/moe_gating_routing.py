# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


class MoeGatingRoutingCase(OpCase):
    """MoeGatingTopK -> MoeInitRoutingV3 composite pattern.

    Represents the MoE token dispatch stage in Qwen3.5-35B-A3B:
    1. `MoeGatingTopK`: Softmax router probabilities across 256 experts and
       selects Top-8 experts.
    2. `MoeInitRoutingV3`: Permutes token hidden states according to assigned
       experts and generates routing metadata for subsequent GroupedMatmul.
    """

    name = "moe_gating_routing"
    callsite = "vllm_ascend/ops/fused_moe/fused_moe.py:261"
    phase = "decode"
    default_tokens: int = 32
    E: int = 256
    K: int = 8
    H: int = 2048
    shape_buckets = (1, 4, 8, 16, 32)
    dynamic_shape_probes = (7, 24)
    recompile_budget_ms = 250.0

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "batch_tokens": self.default_tokens,
            "num_experts": self.E,
            "top_k": self.K,
            "hidden_size": self.H,
            "phase": self.phase,
            "shape_buckets": list(self.shape_buckets),
            "dynamic_shape_probes": list(self.dynamic_shape_probes),
        }

    def get_bucket_input_gen_fn(
        self, is_tp2_ready: bool
    ) -> Callable[[int], tuple[Any, ...]]:
        E, H = self.E, self.H

        def gen_for_tokens(batch_tokens: int) -> tuple[Any, ...]:
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42 + batch_tokens)
                router_logits = torch.randn(batch_tokens, E, dtype=torch.bfloat16).npu(
                    0
                )
                hidden_states = torch.randn(batch_tokens, H, dtype=torch.bfloat16).npu(
                    0
                )
                return router_logits, hidden_states
            else:
                rng = np.random.default_rng(42 + batch_tokens)
                router_logits = rng.standard_normal((batch_tokens, E)).astype(
                    np.float32
                )
                hidden_states = rng.standard_normal((batch_tokens, H)).astype(
                    np.float32
                )
                return router_logits, hidden_states

        return gen_for_tokens

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        bucket_gen = self.get_bucket_input_gen_fn(is_tp2_ready)
        return lambda: bucket_gen(self.default_tokens)

    def get_baseline_fn(self) -> Callable[..., Any]:
        E, K = self.E, self.K

        def baseline_op(router_logits, hidden_states):
            if hasattr(router_logits, "is_npu") and router_logits.is_npu:
                import torch_npu

                batch_tokens = router_logits.shape[0]
                topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                    router_logits,
                    K,
                    k_group=1,
                    group_count=1,
                    group_select_mode=0,
                    renorm=0,
                    norm_type=0,
                )
                expanded_x, expanded_row_idx, expert_tokens, _ = (
                    torch_npu.npu_moe_init_routing_v2(
                        hidden_states,
                        topk_ids,
                        active_num=batch_tokens * K,
                        expert_num=E,
                        expert_tokens_num_type=1,
                        expert_tokens_num_flag=True,
                        active_expert_range=[0, E],
                        quant_mode=-1,
                    )
                )
                return topk_weights, expanded_x, expanded_row_idx, expert_tokens
            else:
                batch_tokens = router_logits.shape[0]
                # Softmax over expert dimension
                exp_r = np.exp(
                    router_logits - np.max(router_logits, axis=-1, keepdims=True)
                )
                probs = exp_r / np.sum(exp_r, axis=-1, keepdims=True)

                # Top-K selection
                topk_ids = np.argsort(-probs, axis=-1)[:, :K].astype(np.int32)
                topk_weights = np.take_along_axis(probs, topk_ids, axis=-1)

                # Expert routing permutation
                flat_expert = topk_ids.reshape(-1)
                flat_token = np.repeat(np.arange(batch_tokens, dtype=np.int32), K)

                perm = np.argsort(flat_expert, kind="stable")
                expanded_row_idx = np.empty(batch_tokens * K, dtype=np.int32)
                expanded_row_idx[perm] = np.arange(batch_tokens * K, dtype=np.int32)

                sorted_tokens = flat_token[perm]
                expanded_x = hidden_states[sorted_tokens]
                expert_tokens = np.bincount(flat_expert, minlength=E).astype(np.int64)

                return topk_weights, expanded_x, expanded_row_idx, expert_tokens

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
import numpy as np

def optimized_op(router_logits, hidden_states):
    if hasattr(router_logits, "is_npu") and router_logits.is_npu:
        import torch_npu

        batch_tokens = router_logits.shape[0]
        topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
            router_logits,
            8,
            k_group=1,
            group_count=1,
            group_select_mode=0,
            renorm=0,
            norm_type=0,
        )
        expanded_x, expanded_row_idx, expert_tokens, _ = (
            torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=batch_tokens * 8,
                expert_num=256,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, 256],
                quant_mode=-1,
            )
        )
        return topk_weights, expanded_x, expanded_row_idx, expert_tokens
    else:
        batch_tokens = router_logits.shape[0]
        exp_r = np.exp(router_logits - np.max(router_logits, axis=-1, keepdims=True))
        probs = exp_r / np.sum(exp_r, axis=-1, keepdims=True)
        topk_ids = np.argsort(-probs, axis=-1)[:, :8].astype(np.int32)
        topk_weights = np.take_along_axis(probs, topk_ids, axis=-1)

        flat_expert = topk_ids.reshape(-1)
        flat_token = np.repeat(np.arange(batch_tokens, dtype=np.int32), 8)

        perm = np.argsort(flat_expert, kind="stable")
        expanded_row_idx = np.empty(batch_tokens * 8, dtype=np.int32)
        expanded_row_idx[perm] = np.arange(batch_tokens * 8, dtype=np.int32)

        sorted_tokens = flat_token[perm]
        expanded_x = hidden_states[sorted_tokens]
        expert_tokens = np.bincount(flat_expert, minlength=256).astype(np.int64)

        return topk_weights, expanded_x, expanded_row_idx, expert_tokens
"""

    def get_task_prompt(self) -> str:
        return (
            f"Optimize {self.name} operator (MoE token routing pipeline).\n"
            f"Input signature: `optimized_op(router_logits, hidden_states)` where "
            f"router_logits=[T, {self.E}] bf16, hidden_states=[T, {self.H}] bf16.\n"
            f"Formula: Softmax + Top-{self.K} expert selection, followed by token "
            "dispatch permutation sorted by expert ID.\n\n"
            "Steps to optimize:\n"
            "1. Write an optimized Triton / fused kernel using "
            "`write_candidate_file`.\n"
            "2. Evaluate it with `evaluate_kernel(file_path=...)`.\n"
            "3. If L3 passes and L0 speedup is achieved, call `final_answer`.\n"
            "4. Otherwise inspect error and iterate."
        )
