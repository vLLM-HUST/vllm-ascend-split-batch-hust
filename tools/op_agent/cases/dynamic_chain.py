# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

from .base import OpCase


def _parse_first_shape(
    shape_str: str, default: tuple[int, int] = (32, 4096)
) -> tuple[int, int]:
    """Parse '[32, 4096]' or '32,4096' into a 2D tuple."""
    nums = [int(n) for n in re.findall(r"\d+", shape_str or "")]
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], default[1]
    return default


class DynamicVectorChainCase(OpCase):
    """Dynamic OpCase synthesized from Timeline Insight vector sandwich chains."""

    phase = "decode"
    iters = 20
    required_cards = 1

    def __init__(self, chain_card: dict[str, Any]) -> None:
        self.card = chain_card
        raw_pat = chain_card.get("pattern_signature", "chain")
        pattern = raw_pat.replace(" -> ", "_").lower()
        clean_pat = re.sub(r"[^a-z0-9_]+", "", pattern)
        rank = chain_card.get("rank", 1)
        self.name = f"dynamic_chain_{clean_pat}_r{rank}"
        ab = chain_card.get("anchor_before", {}).get("type", "START")
        aa = chain_card.get("anchor_after", {}).get("type", "END")
        self.callsite = f"timeline_sandwich:[{ab}]->[{pattern}]->[{aa}]"

        ops = chain_card.get("operators", [])
        first_shape = ops[0].get("in_shape", "") if ops else ""
        self.M, self.Hidden = _parse_first_shape(first_shape, (32, 4096))
        self.ops = ops

    def get_shape_facts(self) -> dict[str, Any]:
        return {
            "M": self.M,
            "Hidden": self.Hidden,
            "phase": self.phase,
            "pattern_signature": self.card.get("pattern_signature"),
            "num_ops_in_chain": self.card.get("num_ops_in_chain", len(self.ops)),
            "anchor_before": self.card.get("anchor_before", {}).get("type"),
            "anchor_after": self.card.get("anchor_after", {}).get("type"),
            "total_chain_us": self.card.get("avg_chain_dur_us", 0.0),
        }

    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        M, H = self.M, self.Hidden

        def input_gen():
            if is_tp2_ready:
                import torch
                import torch_npu  # noqa: F401

                torch.manual_seed(42)
                x = torch.randn(M, H, dtype=torch.bfloat16).npu(0)
                w = torch.randn(H, dtype=torch.bfloat16).npu(0)
                return x, w
            else:
                rng = np.random.default_rng(42)
                x = rng.standard_normal((M, H)).astype(np.float32)
                w = rng.standard_normal(H).astype(np.float32)
                return x, w

        return input_gen

    def get_baseline_fn(self) -> Callable[..., Any]:
        def baseline_op(x, w):
            # General vector chain mock baseline (RMSNorm + SiLU/Gate)
            if hasattr(x, "device") and str(x.device).startswith("npu"):
                import torch.nn.functional as F

                variance = x.pow(2).mean(-1, keepdim=True)
                rrms = (variance + 1e-6).rsqrt()
                normed = x * rrms * w
                return F.silu(normed)
            else:
                variance = np.mean(x**2, axis=-1, keepdims=True)
                rrms = 1.0 / np.sqrt(variance + 1e-6)
                normed = x * rrms * w
                return normed / (1.0 + np.exp(-normed))

        return baseline_op

    def get_baseline_code(self) -> str:
        return """
import torch
import torch.nn.functional as F

def optimized_op(x, w):
    variance = x.pow(2).mean(-1, keepdim=True)
    rrms = torch.rsqrt(variance + 1e-6)
    return F.silu(x * rrms * w)
""".strip()

    def get_task_prompt(self) -> str:
        pat = self.card.get("pattern_signature", "Vector Chain")
        return (
            f"Fuse the contiguous vector chain '{pat}' ({self.M}x{self.Hidden}) "
            "between heavy anchor ops to eliminate intermediate HBM round trips."
        )
