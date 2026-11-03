# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import torch


def baseline_op(w_bf16: torch.Tensor, layers: int = 40) -> list[torch.Tensor]:
    """Simulates 40 MoE layers performing gate.weight.to(torch.float32)."""
    res = []
    for _ in range(layers):
        res.append(w_bf16.to(torch.float32).contiguous())
    return res


def optimized_op(w_fp32: torch.Tensor, layers: int = 40) -> list[torch.Tensor]:
    """Hoisted FP32 cached gate weights (zero Cast during decode step)."""
    res = []
    for _ in range(layers):
        res.append(w_fp32)
    return res
