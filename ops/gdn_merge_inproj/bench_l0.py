# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import torch


def baseline_op(
    x: torch.Tensor, w_qkvz: torch.Tensor, w_ba: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: two separate MatMuls for in_proj_qkvz and in_proj_ba."""
    out_qkvz = torch.matmul(x, w_qkvz)
    out_ba = torch.matmul(x, w_ba)
    return out_qkvz, out_ba


def optimized_op(x: torch.Tensor, w_merged: torch.Tensor) -> torch.Tensor:
    """Optimized: single merged MatMul with pre-concatenated weights."""
    return torch.matmul(x, w_merged)
