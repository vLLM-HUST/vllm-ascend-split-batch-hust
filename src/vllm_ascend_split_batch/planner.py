# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure dual-pad planner migrated from legacy Ascend PR #281."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitBatchConfig:
    enabled: bool = False
    mode: str = "dual_pad"
    min_batch_size_for_split: int = 2
    padding_saved_threshold: int = 0
    force_split: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"dual_pad", "dual_inplace"}:
            raise ValueError("unsupported split-batch mode")
        if self.min_batch_size_for_split < 2 or self.padding_saved_threshold < 0:
            raise ValueError("invalid split-batch thresholds")


@dataclass(frozen=True)
class SplitSlice:
    request_start: int
    request_stop: int
    token_start: int
    token_stop: int
    padded_tokens: int


@dataclass(frozen=True)
class DualPadPlan:
    total_tokens: int
    padded_without_split: int
    padding_saved: int
    slices: tuple[SplitSlice, SplitSlice]


def precheck_reason(
    config: SplitBatchConfig,
    *,
    num_requests: int,
    graph_mode: str,
    uniform_decode: bool,
    has_lora: bool,
    is_mla: bool,
    is_mrope: bool,
    speculative_decode: bool,
) -> str | None:
    if not config.enabled or config.mode != "dual_pad":
        return "mode_disabled"
    if not uniform_decode:
        return "non_uniform_decode"
    if graph_mode not in {"full", "piecewise"}:
        return "graph_mode_not_supported"
    if speculative_decode:
        return "speculative_decode_conflict"
    if has_lora:
        return "lora_conflict"
    if is_mla:
        return "mla_conflict"
    if is_mrope:
        return "mrope_conflict"
    if num_requests < config.min_batch_size_for_split:
        return "batch_too_small"
    return None


def _ceil_capture(tokens: int, sizes: tuple[int, ...]) -> int | None:
    return next((size for size in sizes if size >= tokens), None)


def plan_dual_pad(
    *,
    num_requests: int,
    total_tokens: int,
    main_capture_sizes: tuple[int, ...],
    parallel_capture_sizes: tuple[int, ...] | None = None,
    padding_saved_threshold: int = 0,
    force_split: bool = False,
) -> tuple[DualPadPlan | None, str]:
    if num_requests <= 0 or total_tokens <= 0 or not main_capture_sizes:
        return None, "no_capture_sizes"
    main_sizes = tuple(sorted(set(main_capture_sizes)))
    parallel_sizes = tuple(sorted(set(parallel_capture_sizes or main_sizes)))
    if any(size <= 0 for size in (*main_sizes, *parallel_sizes)):
        raise ValueError("capture sizes must be positive")
    main_tokens = max((size for size in main_sizes if size <= total_tokens), default=0)
    if main_tokens == total_tokens:
        candidates = [size for size in main_sizes if size < total_tokens]
        if not force_split or not candidates:
            return None, "exact_graph_hit"
        main_tokens = max(candidates)
    if main_tokens <= 0:
        return None, "no_main_capture"
    second_tokens = total_tokens - main_tokens
    if second_tokens <= 0 or main_tokens >= num_requests:
        return None, "empty_parallel_slice"
    second_padded = _ceil_capture(second_tokens, parallel_sizes)
    if second_padded is None:
        return None, "no_parallel_capture"
    original_padded = _ceil_capture(total_tokens, main_sizes)
    if original_padded is None:
        return None, "exceeds_max_size"
    padding_saved = (original_padded - total_tokens) - (second_padded - second_tokens)
    if not force_split and padding_saved <= padding_saved_threshold:
        return None, "threshold_not_met"
    return (
        DualPadPlan(
            total_tokens=total_tokens,
            padded_without_split=original_padded,
            padding_saved=padding_saved,
            slices=(
                SplitSlice(0, main_tokens, 0, main_tokens, main_tokens),
                SplitSlice(
                    main_tokens,
                    num_requests,
                    main_tokens,
                    total_tokens,
                    second_padded,
                ),
            ),
        ),
        "split",
    )


__all__ = [
    "DualPadPlan",
    "SplitBatchConfig",
    "SplitSlice",
    "plan_dual_pad",
    "precheck_reason",
]
