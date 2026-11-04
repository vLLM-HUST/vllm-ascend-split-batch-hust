# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from tools.op_agent.workloads import select_workloads_for_operator


class OpCase(ABC):
    """Abstract base class for modular operator optimization cases.

    Supports both single-shape evaluation (`get_input_gen_fn`) and realistic
    multi-bucket / dynamic-shape inner-loop evaluation (`shape_buckets`,
    `dynamic_shape_probes`, and `get_bucket_input_gen_fn`), plus outer-loop
    workload selection (`get_workload_spec`).
    """

    name: str
    callsite: str
    phase: str
    iters: int = 50
    required_cards: int = 2
    max_active_rows: int | None = None
    shape_buckets: tuple[int, ...] = ()
    dynamic_shape_probes: tuple[int, ...] = ()
    recompile_budget_ms: float = 250.0

    @abstractmethod
    def get_shape_facts(self) -> dict[str, Any]:
        """Return canonical shape facts dictionary."""
        ...

    @abstractmethod
    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        """Return input generator generating dummy tensors for L3 and L0."""
        ...

    def get_bucket_input_gen_fn(
        self, is_tp2_ready: bool
    ) -> Callable[[int], tuple[Any, ...]] | None:
        """Optional generator accepting `batch_tokens: int` for shape sweeps.

        Subclasses should reproduce actual upstream tensor views/strides
        (e.g. non-contiguous column slices) rather than idealized contiguous
        buffers.
        """
        return None

    def get_workload_spec(self) -> dict[str, Any]:
        """Return outer-loop serving workload pairing (`random`, `sharegpt`, etc.)."""
        return select_workloads_for_operator(
            phase=self.phase, max_active_rows=self.max_active_rows
        )

    @abstractmethod
    def get_baseline_fn(self) -> Callable[..., Any]:
        """Return reference baseline implementation."""
        ...

    @abstractmethod
    def get_baseline_code(self) -> str:
        """Return pure baseline code string defining `optimized_op`."""
        ...

    @abstractmethod
    def get_task_prompt(self) -> str:
        """Return optimization instruction prompt for CodeAgent."""
        ...

    def build_test_case(self, is_tp2_ready: bool) -> dict[str, Any]:
        """Package into test_case dictionary consumed by evaluator."""
        return {
            "name": self.name,
            "callsite": self.callsite,
            "shape_facts": self.get_shape_facts(),
            "workload_spec": self.get_workload_spec(),
            "input_gen_fn": self.get_input_gen_fn(is_tp2_ready),
            "bucket_input_gen_fn": self.get_bucket_input_gen_fn(is_tp2_ready),
            "shape_buckets": list(self.shape_buckets),
            "dynamic_shape_probes": list(self.dynamic_shape_probes),
            "recompile_budget_ms": self.recompile_budget_ms,
            "baseline_fn": self.get_baseline_fn(),
            "iters": self.iters,
            "required_cards": self.required_cards,
        }
