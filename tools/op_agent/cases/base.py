# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class OpCase(ABC):
    """Abstract base class for modular operator optimization cases."""

    name: str
    callsite: str
    phase: str
    iters: int = 50
    required_cards: int = 2

    @abstractmethod
    def get_shape_facts(self) -> dict[str, Any]:
        """Return canonical shape facts dictionary."""
        ...

    @abstractmethod
    def get_input_gen_fn(self, is_tp2_ready: bool) -> Callable[..., tuple[Any, ...]]:
        """Return input generator generating dummy tensors for L3 and L0."""
        ...

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
            "input_gen_fn": self.get_input_gen_fn(is_tp2_ready),
            "baseline_fn": self.get_baseline_fn(),
            "iters": self.iters,
            "required_cards": self.required_cards,
        }
