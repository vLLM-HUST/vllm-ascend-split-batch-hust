#!/usr/bin/env python3
"""L0 and L3 test harness for op agent optimization loop."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from typing import Any

import numpy as np


def evaluate_kernel(
    kernel_code: str,
    test_case: dict[str, Any],
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, Any]:
    """Execute L3 numerical check and L0 benchmark on supplied kernel code.

    Returns a dict with:
        success: bool (whether L3 passed)
        l3_passed: bool
        l0_median_us: float
        error: str (if failed)
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(kernel_code)
        temp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("candidate_kernel", temp_path)
        if spec is None or spec.loader is None:
            return {
                "success": False,
                "l3_passed": False,
                "error": "Failed to load candidate module spec",
            }
        module = importlib.util.module_from_spec(spec)
        sys.modules["candidate_kernel"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "optimized_op"):
            return {
                "success": False,
                "l3_passed": False,
                "error": "Missing entrypoint function `optimized_op` in candidate.",
            }

        candidate_op = module.optimized_op

        # 1. L3 Numerical Check
        baseline_fn = test_case["baseline_fn"]
        input_gen_fn = test_case["input_gen_fn"]

        inputs = input_gen_fn()
        expected = baseline_fn(*inputs)
        actual = candidate_op(*inputs)

        # Handle numpy arrays or tensors
        if isinstance(expected, np.ndarray):
            matches = np.allclose(actual, expected, atol=atol, rtol=rtol)
            max_diff = float(np.max(np.abs(actual - expected)))
            if not matches:
                return {
                    "success": False,
                    "l3_passed": False,
                    "error": (
                        f"L3 Failed: max absolute diff is {max_diff}, "
                        f"exceeds tolerance (atol={atol}, rtol={rtol})"
                    ),
                }
        elif hasattr(expected, "allclose"):  # torch.Tensor
            import torch

            matches = torch.allclose(actual, expected, atol=atol, rtol=rtol)
            max_diff = torch.max(torch.abs(actual - expected)).item()
            if not matches:
                return {
                    "success": False,
                    "l3_passed": False,
                    "error": (
                        f"L3 Failed: max absolute diff is {max_diff}, "
                        f"exceeds tolerance (atol={atol}, rtol={rtol})"
                    ),
                }
        else:
            if actual != expected:
                return {
                    "success": False,
                    "l3_passed": False,
                    "error": (
                        f"L3 Failed: output does not match expected "
                        f"(actual={actual}, expected={expected})"
                    ),
                }

        # 2. L0 Benchmark (Warmup + iterations)
        for _ in range(10):
            _ = candidate_op(*inputs)

        durations = []
        iters = test_case.get("iters", 50)
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = candidate_op(*inputs)
            t1 = time.perf_counter()
            durations.append((t1 - t0) * 1e6)

        durations.sort()
        median_us = durations[len(durations) // 2]

        return {
            "success": True,
            "l3_passed": True,
            "l0_median_us": round(median_us, 2),
            "error": "",
        }

    except Exception as e:
        return {
            "success": False,
            "l3_passed": False,
            "error": f"Exception during execution: {type(e).__name__}: {str(e)}",
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
