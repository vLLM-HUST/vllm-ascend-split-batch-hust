#!/usr/bin/env python3
"""Dual-mode L0 and L3 evaluator supporting Eager and NPUGraph execution."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from typing import Any

import numpy as np

from tools.op_agent.guard import check_npu_environment


def run_candidate_evaluation(
    kernel_code: str,
    test_case: dict[str, Any],
    use_npu_graph: bool = True,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    iters: int = 50,
) -> dict[str, Any]:
    """Execute L3 numerical check and L0 benchmark on candidate operator code.

    Supports both Eager timing and NPUGraph replay timing to measure pure
    compute latency and eliminate Triton kernel launch overheads.
    """
    if "TORCH_DEVICE_BACKEND_AUTOLOAD" not in os.environ:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    hw_info = check_npu_environment(required_cards=test_case.get("required_cards", 1))

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(kernel_code)
        temp_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("candidate_module", temp_path)
        if spec is None or spec.loader is None:
            return {
                "success": False,
                "l3_passed": False,
                "error": "Failed to create module spec from candidate code.",
            }
        module = importlib.util.module_from_spec(spec)
        sys.modules["candidate_module"] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "optimized_op"):
            return {
                "success": False,
                "l3_passed": False,
                "error": "Missing entrypoint function `optimized_op` in candidate.",
            }

        candidate_op = module.optimized_op
        baseline_fn = test_case["baseline_fn"]
        input_gen_fn = test_case["input_gen_fn"]

        # =====================================================================
        # Step 1: L3 Numerical Consistency Guard (Strict Pass/Fail)
        # =====================================================================
        inputs = input_gen_fn()
        expected = baseline_fn(*inputs)
        actual = candidate_op(*inputs)

        # Check tensor or numpy
        if isinstance(expected, np.ndarray):
            matches = np.allclose(actual, expected, atol=atol, rtol=rtol)
            max_diff = float(np.max(np.abs(actual - expected)))
            if not matches:
                return {
                    "success": False,
                    "l3_passed": False,
                    "error": (
                        f"L3 VETO: Max diff={max_diff:.4e} exceeds "
                        f"tolerance (atol={atol}, rtol={rtol})."
                    ),
                }
        elif hasattr(expected, "allclose"):
            import torch

            matches = torch.allclose(actual, expected, atol=atol, rtol=rtol)
            max_diff = torch.max(torch.abs(actual - expected)).item()
            if not matches:
                return {
                    "success": False,
                    "l3_passed": False,
                    "error": (
                        f"L3 VETO: Max diff={max_diff:.4e} exceeds "
                        f"tolerance (atol={atol}, rtol={rtol})."
                    ),
                }
        else:
            if actual != expected:
                return {
                    "success": False,
                    "l3_passed": False,
                    "error": "L3 VETO: Output does not match expected result.",
                }

        # =====================================================================
        # Step 2: L0 Pure Operator Benchmark (Warmup + Statistical Timing)
        # =====================================================================
        import contextlib

        import torch

        with contextlib.suppress(ImportError):
            import torch_npu  # noqa: F401

        # Case A: Real NPU Execution
        if hw_info.has_npu and hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()

            # Warmup
            for _ in range(10):
                _ = candidate_op(*inputs)
            torch.npu.synchronize()

            # If NPUGraph is requested and supported
            if use_npu_graph and hasattr(torch.npu, "NPUGraph"):
                repeat_in_graph = 20
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    for _ in range(repeat_in_graph):
                        _ = candidate_op(*inputs)
                torch.npu.synchronize()

                for _ in range(5):
                    graph.replay()
                torch.npu.synchronize()

                durations = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    graph.replay()
                    torch.npu.synchronize()
                    t1 = time.perf_counter()
                    durations.append(((t1 - t0) * 1e6) / repeat_in_graph)
            else:
                # Eager stream timing
                durations = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    _ = candidate_op(*inputs)
                    torch.npu.synchronize()
                    t1 = time.perf_counter()
                    durations.append((t1 - t0) * 1e6)

        # Case B: Offline / CPU Simulation
        else:
            for _ in range(10):
                _ = candidate_op(*inputs)

            durations = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = candidate_op(*inputs)
                t1 = time.perf_counter()
                durations.append((t1 - t0) * 1e6)

        durations.sort()
        median_us = durations[len(durations) // 2]
        p90_us = durations[int(len(durations) * 0.9)]

        timing_mode = "NPUGraph" if (hw_info.has_npu and use_npu_graph) else "Eager"
        return {
            "success": True,
            "l3_passed": True,
            "l0_median_us": round(median_us, 2),
            "l0_p90_us": round(p90_us, 2),
            "timing_mode": timing_mode,
            "error": "",
        }

    except Exception as e:
        return {
            "success": False,
            "l3_passed": False,
            "error": f"Evaluation exception: {type(e).__name__}: {str(e)}",
        }
    finally:
        sys.modules.pop("candidate_module", None)
        if "torch" in sys.modules:
            try:
                import torch

                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.empty_cache()
            except Exception:
                pass
        if os.path.exists(temp_path):
            os.remove(temp_path)
