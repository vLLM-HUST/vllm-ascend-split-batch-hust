#!/usr/bin/env python3
"""Dual-mode L0 and L3 inner-loop evaluator with dynamic-shape recompilation guard."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
import time
from typing import Any

import numpy as np

from tools.op_agent.guard import check_npu_environment


def _check_outputs_match(
    actual: Any, expected: Any, atol: float, rtol: float, label: str = ""
) -> str | None:
    """Return None when outputs match within tolerance, else an error message."""
    prefix = f"[{label}] " if label else ""
    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            return f"L3 VETO: {prefix}Return structure mismatch (expected tuple)."
        import torch

        for i, (act_item, exp_item) in enumerate(zip(actual, expected, strict=True)):
            if hasattr(exp_item, "allclose"):
                if not torch.allclose(act_item, exp_item, atol=atol, rtol=rtol):
                    diff = torch.max(torch.abs(act_item - exp_item)).item()
                    return (
                        f"L3 VETO: {prefix}Element {i} max diff={diff:.4e} "
                        "exceeds tolerance."
                    )
            elif np.any(act_item != exp_item):
                return f"L3 VETO: {prefix}Element {i} mismatch."
        return None

    if isinstance(expected, np.ndarray):
        matches = np.allclose(actual, expected, atol=atol, rtol=rtol)
        if not matches:
            max_diff = float(np.max(np.abs(actual - expected)))
            return (
                f"L3 VETO: {prefix}Max diff={max_diff:.4e} exceeds "
                f"tolerance (atol={atol}, rtol={rtol})."
            )
        return None

    if hasattr(expected, "allclose"):
        import torch

        matches = torch.allclose(actual, expected, atol=atol, rtol=rtol)
        if not matches:
            max_diff = torch.max(torch.abs(actual - expected)).item()
            return (
                f"L3 VETO: {prefix}Max diff={max_diff:.4e} exceeds "
                f"tolerance (atol={atol}, rtol={rtol})."
            )
        return None

    if actual != expected:
        return f"L3 VETO: {prefix}Output does not match expected result."
    return None


def _measure_single_input(
    fn: Any,
    inputs: tuple[Any, ...],
    has_npu: bool,
    use_npu_graph: bool,
    iters: int,
) -> tuple[float, float]:
    """Measure (median_us, p90_us) of `fn(*inputs)`."""
    if has_npu:
        import torch

        torch.npu.synchronize()
        for _ in range(10):
            _ = fn(*inputs)
        torch.npu.synchronize()

        if use_npu_graph and hasattr(torch.npu, "NPUGraph"):
            repeat_in_graph = 20
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                for _ in range(repeat_in_graph):
                    _ = fn(*inputs)
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
            durations = []
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = fn(*inputs)
                torch.npu.synchronize()
                t1 = time.perf_counter()
                durations.append((t1 - t0) * 1e6)
    else:
        for _ in range(10):
            _ = fn(*inputs)
        durations = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn(*inputs)
            t1 = time.perf_counter()
            durations.append((t1 - t0) * 1e6)

    durations.sort()
    median_us = durations[len(durations) // 2]
    p90_us = durations[int(len(durations) * 0.9)]
    return round(median_us, 2), round(p90_us, 2)


def run_candidate_evaluation(
    kernel_code: str,
    test_case: dict[str, Any],
    use_npu_graph: bool = True,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    iters: int | None = None,
) -> dict[str, Any]:
    """Execute L3 numerical check, dynamic-shape recompile check, and L0 benchmark.

    Supports single-shape cases as well as multi-bucket shape sweeps
    (`bucket_input_gen_fn` + `shape_buckets` + `dynamic_shape_probes`).
    """
    if "TORCH_DEVICE_BACKEND_AUTOLOAD" not in os.environ:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

    run_iters = iters if iters is not None else int(test_case.get("iters", 50))
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
        bucket_gen = test_case.get("bucket_input_gen_fn")
        shape_buckets: list[int] = list(test_case.get("shape_buckets") or [])
        dynamic_probes: list[int] = list(test_case.get("dynamic_shape_probes") or [])
        recompile_budget_ms = float(test_case.get("recompile_budget_ms", 250.0))

        # =====================================================================
        # Step 1: L3 Numerical Consistency Guard (Default + Shape Buckets)
        # =====================================================================
        inputs = input_gen_fn()
        expected = baseline_fn(*inputs)
        actual = candidate_op(*inputs)
        err = _check_outputs_match(actual, expected, atol=atol, rtol=rtol)
        if err is not None:
            return {"success": False, "l3_passed": False, "error": err}

        if callable(bucket_gen):
            for b_tok in shape_buckets:
                b_in = bucket_gen(b_tok)
                b_err = _check_outputs_match(
                    candidate_op(*b_in),
                    baseline_fn(*b_in),
                    atol=atol,
                    rtol=rtol,
                    label=f"T={b_tok}",
                )
                if b_err is not None:
                    return {"success": False, "l3_passed": False, "error": b_err}

        # =====================================================================
        # Step 2: Dynamic-Shape Zero-Recompile Gate (Unseen Token Counts)
        # =====================================================================
        has_real_npu = False
        with contextlib.suppress(ImportError):
            import torch
            import torch_npu  # noqa: F401

            has_real_npu = (
                hw_info.has_npu
                and hasattr(torch, "npu")
                and bool(torch.npu.is_available())
            )

        dynamic_probe_ms: dict[int, float] = {}
        if callable(bucket_gen) and dynamic_probes:
            for p_tok in dynamic_probes:
                p_in = bucket_gen(p_tok)
                p_exp = baseline_fn(*p_in)
                if has_real_npu:
                    import torch

                    torch.npu.synchronize()
                t0 = time.perf_counter()
                p_act = candidate_op(*p_in)
                if has_real_npu:
                    import torch

                    torch.npu.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                dynamic_probe_ms[p_tok] = round(elapsed_ms, 2)

                p_err = _check_outputs_match(
                    p_act, p_exp, atol=atol, rtol=rtol, label=f"probe_T={p_tok}"
                )
                if p_err is not None:
                    return {"success": False, "l3_passed": False, "error": p_err}

                if elapsed_ms > recompile_budget_ms:
                    return {
                        "success": False,
                        "l3_passed": False,
                        "recompile_veto": True,
                        "dynamic_probe_ms": dynamic_probe_ms,
                        "error": (
                            f"RECOMPILE VETO: Unseen shape T={p_tok} took "
                            f"{elapsed_ms:.1f} ms (> {recompile_budget_ms:.1f} ms). "
                            "Mark dynamic dimensions/strides with `do_not_specialize`."
                        ),
                    }

        # =====================================================================
        # Step 3: L0 Pure Operator Benchmark (Default + Optional Multi-Bucket)
        # =====================================================================
        median_us, p90_us = _measure_single_input(
            candidate_op,
            inputs,
            has_npu=has_real_npu,
            use_npu_graph=use_npu_graph,
            iters=run_iters,
        )

        bucket_latencies_us: dict[int, float] = {}
        if callable(bucket_gen) and shape_buckets:
            bucket_iters = max(10, run_iters // 2)
            for b_tok in shape_buckets:
                b_med, _ = _measure_single_input(
                    candidate_op,
                    bucket_gen(b_tok),
                    has_npu=has_real_npu,
                    use_npu_graph=use_npu_graph,
                    iters=bucket_iters,
                )
                bucket_latencies_us[b_tok] = b_med

        timing_mode = "NPUGraph" if (has_real_npu and use_npu_graph) else "Eager"
        return {
            "success": True,
            "l3_passed": True,
            "l0_median_us": median_us,
            "l0_p90_us": p90_us,
            "bucket_latencies_us": bucket_latencies_us,
            "dynamic_probe_ms": dynamic_probe_ms,
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
            with contextlib.suppress(Exception):
                import torch

                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.empty_cache()
        if os.path.exists(temp_path):
            os.remove(temp_path)
