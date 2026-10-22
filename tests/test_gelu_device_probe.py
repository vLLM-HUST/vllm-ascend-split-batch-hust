from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_probe():
    path = Path(__file__).resolve().parents[1] / "tools/probe_gelu_device.py"
    spec = importlib.util.spec_from_file_location("gelu_device_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_import_is_device_independent():
    before = set(sys.modules)
    load_probe()
    loaded = set(sys.modules) - before
    assert not loaded & {"torch", "torch_npu", "vllm", "triton"}


def test_precision_gate_matches_float_compute_bfloat16_row():
    probe = load_probe()
    assert probe.PRECISION == {
        "rtol": 2.0**-6,
        "atol": 2.0**-6,
        "required_matched_ratio": 0.99,
        "max_abs_error_limit": 1.0,
    }


def test_summarize_reports_median_and_min():
    assert load_probe().summarize([0.3, 0.1, 0.2]) == {
        "median_ms_per_iter": 2.0,
        "min_ms_per_iter": 1.0,
        "repeats": 3,
    }


def test_mixed_tolerance_decisions():
    torch = __import__("pytest").importorskip("torch")
    probe = load_probe()
    golden = torch.zeros(1000)
    actual = golden + 2.0**-7
    result = probe.mixed_tolerance(actual, golden)
    assert result["is_pass"] and result["matched_ratio"] == 1.0
    spiked = golden.clone()
    spiked[:5] = 100.0
    result = probe.mixed_tolerance(spiked, golden)
    assert not result["is_pass"] and result["matched_ratio"] == 0.995
