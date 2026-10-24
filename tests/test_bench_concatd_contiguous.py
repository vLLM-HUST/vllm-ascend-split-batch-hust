# SPDX-License-Identifier: Apache-2.0

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "contiguous_bench", TOOLS / "bench_concatd_contiguous.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 32])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_current_helper_matches_stock_without_shared_backing_assumption(
    tool, tokens, dtype
):
    dims = tool.bench.Dims()
    x = tool.bench.build_input(tokens, dims.width, dtype, 42, True, "cpu")
    check = tool.check_outputs(x, dtype, dims)
    assert check["ok"]
    assert check["comparison"]["ok"]
    assert check["contiguous"] and check["input_unmodified"]
    if tokens > 1:
        outputs = tool.adapter.unpack(x)
        assert len({t.untyped_storage().data_ptr() for t in outputs}) == 3


def write_source(tmp_path):
    rank = tmp_path / "rank0_ascend_pt"
    out = rank / "ASCEND_PROFILER_OUTPUT"
    out.mkdir(parents=True)
    with (out / "kernel_details.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Type", "Input Shapes", "Input Data Types"])
        w.writerow(
            [
                "RecurrentGatedDeltaRule",
                "16,8,128;16,8,128;16,16,128;16,16",
                "DT_BF16;DT_BF16;DT_BF16;FLOAT",
            ]
        )
    path = tmp_path / "source-opstat.json"
    path.write_text(
        json.dumps(
            {
                "asc_dir": str(rank),
                "cohort": {"model": "Qwen3.5-35B-A3B", "precision": "bf16"},
            }
        )
    )
    return path


def test_shape_source_requires_actual_csv_match(tool, tmp_path):
    path = write_source(tmp_path)
    doc = tool.shape_source(path, tool.bench.Dims())
    assert doc["decode_calls_by_tokens"] == {16: 1}
    assert doc["kernel_details"]["sha256"]
    bad = json.loads(path.read_text())
    bad["cohort"]["precision"] = "fp16"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        tool.shape_source(path, tool.bench.Dims())


def test_paired_timing_and_consumer_gate(tool, monkeypatch):
    values = iter([100.0, 50.0, 50.0, 100.0, 100.0, 50.0])
    monkeypatch.setattr(
        tool.bench, "timed", lambda fn, n, device: {"us_host_total": next(values)}
    )
    args = SimpleNamespace(seed=42, iters=1, repeats=3, warmup=0)
    cell = tool.run_cell(
        16,
        args,
        tool.bench.Dims(),
        {"decode_calls_by_tokens": {16: 1}},
        "cpu",
        lambda x: x,
    )
    assert cell["ok"] and cell["shape_role"] == "measured-shape"
    assert cell["l2norm_bitwise"] == [True, True]
    assert cell["performance"]["paired_delta_pct"] == [-50.0, -50.0, -50.0]


def test_failed_consumer_stops_before_timing(tool, monkeypatch):
    calls = iter([False, True, False, True])

    def normalize(x):
        return x + 1 if next(calls) else x

    def timing(*args):
        raise AssertionError("must not time a failed numerical cell")

    monkeypatch.setattr(tool.bench, "timed", timing)
    args = SimpleNamespace(seed=42, iters=1, repeats=3, warmup=0)
    cell = tool.run_cell(
        16,
        args,
        tool.bench.Dims(),
        {"decode_calls_by_tokens": {16: 1}},
        "cpu",
        normalize,
    )
    assert not cell["ok"]
