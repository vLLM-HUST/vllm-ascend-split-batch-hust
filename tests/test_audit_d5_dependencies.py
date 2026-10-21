# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import csv
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/audit_d5_dependencies.py"
SPEC = importlib.util.spec_from_file_location("audit_d5", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def source(statements):
    return "class Layer:\n    def forward(self):\n" + "".join(
        f"        {s}\n" for s in statements
    )


def test_ordered_statement_witnesses():
    statements = ["x = self.producer(x)", "y = self.consumer(x)"]
    result = audit.inspect_contract(source(statements), "Layer", statements)
    assert [s["line"] for s in result["statements"]] == [3, 4]
    assert "def forward" in result["source"]


@pytest.mark.parametrize(
    "statements",
    [
        ["y = self.consumer(x)", "x = self.producer(x)"],
        ["x = self.producer(x)", "y = self.consumer(other)"],
        ["x = self.producer(x)"],
    ],
)
def test_missing_reordered_or_changed_statements_fail_closed(statements):
    with pytest.raises(ValueError, match="unrecognized"):
        audit.inspect_contract(
            source(statements),
            "Layer",
            ["x = self.producer(x)", "y = self.consumer(x)"],
        )


def test_inherited_forward_required():
    assert (
        audit.inspect_inheritance(
            "class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):\n    pass\n"
        )["inherits_forward_from"]
        == "Qwen3NextDecoderLayer"
    )


@pytest.mark.parametrize("body", ["def forward(self): pass", "pass"])
def test_override_or_changed_base_fails_closed(body):
    base = "Qwen3NextDecoderLayer" if "forward" in body else "Other"
    with pytest.raises(ValueError):
        audit.inspect_inheritance(f"class Qwen3_5DecoderLayer({base}):\n    {body}\n")


def test_shape_counts_preserve_formats_without_inventing_expert_counts(tmp_path):
    path = tmp_path / "kernels.csv"
    fields = ["Type", "Input Shapes", "Input Data Types", "Input Formats"]
    rows = [
        dict(zip(fields, [op, "8,2048;256,2048,512", "DT_BF16", "ND;NCL"], strict=True))
        for op in sorted(audit.TARGETS)
    ]
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows + rows[:1])
    result = audit.shape_summary(path)
    assert sum(row["count"] for row in result) == 4
    assert all(row["Input Formats"] == "ND;NCL" for row in result)
    assert all("expert_counts" not in row for row in result)


def test_missing_shape_fields_stop(tmp_path):
    path = tmp_path / "kernels.csv"
    path.write_text("Type\nGroupedMatmul\n")
    with pytest.raises(ValueError, match="missing shape metadata"):
        audit.shape_summary(path)


def test_missing_target_stops(tmp_path):
    path = tmp_path / "kernels.csv"
    path.write_text("Type,Input Shapes,Input Data Types,Input Formats\n")
    with pytest.raises(ValueError, match="missing target kernels"):
        audit.shape_summary(path)


def test_existing_evidence_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    path.write_text("preserved\n")
    monkeypatch.setattr(
        "sys.argv",
        [
            str(PATH),
            "--core",
            "/missing",
            "--ascend",
            "/missing",
            "--kernels",
            "/missing",
            "--out",
            str(path),
        ],
    )
    with pytest.raises(SystemExit) as error:
        audit.main()
    assert error.value.code == 2
    assert path.read_text() == "preserved\n"
