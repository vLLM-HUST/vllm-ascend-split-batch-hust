"""Guards for the collection-artifact contract and its read-only views.

The fixtures here are synthetic on purpose: CI has no access to the archived
collection runs, and the contract has to be checkable from the shape of the
artifacts alone.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.operator_tools.artifact_contract import KINDS, classify, inspect
from tools.operator_tools.artifact_view import (
    NOISE_FLOOR_PCT,
    Archive,
    e2e_view,
    hotspot_view,
    normalize,
)

STEP_TRACE_COLUMNS = (
    "Device_id",
    "Step",
    "Computing",
    "Communication(Not Overlapped)",
    "Overlapped",
    "Communication",
    "Free",
    "Stage",
    "Bubble",
)


def write_json(path: Path, body) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False))
    return path


def write_step_trace(path: Path, computing: float, free: float, comm: float) -> Path:
    stage = computing + free + comm
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_TRACE_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "Device_id": 0,
                "Step": "",
                "Computing": computing,
                "Communication(Not Overlapped)": comm,
                "Overlapped": 0.0,
                "Communication": comm,
                "Free": free,
                "Stage": stage,
                "Bubble": 0.0,
            }
        )
    return path


def write_pipes(
    path: Path, op: str, calls: int, total: float, median: float, share: float
) -> Path:
    return write_json(
        path,
        {
            "compute_total_us": total,
            "ops": [
                {
                    "op": op,
                    "calls": calls,
                    "total_us": total,
                    "share_pct": share,
                    "median_us": median,
                    "busy_pipe_pct": 42.0,
                    "bottleneck_form": "aic_mte2_bound",
                }
            ],
        },
    )


def write_checklist(path: Path, op: str, share: float, sol) -> Path:
    return write_json(
        path,
        [
            {
                "op": op,
                "share_pct": share,
                "measured_us": 1000.0,
                "calls": 10,
                "section": "decidable" if sol is not None else "not_modelable",
                "sol_pct": sol,
            }
        ],
    )


def build_archive(
    root: Path, computing: float = 250.0, free: float = 700.0, comm: float = 50.0
) -> Path:
    write_step_trace(root / "hotspot/step_trace_rank0.csv", computing, free, comm)
    write_pipes(
        root / "hotspot/pipes_rank0.json",
        "aclnnMatmul_MatMulCommon_MatMulV2",
        100,
        500.0,
        3.0,
        50.0,
    )
    write_json(
        root / "hotspot/hotspots.json",
        [{"op": "MatMulV2", "count": 100, "total_us": 500.0, "share_pct": 50.0}],
    )
    write_checklist(root / "sol/checklist.json", "MatMulV2", 50.0, 40.0)
    return root


class TestNormalize:
    def test_aclnn_call_name_collapses_to_short_name(self):
        assert normalize("aclnnMatmul_MatMulCommon_MatMulV2") == "MatMulV2"
        assert (
            normalize("aclnnGroupedMatmulV5_GroupedMatmul_GroupedMatmul")
            == "GroupedMatmul"
        )
        assert normalize("aclnnInplaceCopy_CastAiCore_Cast") == "Cast"

    def test_short_names_pass_through(self):
        assert normalize("AddRmsNormBias") == "AddRmsNormBias"
        assert normalize("SwiGlu") == "SwiGlu"

    def test_kernel_names_with_underscores_are_untouched(self):
        assert (
            normalize("_layer_norm_fwd_1pass_kernel") == "_layer_norm_fwd_1pass_kernel"
        )


class TestClassification:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("hotspots.json", "hotspot-summary"),
            ("pipes_rank0.json", "hotspot-pipes"),
            ("pipes_sharegpt.json", "hotspot-pipes"),
            ("op_statistic_rank0.csv", "hotspot-statistic"),
            ("step_trace_rank1.csv", "step-trace"),
            ("sol_list.json", "sol-plan"),
            ("checklist.json", "sol-checklist"),
            ("sol_gap_bench.json", "sol-gap-bench"),
            ("window.json", "peak-calibration"),
            ("12_hbm_ceiling.json", "peak-calibration"),
            ("invocation.json", "round-invocation"),
            ("exit.json", "round-exit"),
            ("owner.json", "round-owner"),
            ("responses.json", "round-response"),
            ("greedy.json", "round-output"),
        ],
    )
    def test_known_artifacts_are_recognised(self, name, expected):
        kind = classify(name)
        assert kind is not None, name
        assert kind.name == expected

    def test_unrelated_names_are_not_claimed(self):
        assert classify("server.log") is None
        assert classify("analyse.log") is None
        assert classify("profiler_metadata.json") is None

    def test_every_kind_has_a_note(self):
        for kind in KINDS:
            assert kind.note, f"{kind.name} needs a note explaining what it carries"


class TestLegacyInspection:
    def test_well_formed_archive_passes(self, tmp_path):
        build_archive(tmp_path)
        report = inspect(tmp_path, strict=False)
        assert report.ok, report.findings
        assert report.tolerated >= 4
        assert "hotspot-pipes" in report.kinds_seen
        assert "step-trace" in report.kinds_seen

    def test_missing_required_field_is_an_error(self, tmp_path):
        write_json(tmp_path / "sol/sol_list.json", {"decidable": []})
        report = inspect(tmp_path, strict=False)
        assert not report.ok
        assert any("fields missing" in f.problem for f in report.findings)

    def test_missing_item_field_is_an_error(self, tmp_path):
        write_json(
            tmp_path / "hotspot/hotspots.json",
            [{"op": "X", "count": 1}],
        )
        report = inspect(tmp_path, strict=False)
        assert not report.ok
        assert any("item fields missing" in f.problem for f in report.findings)

    def test_csv_without_expected_columns_is_an_error(self, tmp_path):
        path = tmp_path / "hotspot/step_trace_rank0.csv"
        path.parent.mkdir(parents=True)
        path.write_text("a,b\n1,2\n")
        report = inspect(tmp_path, strict=False)
        assert not report.ok
        assert any("csv columns missing" in f.problem for f in report.findings)

    def test_body_that_is_not_a_list_is_an_error(self, tmp_path):
        write_json(tmp_path / "hotspot/hotspots.json", {"op": "X"})
        report = inspect(tmp_path, strict=False)
        assert not report.ok
        assert any("non-empty list" in f.problem for f in report.findings)


class TestStrictInspection:
    def test_strict_requires_schema(self, tmp_path):
        build_archive(tmp_path)
        report = inspect(tmp_path, strict=True)
        assert not report.ok
        assert any(f.problem == "no schema field" for f in report.findings)

    def test_strict_accepts_a_declared_schema(self, tmp_path):
        write_pipes(
            tmp_path / "hotspot/pipes_rank0.json",
            "aclnnMatmul_MatMulCommon_MatMulV2",
            1,
            1.0,
            1.0,
            100.0,
        )
        body = json.loads((tmp_path / "hotspot/pipes_rank0.json").read_text())
        body["schema"] = "vllm-hust/collection/hotspot-pipes/v1"
        body["produced_at"] = "2026-11-17T00:00:00Z"
        write_json(tmp_path / "hotspot/pipes_rank0.json", body)
        report = inspect(tmp_path, strict=True)
        assert report.ok, report.findings

    def test_strict_rejects_a_mismatched_schema(self, tmp_path):
        path = write_pipes(
            tmp_path / "hotspot/pipes_rank0.json",
            "aclnnX_Y_Z",
            1,
            1.0,
            1.0,
            100.0,
        )
        body = json.loads(path.read_text())
        body["schema"] = "vllm-hust/collection/hotspot-summary/v1"
        body["produced_at"] = "2026-11-17T00:00:00Z"
        write_json(path, body)
        report = inspect(tmp_path, strict=True)
        assert not report.ok
        assert any("expected" in f.problem for f in report.findings)


class TestArchiveViews:
    def test_wall_clock_splits_stage(self, tmp_path):
        build_archive(tmp_path, computing=250.0, free=700.0, comm=50.0)
        wall = Archive(tmp_path).wall_clock()
        assert wall is not None
        assert wall["stage_us"] == pytest.approx(1000.0)
        assert wall["computing_ratio"] == pytest.approx(0.25)
        assert wall["free_ratio"] == pytest.approx(0.70)

    def test_wall_clock_absent_returns_none(self, tmp_path):
        assert Archive(tmp_path).wall_clock() is None

    def test_hotspot_split_separates_long_calls(self, tmp_path):
        build_archive(tmp_path)
        rows = hotspot_view(Archive(tmp_path))
        assert len(rows) == 1
        row = rows[0]
        assert row["op"] == "MatMulV2"
        assert row["typical_us"] == pytest.approx(300.0)
        assert row["long_call_us"] == pytest.approx(200.0)

    def test_long_call_never_goes_negative(self, tmp_path):
        write_pipes(
            tmp_path / "hotspot/pipes_rank0.json",
            "aclnnX_Y_Z",
            10,
            5.0,
            9.0,
            100.0,
        )
        row = hotspot_view(Archive(tmp_path))[0]
        assert row["long_call_us"] == 0.0

    def test_hotspot_falls_back_to_summary_without_pipes(self, tmp_path):
        write_json(
            tmp_path / "hotspot/hotspots.json",
            [{"op": "SwiGlu", "count": 4, "total_us": 8.0, "share_pct": 12.5}],
        )
        rows = hotspot_view(Archive(tmp_path))
        assert rows[0]["op"] == "SwiGlu"
        assert rows[0]["long_call_us"] is None

    def test_e2e_upper_bound_uses_computing_ratio(self, tmp_path):
        build_archive(tmp_path, computing=250.0, free=700.0, comm=50.0)
        rows = e2e_view(Archive(tmp_path))
        assert len(rows) == 1
        row = rows[0]
        # share 50%, sol 40%, computing ratio 0.25
        # device gap = 50 * 0.6 = 30 ; end to end = 30 * 0.25 = 7.5
        assert row["device_gap_pct"] == pytest.approx(30.0)
        assert row["e2e_upper_pct"] == pytest.approx(7.5)
        assert row["decidable"] is True

    def test_operator_below_noise_floor_is_marked_undecidable(self, tmp_path):
        write_step_trace(tmp_path / "hotspot/step_trace_rank0.csv", 250.0, 700.0, 50.0)
        write_pipes(
            tmp_path / "hotspot/pipes_rank0.json",
            "aclnnMatmul_MatMulCommon_MatMulV2",
            100,
            500.0,
            3.0,
            5.0,
        )
        write_checklist(tmp_path / "sol/checklist.json", "MatMulV2", 5.0, 40.0)
        row = e2e_view(Archive(tmp_path))[0]
        # device gap = 5 * 0.6 = 3 ; end to end = 3 * 0.25 = 0.75, below the floor
        assert row["e2e_upper_pct"] == pytest.approx(0.75)
        assert row["e2e_upper_pct"] < NOISE_FLOOR_PCT
        assert row["decidable"] is False

    def test_e2e_skips_operators_without_a_roofline_number(self, tmp_path):
        write_step_trace(tmp_path / "hotspot/step_trace_rank0.csv", 250.0, 700.0, 50.0)
        write_pipes(
            tmp_path / "hotspot/pipes_rank0.json",
            "aclnnX_Y_Z",
            1,
            10.0,
            10.0,
            100.0,
        )
        write_checklist(tmp_path / "sol/checklist.json", "Z", 100.0, None)
        assert e2e_view(Archive(tmp_path)) == []

    def test_e2e_needs_both_artifacts(self, tmp_path):
        write_step_trace(tmp_path / "hotspot/step_trace_rank0.csv", 250.0, 700.0, 50.0)
        assert e2e_view(Archive(tmp_path)) == []

    def test_multiple_variants_record_provenance(self, tmp_path):
        write_step_trace(tmp_path / "hotspot/step_trace_random.csv", 100.0, 100.0, 0.0)
        write_step_trace(
            tmp_path / "hotspot/step_trace_sharegpt.csv", 200.0, 200.0, 0.0
        )
        archive = Archive(tmp_path)
        archive.wall_clock()
        assert "step_trace_*.csv" in archive.sources
        assert "1 of 2" in archive.sources["step_trace_*.csv"]
