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


def write_manifest(root: Path) -> Path:
    return write_json(
        root / "collection_manifest.json",
        {
            "schema": "vllm-hust/collection/collection-manifest/v1",
            "produced_at": "2026-11-17T00:00:00Z",
            "host_commits": {"vllm": "a" * 40, "vllm-ascend": "b" * 40},
            "collection_shape": {"server_mode": "graph"},
            "artifacts": ["op_statistic.csv"],
        },
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
            ("op_statistic.csv", "hotspot-statistic"),
            ("kernel_details.csv", "kernel-details"),
            ("api_statistic.csv", "api-statistic"),
            ("communication_matrix.json", "communication-matrix"),
            ("collection_manifest.json", "collection-manifest"),
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

    def test_raw_profiler_state_is_not_a_contract_artifact(self, tmp_path):
        """Raw trace and profiler bookkeeping sit beside the artifacts."""
        raw = tmp_path / "hotspot/profile/ASCEND_PROFILER_OUTPUT"
        raw.mkdir(parents=True)
        for name in (
            "trace_view.json",
            "ascend_pytorch_profiler_0.db",
            "analyse.done",
            "PROF_000001_x",
            "task_time.csv",
            "operator_details.csv",
            "profiler_metadata.json",
        ):
            (raw / name).write_text("x")
        report = inspect(tmp_path)
        assert report.tolerated == 0
        assert report.skipped == 7

    def test_artifacts_beside_raw_files_are_still_checked(self, tmp_path):
        """Excluding the directory would hide these; excluding by name does not."""
        raw = tmp_path / "hotspot/profile/ASCEND_PROFILER_OUTPUT"
        raw.mkdir(parents=True)
        (raw / "trace_view.json").write_text("x")
        (raw / "op_statistic.csv").write_text(
            "OP Type,Count,Total Time(us)\nGroupedMatmul,10,20.0\n"
        )
        report = inspect(tmp_path)
        assert report.tolerated == 1
        assert "hotspot-statistic" in report.kinds_seen
        assert report.ok, report.findings

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
        write_manifest(tmp_path)
        report = inspect(tmp_path, strict=True)
        assert not report.ok
        assert any(f.problem == "no schema field" for f in report.findings)

    def test_strict_requires_a_manifest(self, tmp_path):
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
        assert not report.ok
        assert any("must declare itself" in f.problem for f in report.findings)

    def test_strict_accepts_a_declared_archive(self, tmp_path):
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
        write_manifest(tmp_path)
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
        write_manifest(tmp_path)
        report = inspect(tmp_path, strict=True)
        assert not report.ok
        assert any("expected" in f.problem for f in report.findings)

    def test_native_profiler_artifacts_need_no_schema(self, tmp_path):
        """CSV and profiler-written JSON cannot carry a schema; fields are enough."""
        path = tmp_path / "hotspot/op_statistic_rank0.csv"
        path.parent.mkdir(parents=True)
        path.write_text("OP Type,Count,Total Time(us)\nGroupedMatmul,10,20.0\n")
        write_manifest(tmp_path)
        report = inspect(tmp_path, strict=True)
        assert report.ok, report.findings
        assert "hotspot-statistic" in report.kinds_seen


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


API_COLUMNS = ("Device_id", "Level", "API Name", "Time(us)", "Count", "Avg(us)")


def write_api(path: Path, rows: list[tuple[str, str, float, int]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=API_COLUMNS)
        writer.writeheader()
        for level, name, total, count in rows:
            writer.writerow(
                {
                    "Device_id": 0,
                    "Level": level,
                    "API Name": name,
                    "Time(us)": total,
                    "Count": count,
                    "Avg(us)": total / count if count else 0.0,
                }
            )
    return path


KERNEL_COLUMNS = ("Device_id", "Name", "Duration(us)", "Wait Time(us)")


def write_kernel_details(path: Path, rows: list[tuple[str, float, float]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KERNEL_COLUMNS)
        writer.writeheader()
        for name, duration, wait in rows:
            writer.writerow(
                {
                    "Device_id": 0,
                    "Name": name,
                    "Duration(us)": duration,
                    "Wait Time(us)": wait,
                }
            )
    return path


class TestApiView:
    def test_ranks_by_total_time_and_derives_average(self, tmp_path):
        write_api(
            tmp_path / "profile/api_statistic_rank0.csv",
            [
                ("acl", "aclrtLaunchKernelWithHostArgs", 120.0, 20),
                ("acl", "aclrtSynchronizeEvent", 2400.0, 4),
            ],
        )
        rows = Archive(tmp_path).api()
        assert rows is not None
        assert [row["api"] for row in rows] == [
            "aclrtSynchronizeEvent",
            "aclrtLaunchKernelWithHostArgs",
        ]
        # A blocking synchronise is visible as a large average, not a large count.
        assert rows[0]["avg_us"] == pytest.approx(600.0)
        assert rows[1]["count"] == 20

    def test_absent_artifact_returns_none(self, tmp_path):
        assert Archive(tmp_path).api() is None

    def test_survives_rows_with_unparseable_numbers(self, tmp_path):
        path = tmp_path / "api_statistic.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "Device_id,Level,API Name,Time(us),Count\n"
            "0,acl,good_call,100.0,2\n"
            "0,acl,bad_call,N/A,2\n"
        )
        rows = Archive(tmp_path).api()
        assert rows is not None
        assert [row["api"] for row in rows] == ["good_call"]


class TestKernelWaitView:
    def test_flags_synchronisation_points_by_wait_over_duration(self, tmp_path):
        write_kernel_details(
            tmp_path / "profile/ASCEND_PROFILER_OUTPUT/kernel_details.csv",
            [
                ("busy_kernel", 1000.0, 10.0),
                ("sync_point", 5.0, 500.0),
                ("sync_point", 5.0, 500.0),
            ],
        )
        rows = Archive(tmp_path).kernel_wait()
        assert rows is not None
        assert rows[0]["kernel"] == "sync_point"
        assert rows[0]["count"] == 2
        assert rows[0]["wait_us"] == pytest.approx(1000.0)
        # wait dwarfs duration: a synchronisation point, not compute
        assert rows[0]["wait_over_duration"] == pytest.approx(100.0)
        assert rows[1]["wait_over_duration"] == pytest.approx(0.01)

    def test_absent_artifact_returns_none(self, tmp_path):
        assert Archive(tmp_path).kernel_wait() is None

    def test_zero_duration_does_not_divide_by_zero(self, tmp_path):
        write_kernel_details(tmp_path / "kernel_details.csv", [("empty", 0.0, 50.0)])
        rows = Archive(tmp_path).kernel_wait()
        assert rows is not None
        assert rows[0]["wait_over_duration"] == 0.0
