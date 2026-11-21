"""Offline checks for Recurrent closure; no host, NPU, or archived traces required."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analysis import gdn_recurrent_closeout as closure  # noqa: E402
from tools.analysis import gdn_recurrent_trace as trace  # noqa: E402


def test_paired_difference_is_not_difference_of_medians():
    sample = {"off": [1, 5, 10, 30] * 25, "on": [21, 3, 8, 29] * 25}
    result = closure.timing(sample)
    assert result["on_faster"] == 75 and result["paired_median_on_minus_off_us"] == -1.5
    sample["on"][0] = float("nan")
    with pytest.raises(ValueError):
        closure.timing(sample)


def test_only_ordinary_matching_rows_contribute_to_budget(tmp_path):
    p = tmp_path / "kernel_details.csv"
    p.write_text(
        "Device_id,Start Time(us),Duration(us),Model ID,Type,Input Shapes\n"
        '0,0,20,4294967295,RecurrentGatedDeltaRule,"1,8,128"\n'
        '0,20,3,11,RecurrentGatedDeltaRule,"1,8,128"\n'
        '0,23,5,12,RecurrentGatedDeltaRule,"2,8,128"\n'
        '0,28,12,11,Other,X\n0,40,60,18,RecurrentGatedDeltaRule,"8,8,128"\n'
    )
    expected = {"1": 1, "2": 1, "8": 1}
    data = closure.rank_budget(p, "0", expected)
    assert data["eligible_main_us"] == 8 and data["eligible_main_over_span_pct"] == 8
    assert data["eligible_main_over_same_graph_kernel_sum_pct"]["1"] == 20
    with pytest.raises(ValueError, match="rank mismatch"):
        closure.rank_budget(p, "1", expected)
    with pytest.raises(ValueError, match="unaccounted"):
        closure.rank_budget(p, "0", {"1": 2})


def test_incomplete_trace_is_not_a_zero_time_measurement(tmp_path):
    (tmp_path / "kernel_details.csv").write_text("Name,Start Time(us),Duration(us)\n")
    (tmp_path / "trace_view.json").write_text("[]")
    with pytest.raises(ValueError, match="missing or extra"):
        trace.window(tmp_path, True)


def test_diagnostic_not_promoted_to_service_conclusion():
    data = json.loads(closure.RECORD.read_bytes())
    assert (
        data["status"] == "closed_default_off"
        and data["verdict"]["service"] == "not_measured"
    )
    for row in data["measurements"]["local_ab"].values():
        assert row["main_on_range_us"][0] > row["main_off_range_us"][1]
        assert row["gate_expansion_on_us"] > 0
    assert data["native_intra_head_review"]["status"] == "unmeasured_not_scheduled"
