"""路由对账的 CPU 门禁：通过样例逐槽位唯一对上，歧义样例 fail closed。

样例与字段表来自 docs/optimization-history/gmm-route-reconciliation.json；
本测试只跑纯 CPU 合成数据，不占设备、不起服务。
"""

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.analysis import gmm_route_reconciliation as grr  # noqa: E402

RECORD = ROOT / "docs/optimization-history/gmm-route-reconciliation.json"


def record():
    return json.loads(RECORD.read_bytes())


def pass_sample():
    return copy.deepcopy(record()["samples"]["pass"])


def find_gl(sample, sid, layer, rank):
    for item in sample["group_lists"]:
        if (item["step_id"], item["layer"], item["physical_rank"]) == (
            sid,
            layer,
            rank,
        ):
            return item
    raise AssertionError(f"no group_list for {(sid, layer, rank)}")


def find_step(sample, sid):
    for step in sample["steps"]:
        if step["step_id"] == sid:
            return step
    raise AssertionError(f"no step {sid}")


def rejects(sample, code):
    with pytest.raises(grr.Rejected) as info:
        grr.reconcile(sample)
    assert info.value.code == code


def test_pass_sample_reconciles_per_slot():
    result = grr.reconcile(pass_sample())
    assert result["status"] == "reconciled"
    assert result["valid_tokens"] == 4
    assert result["padded_rows"] == 1
    assert result["out_of_cohort_steps"] == [0, 1]
    assert result["out_of_cohort_group_lists"] == 0
    assert result["trailing_not_forwarded"] == {"R0": 4}
    assert len(result["calls"]) == 12
    assert result["calls"]["(2, 0, 0)"] == [1, 0, 1]
    assert result["calls"]["(2, 1, 1)"] == [1, 0, 1]
    assert result["calls"]["(3, 0, 1)"] == [1, 1, 0]
    assert result["calls"]["(3, 1, 1)"] == [1, 0, 1]
    assert result["calls"]["(4, 0, 0)"] == [1, 1, 0]
    assert result["calls"]["(4, 1, 0)"] == [1, 0, 1]


def test_zero_route_rank_counts_are_reported_not_rejected():
    """回归：全零 counts 必须实报并通过，不得被当成缺数据而拒绝。"""
    result = grr.reconcile(pass_sample())
    assert result["calls"]["(3, 0, 0)"] == [0, 0, 0]
    assert result["calls"]["(3, 1, 0)"] == [0, 0, 0]
    assert result["calls"]["(4, 0, 1)"] == [0, 0, 0]
    assert result["calls"]["(4, 1, 1)"] == [0, 0, 0]
    assert result["status"] == "reconciled"


@pytest.mark.parametrize("sid,layer,rank", [(3, 0, 0), (4, 1, 1), (2, 1, 0)])
def test_missing_group_list_record_rejected(sid, layer, rank):
    """回归：期望键完整枚举；零路由 rank 漏整份记录也必须拒绝。"""
    sample = pass_sample()
    sample["group_lists"] = [
        gl
        for gl in sample["group_lists"]
        if (gl["step_id"], gl["layer"], gl["physical_rank"]) != (sid, layer, rank)
    ]
    rejects(sample, "missing_group_list")


def test_same_total_different_distribution_rejected():
    sample = pass_sample()
    find_gl(sample, 3, 0, 1)["counts"] = [2, 0, 0]
    rejects(sample, "count_mismatch")


def test_padding_style_shift_is_not_absorbed():
    """真实分发混入 padding 贡献时不得抹除重试，逐槽位失配即拒。"""
    sample = pass_sample()
    find_gl(sample, 2, 0, 0)["counts"] = [2, 1, 2]
    rejects(sample, "count_mismatch")


@pytest.mark.parametrize("value", [True, 1.0, -1, "1", None])
def test_counts_must_be_non_negative_strict_int(value):
    sample = pass_sample()
    find_gl(sample, 2, 0, 0)["counts"] = [1, value, 1]
    rejects(sample, "bad_counts_value")


@pytest.mark.parametrize(
    "key,value",
    [
        ("group_list_type", True),
        ("group_list_type", 1.0),
        ("step_id", 2.0),
        ("step_id", True),
        ("physical_rank", 1.0),
        ("physical_rank", True),
    ],
)
def test_group_list_keys_reject_bool_and_float(key, value):
    """True==1 且 2.0 与 2 同哈希：布尔/浮点不得混进调用键。"""
    sample = pass_sample()
    sample["group_lists"][0][key] = value
    rejects(sample, "bad_group_list")


def test_ep_map_physical_rank_must_be_strict_non_negative():
    sample = pass_sample()
    sample["ep_map"]["ranks"][0]["physical_rank"] = -1
    rejects(sample, "bad_ep_map")
    sample = pass_sample()
    sample["ep_map"]["ranks"][0]["physical_rank"] = 0.0
    rejects(sample, "bad_ep_map")


@pytest.mark.parametrize(
    "rank_device",
    [
        {"0": 4, "1": 4},
        {"0": -1, "1": 6},
        {"01": 4, "1": 6},
        {"0": 4, "1": 6, "2": 8},
        {"0": 4.0, "1": 6},
    ],
)
def test_rank_device_is_one_rank_one_device(rank_device):
    sample = pass_sample()
    sample["window"]["rank_device"] = rank_device
    rejects(sample, "bad_window")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "f18cf803c5f63625e2c71253ddaf8b0bad0bad1",
        "F18CF803C5F63625E2C71253DDAF8B0BAD0BAD1A",
    ],
)
def test_host_commits_must_be_40_hex(value):
    sample = pass_sample()
    sample["window"]["host_commits"]["vllm"] = value
    rejects(sample, "host_mismatch")


@pytest.mark.parametrize("wid", [None, "", 0])
def test_window_id_must_be_non_empty_string(wid):
    sample = pass_sample()
    sample["window"]["window_id"] = wid
    rejects(sample, "bad_window")


def test_null_window_id_cannot_associate_with_null_blocks():
    """窗口与数据块的 window_id 同为 null 时不得互相“对上”。"""
    sample = pass_sample()
    sample["window"]["window_id"] = None
    sample["requests"][0]["window_id"] = None
    sample["ep_map"]["window_id"] = None
    rejects(sample, "bad_window")


def test_host_commits_must_be_complete_and_consistent():
    sample = pass_sample()
    del sample["window"]["host_commits"]["vllm-ascend"]
    rejects(sample, "host_mismatch")
    sample = pass_sample()
    sample["ep_map"]["host_commits"]["vllm"] = "0" * 40
    rejects(sample, "host_mismatch")


def test_ep_size_and_rank_device_are_required():
    sample = pass_sample()
    del sample["window"]["ep_size"]
    rejects(sample, "bad_window")
    sample = pass_sample()
    sample["window"]["ep_size"] = 3
    rejects(sample, "bad_window")
    sample = pass_sample()
    del sample["window"]["rank_device"]["1"]
    rejects(sample, "bad_window")
    sample = pass_sample()
    sample["window"]["rank_device"]["0"] = "4"
    rejects(sample, "bad_window")


def test_empty_window_rejections():
    sample = pass_sample()
    sample["requests"] = []
    rejects(sample, "empty_window")
    sample = pass_sample()
    for step in sample["steps"]:
        step["phase"] = "warmup"
        step["rows"] = []
    rejects(sample, "empty_window")
    sample = pass_sample()
    for sid in (2, 3, 4):
        find_step(sample, sid)["rows"] = [{"kind": "padded"}]
    rejects(sample, "empty_window")


def test_warmup_capture_cannot_be_declared_in_cohort():
    sample = pass_sample()
    sample["window"]["in_cohort_phases"] = ["prefill", "decode", "warmup"]
    rejects(sample, "phase_forbidden")


def test_ep_map_must_cover_all_experts_exactly_once():
    sample = pass_sample()
    del sample["ep_map"]["ranks"][0]["log2phy"]["5"]
    rejects(sample, "expert_unmapped")
    sample = pass_sample()
    sample["ep_map"]["ranks"][1]["log2phy"]["2"] = 0
    rejects(sample, "bad_ep_map")
    sample = pass_sample()
    sample["ep_map"]["ranks"][0]["log2phy"]["2"] = 3
    rejects(sample, "bad_ep_map")


def test_row_binding_strictness():
    sample = pass_sample()
    find_step(sample, 4)["rows"].append(
        {"request_id": "R0", "token_position": 4, "kind": "forwarded"}
    )
    rejects(sample, "row_out_of_range")
    sample = pass_sample()
    find_step(sample, 4)["rows"] = []
    rejects(sample, "tokens_not_covered")
    sample = pass_sample()
    find_step(sample, 3)["rows"].append(
        {"request_id": "R0", "token_position": 2, "kind": "forwarded"}
    )
    rejects(sample, "duplicate_token")
    sample = pass_sample()
    find_step(sample, 2)["rows"].append(
        {"request_id": "R0", "token_position": 2, "kind": "forwarded"}
    )
    rejects(sample, "duplicate_token")
    sample = pass_sample()
    find_step(sample, 3)["rows"][0]["request_id"] = "R9"
    rejects(sample, "unknown_request")


def test_padded_and_excluded_rows_cannot_hide_bindings():
    sample = pass_sample()
    find_step(sample, 2)["rows"][-1]["request_id"] = "R0"
    rejects(sample, "bad_row")
    sample = pass_sample()
    find_step(sample, 2)["rows"][-1]["token_position"] = 0
    rejects(sample, "bad_row")
    sample = pass_sample()
    find_step(sample, 3)["rows"][0]["kind"] = "unknown"
    rejects(sample, "bad_row")
    sample = pass_sample()
    find_step(sample, 0)["rows"] = [
        {"request_id": "R0", "token_position": 0, "kind": "forwarded"}
    ]
    rejects(sample, "row_outside_cohort")


def test_orphan_and_duplicate_group_list_records():
    sample = pass_sample()
    sample["group_lists"][0]["step_id"] = 9
    rejects(sample, "orphan_group_list")
    sample = pass_sample()
    sample["group_lists"].append(dict(sample["group_lists"][0]))
    rejects(sample, "duplicate_call_record")


@pytest.mark.parametrize(
    "key,value",
    [
        ("group_list_type", 0),
        ("layer", 9),
        ("physical_rank", 7),
        ("counts", [1, 0]),
    ],
)
def test_group_list_structure_strictness(key, value):
    sample = pass_sample()
    sample["group_lists"][0][key] = value
    rejects(sample, "bad_group_list")


@pytest.mark.parametrize(
    "key,value",
    [
        ("routed_experts", [[[0, 5], [3, 0]]]),
        (
            "routed_experts",
            [[[0, 5], [3, 0]], [[2, 4], [1, 4]], [[1, 2], [4, 1]], [[9, 3], [5, 0]]],
        ),
        (
            "routed_experts",
            [[[0, 0], [3, 0]], [[2, 4], [1, 4]], [[1, 2], [4, 1]], [[0, 3], [5, 0]]],
        ),
        ("prompt_tokens", 0),
    ],
)
def test_request_fields_are_strict(key, value):
    sample = pass_sample()
    sample["requests"][0][key] = value
    rejects(sample, "bad_request")


def test_duplicate_request_id_and_window_mismatch_rejected():
    sample = pass_sample()
    sample["requests"].append(dict(sample["requests"][0]))
    rejects(sample, "bad_request")
    sample = pass_sample()
    sample["ep_map"]["window_id"] = "wr-other"
    rejects(sample, "window_mismatch")


def test_out_of_cohort_group_list_is_excluded_not_fatal():
    sample = pass_sample()
    sample["group_lists"].append(
        {
            "window_id": "wr-sample-000",
            "step_id": 0,
            "layer": 0,
            "physical_rank": 0,
            "group_list_type": 1,
            "counts": [9, 9, 9],
        }
    )
    result = grr.reconcile(sample)
    assert result["status"] == "reconciled"
    assert result["out_of_cohort_group_lists"] == 1


def test_must_reject_sample_fails_closed_with_declared_code():
    reject = record()["samples"]["must_reject"]
    with pytest.raises(grr.Rejected) as info:
        grr.reconcile(copy.deepcopy(reject["sample"]))
    assert info.value.code == reject["expected_reject_code"] == "duplicate_token"


def test_cli_reproduces_contract_command():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.analysis.gmm_route_reconciliation",
            "docs/optimization-history/gmm-route-reconciliation.json",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "pass sample reconciled" in result.stdout
    assert "duplicate_token" in result.stdout
