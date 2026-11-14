"""`DELIVERY_INDEX.json` 的状态标签必须被 `ops/<operator>/evidence.json` 支撑。

为什么要有这个测试：索引里的 `serving_status`、`l3_numerical_guard`、`status`
是**结论标签**，分别对应四级阶梯里的不同层级。曾经出现过的错误形态是把
L0 微基准的加速比直接写成 `serving_status: verified_faster`，而同一算子的
`evidence.json` 里 `serving_end_to_end.tested` 仍是 `false`、`speedup` 仍是
`null`——读者只看索引就会以为服务端已经验证过。这类"标签比证据走得远"的
写入，人为复核很容易漏，所以落成机械判据。

判据只做**层级与事实的一致性**，不判断数字好坏：

1. 索引词汇表：`layer`、`default_enabled`、`gate_env` 只允许文档化取值；
2. 服务结论必须有测量：`verified_faster` / `within_noise` /
   `verified_on_par_with_baseline` 要求 `serving_end_to_end.tested` 为真且
   `speedup` 为数字，`measured_no_gain` 至少要求测过；
3. 未测标签不得有测量：`pending_outer_loop` / `negative_or_pending` 不得
   出现 `tested: true`；
4. 归因先行：`l4_attribution.status` 仍是 `pending_outer_loop` 时，服务标签
   只能是"无结论"类；
5. L3 一致：索引的 `l3_numerical_guard` 与证据的 `l3_numerical_guard.passed`
   必须同向；L3 失败时禁止任何服务收益标签；
6. 方向一致：`verified_faster` 的 `speedup` 必须大于 1，`measured_no_gain`
   不得声称大于 1；
7. `superseded_by_host_fork` 必须带宿主指针（`host_operator` 与
   `host_branch`）；
8. 每份 `evidence.json` 要么声明 `operator-benchmark-evidence/v1`，要么登记
   在 `LEGACY_EVIDENCE` 里（旧 schema 的存量债，迁移后删除该条）。

前七条只对声明了 v1 schema 的证据生效；`LEGACY_EVIDENCE` 里的旧文件跳过字段
级判据，但仍受索引词汇表约束。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "operator-benchmark-evidence/v1"

#: 需要被测过的服务结论标签；`measured_no_gain` 只要求测过、不要求有数字。
MEASURED_CLAIMS = {
    "measured_no_gain",
    "verified_faster",
    "within_noise",
    "verified_on_par_with_baseline",
}
NUMERIC_CLAIMS = {
    "verified_faster",
    "within_noise",
    "verified_on_par_with_baseline",
}

#: 明确表示"还没有服务结论"的标签。
NO_CLAIM = {"pending_outer_loop", "negative_or_pending"}

#: L4 未归因时允许出现（且只允许出现）的无结论标签。
ATTR_PENDING_ALLOWED = {
    "pending_outer_loop",
    "negative_or_pending",
    "superseded_by_host_fork",
    "l3_failed_default_off",
    "historical_only",
}

#: 文档化的归属层（见 GLOSSARY.md §3.4 的 `layer` 词条）。
ALLOWED_LAYERS = {"host_fork", "plugin_carrier", "rejected_exploration"}

#: 文档化的 L3 标签；空串表示该算子没有 L3 结论。
ALLOWED_L3 = {"passed", "failed"}
ALLOWED_SERVING = MEASURED_CLAIMS | NO_CLAIM | ATTR_PENDING_ALLOWED

#: `gate_env` 是"变量名=启用值"的赋值写法（如 `VLLM_HUST_GATE_FP32=1`），
#: 未对外暴露时写 `unexposed`。
GATE_ENV_RE = re.compile(r"^VLLM_HUST_[A-Z0-9_]+=\S+$")

#: 旧 schema 证据的存量债：文件 -> 现状说明。迁移到 v1 后删除对应条目。
LEGACY_EVIDENCE = {
    "ops/gdn_input_norm/evidence.json": (
        "旧字段名（operator / l3_verification / l0_benchmark），未迁移到 v1；"
        "缺少 serving_end_to_end 与 l4_attribution"
    ),
}


def _index() -> dict:
    return json.loads((ROOT / "DELIVERY_INDEX.json").read_text(encoding="utf-8"))


def _evidence(entry: dict) -> dict:
    path = ROOT / entry["evidence"]
    return json.loads(path.read_text(encoding="utf-8"))


def _v1_entries() -> list[tuple[dict, dict]]:
    """返回 (index 条目, v1 证据) 的列表；旧 schema 条目被跳过。"""
    pairs = []
    for entry in _index()["operators"]:
        evidence = _evidence(entry)
        if evidence.get("$schema") == SCHEMA:
            pairs.append((entry, evidence))
    return pairs


def test_index_uses_documented_vocabulary():
    """`layer` / `default_enabled` / `gate_env` 只允许文档化取值。

    工具新登记时留下的占位值 `unclassified` 也会在这里报红：索引是对外结论面，
    归属层必须在合入前由人工判定（见 GLOSSARY.md 的 `layer` 词条）。
    """
    violations = []
    for entry in _index()["operators"]:
        op_id = entry["id"]
        if entry["layer"] not in ALLOWED_LAYERS:
            violations.append(
                f"{op_id}: layer={entry['layer']!r}（占位值必须改成真实归属层）"
            )
        if entry["default_enabled"] is not False:
            violations.append(f"{op_id}: default_enabled 必须为 false（default-off）")
        if entry["serving_status"] not in ALLOWED_SERVING:
            violations.append(
                f"{op_id}: unknown serving_status={entry['serving_status']!r}"
            )
        gate_env = entry["gate_env"]
        if gate_env != "unexposed" and not GATE_ENV_RE.match(gate_env):
            violations.append(
                f"{op_id}: gate_env={gate_env!r} 不是 VLLM_HUST_<NAME>=<value> 形式"
            )
        if entry["l3_numerical_guard"] not in ALLOWED_L3:
            violations.append(
                f"{op_id}: l3_numerical_guard={entry['l3_numerical_guard']!r}"
            )
    assert not violations, "索引词汇表越界: " + "; ".join(violations)


def test_serving_claims_are_backed_by_measurements():
    """声称有服务结论的算子，证据里必须真的测过，且结论标签要有数字。"""
    violations = []
    for entry, evidence in _v1_entries():
        op_id = entry["id"]
        status = entry["serving_status"]
        serving = evidence.get("serving_end_to_end") or {}
        tested = serving.get("tested")
        speedup = serving.get("speedup")

        if status in MEASURED_CLAIMS and tested is not True:
            violations.append(
                f"{op_id}: serving_status={status} 但 evidence 里 tested={tested}"
            )
        if status in NUMERIC_CLAIMS and (
            type(speedup) not in (int, float)
            or not math.isfinite(speedup)
            or speedup <= 0
        ):
            violations.append(
                f"{op_id}: serving_status={status} 但 evidence 里 speedup={speedup}"
            )
        if status in NO_CLAIM and tested is True:
            violations.append(
                f"{op_id}: serving_status={status} 却写了 tested=true，标签落后于证据"
            )
    assert not violations, "服务结论缺少测量支撑: " + "; ".join(violations)


def assert_serving_prerequisites(entry: dict, evidence: dict) -> None:
    if entry["serving_status"] not in MEASURED_CLAIMS:
        return
    assert (evidence.get("l3_numerical_guard") or {}).get("passed") is True
    attribution = evidence.get("l4_attribution") or {}
    assert attribution.get("status") in (None, "completed")
    assert attribution.get("verified", True) is True
    for key in ("call_count_expected", "call_count_actual"):
        assert type(attribution.get(key)) is int and attribution[key] > 0
    if attribution["call_count_expected"] != attribution["call_count_actual"]:
        assert attribution.get("verified") is True and attribution.get("note")
    serving = evidence.get("serving_end_to_end") or {}
    assert serving.get("tested") is True
    assert serving.get("applicability", "current") == "current"


def test_serving_conclusion_requires_l4_attribution():
    for entry, evidence in _v1_entries():
        assert_serving_prerequisites(entry, evidence)
        if (evidence.get("l4_attribution") or {}).get("status") == "pending_outer_loop":
            assert entry["serving_status"] in ATTR_PENDING_ALLOWED
        if entry["serving_status"] == "historical_only":
            serving = evidence.get("serving_end_to_end") or {}
            assert serving.get("tested") is True
            assert serving.get("applicability") == "historical"


@pytest.mark.parametrize(
    "change",
    [
        "missing_l3",
        "unknown_l3",
        "failed_l3",
        "missing_l4",
        "unknown_l4",
        "pending_l4",
        "zero_calls",
        "mismatched_calls",
        "unverified",
        "historical",
        "not_tested",
    ],
)
def test_serving_prerequisites_reject_incomplete_evidence(change):
    entry = {"serving_status": "verified_faster"}
    evidence = {
        "l3_numerical_guard": {"passed": True},
        "l4_attribution": {"call_count_expected": 10, "call_count_actual": 10},
        "serving_end_to_end": {"tested": True},
    }
    if change == "missing_l3":
        del evidence["l3_numerical_guard"]
    elif change in {"unknown_l3", "failed_l3"}:
        evidence["l3_numerical_guard"]["passed"] = (
            None if change == "unknown_l3" else False
        )
    elif change == "missing_l4":
        del evidence["l4_attribution"]
    elif change in {"unknown_l4", "pending_l4"}:
        evidence["l4_attribution"]["status"] = (
            "unknown" if change == "unknown_l4" else "pending_outer_loop"
        )
    elif change in {"zero_calls", "mismatched_calls"}:
        evidence["l4_attribution"]["call_count_actual"] = (
            0 if change == "zero_calls" else 9
        )
    elif change == "unverified":
        evidence["l4_attribution"]["verified"] = False
    elif change == "historical":
        evidence["serving_end_to_end"]["applicability"] = "historical"
    else:
        evidence["serving_end_to_end"]["tested"] = False
    with pytest.raises(AssertionError):
        assert_serving_prerequisites(entry, evidence)


def test_measured_no_gain_requires_measurement(monkeypatch):
    monkeypatch.setattr(
        __import__(__name__),
        "_v1_entries",
        lambda: [
            (
                {"id": "example", "serving_status": "measured_no_gain"},
                {"serving_end_to_end": {"tested": False}},
            )
        ],
    )
    with pytest.raises(AssertionError):
        test_serving_claims_are_backed_by_measurements()


def test_l3_label_matches_evidence_and_vetoes_serving_claims():
    """索引的 L3 标签必须与证据同向；L3 失败时禁止服务收益结论。"""
    violations = []
    for entry, evidence in _v1_entries():
        op_id = entry["id"]
        guard = evidence.get("l3_numerical_guard") or {}
        passed = guard.get("passed")
        if passed is not None:
            expected = "passed" if passed else "failed"
            if entry["l3_numerical_guard"] != expected:
                violations.append(
                    f"{op_id}: 索引 l3={entry['l3_numerical_guard']}，"
                    f"evidence passed={passed}"
                )
        if entry["l3_numerical_guard"] == "failed" and (
            entry["serving_status"] in MEASURED_CLAIMS
        ):
            violations.append(
                f"{op_id}: L3 失败却声明 serving_status={entry['serving_status']}"
            )
    assert not violations, "L3 标签与证据不一致: " + "; ".join(violations)


def test_serving_label_direction_matches_speedup():
    """`verified_faster` 必须真比基线快；`measured_no_gain` 不得写成更快。"""
    violations = []
    for entry, evidence in _v1_entries():
        op_id = entry["id"]
        status = entry["serving_status"]
        speedup = (evidence.get("serving_end_to_end") or {}).get("speedup")
        if not isinstance(speedup, (int, float)):
            continue
        if status == "verified_faster" and speedup <= 1.0:
            violations.append(f"{op_id}: verified_faster 但 speedup={speedup}")
        if status == "measured_no_gain" and speedup > 1.0:
            violations.append(f"{op_id}: measured_no_gain 但 speedup={speedup}")
    assert not violations, "结论方向与数字相反: " + "; ".join(violations)


def test_superseded_labels_carry_the_host_pointer():
    """`superseded_by_host_fork` 必须写明被哪个宿主算子取代。"""
    violations = []
    for entry in _index()["operators"]:
        if entry["serving_status"] != "superseded_by_host_fork":
            continue
        pointer = entry.get("superseded_by_host_fork") or {}
        for key in ("host_operator", "host_branch"):
            if not pointer.get(key):
                violations.append(f"{entry['id']}: 缺少 superseded_by_host_fork.{key}")
    assert not violations, "取代关系缺少宿主指针: " + "; ".join(violations)


@pytest.mark.parametrize("entry_index", range(len(_index()["operators"])))
def test_every_evidence_declares_schema_or_is_known_legacy(entry_index: int):
    """每份证据要么声明 v1 schema，要么明确登记为旧 schema 存量债。"""
    entry = _index()["operators"][entry_index]
    evidence = _evidence(entry)
    schema = evidence.get("$schema")
    if schema == SCHEMA:
        return
    relative = Path(entry["evidence"]).as_posix()
    assert relative in LEGACY_EVIDENCE, (
        f"{relative} 未声明 {SCHEMA}，也不在 LEGACY_EVIDENCE 清单里；"
        "请按 docs/operator-benchmark-standard.md §4 的结构补齐"
    )
    assert LEGACY_EVIDENCE[relative], f"{relative} 的存量债说明不能为空"
