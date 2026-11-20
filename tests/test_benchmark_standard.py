"""测试算子优化评测标准、报告模板及证据 JSON Schema 规范。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_operator_benchmark_standard_document_structure():
    doc = (ROOT / "docs/operator-benchmark-standard.md").read_text(encoding="utf-8")
    assert "形状事实与单变量上下文合同" in doc
    assert "算子优化四级验证阶梯" in doc
    assert "L0: 单算子微基准" in doc
    assert "L3: 数值正确性护栏" in doc
    assert "L4: 图内调用与链路归因" in doc
    assert "服务收益: 端到端服务表现" in doc
    assert "operator-benchmark-evidence/v1" in doc


def test_standard_evidence_schema_example():
    """验证文档中的标准证据 JSON 示例合法且包含关键形状事实字段。"""
    doc = (ROOT / "docs/operator-benchmark-standard.md").read_text(encoding="utf-8")
    json_start = doc.find('```json\n{\n  "$schema": "operator-benchmark-evidence/v1"')
    assert json_start != -1, "文档中必须包含标准证据 JSON 示例"
    json_body = doc[json_start + 7 :]
    json_end = json_body.find("\n```")
    evidence_json = json_body[:json_end]
    data = json.loads(evidence_json)

    assert data["$schema"] == "operator-benchmark-evidence/v1"
    assert "operator_name" in data
    assert "callsite" in data

    env = data["environment"]
    for key in ("device", "cann_version", "vllm_version", "model", "dtype"):
        assert key in env, f"缺少环境事实: {key}"

    shapes = data["shape_facts"]
    for key in ("phase", "batch_size", "prompt_tokens", "decode_tokens"):
        assert key in shapes, f"缺少形状事实: {key}"

    l0 = data["l0_microbenchmark"]
    assert "native_median_us" in l0
    assert "optimized_median_us" in l0
    assert "speedup" in l0

    l3 = data["l3_numerical_guard"]
    assert l3["passed"] is None

    l4 = data["l4_attribution"]
    assert "call_count_expected" in l4
    assert "call_count_actual" in l4
    assert l4["verified"] is False

    serving = data["serving_end_to_end"]
    assert "concurrency" in serving
    assert serving["tested"] is False
    assert serving["speedup"] is None
    assert all(value is None for value in l0.values())


def test_mod_contract_keeps_protocol_and_measurements_separate():
    data = json.loads((ROOT / "docs/mod-acceptance/evidence.json").read_text())
    assert data["schema"] == "mod-acceptance-evidence/v1"
    assert data["sources"]["standard_id"] == "five-datasets-20261113-v1"
    contract = data["contract"]
    datasets = {row["id"]: row for row in contract["datasets"]}
    assert set(datasets) == {"mmlu", "hle", "frontier", "swe", "terminal"}
    assert sum(datasets["hle"]["strata"].values()) == datasets["hle"]["tasks"]
    assert sum(datasets["frontier"]["tracks"].values()) == datasets["frontier"]["tasks"]
    for row in datasets.values():
        assert row["protocol"] in contract["protocol_artifacts"]
        if row["mode"] == "quality_and_inference":
            assert row["requests_per_arm"] == row["tasks"]
        else:
            assert row["id"] in {"swe", "terminal"}
    assert contract["generation"]["preserve_eos"] is True
    assert contract["trace"]["execute_tools"] is False
    assert contract["trace"]["kv_reuse"] is False
    assert contract["trace"]["tokenize_hash_match_required"] is True
    targets = contract["targets"]
    assert set(targets) == {"effective_compute", "serving", "stability", "cost"}
    compute = targets["effective_compute"]
    assert compute["current_eta_ref_min_percent"] < compute["final_eta_ref_min_percent"]
    assessment = data["assessment"]
    assert set(assessment["targets"]) == set(targets)
    assert set(assessment["dataset_execution"]) == set(datasets)
    if assessment["execution_complete"]:
        assert all(row["tested"] for row in assessment["dataset_execution"].values())
    for row in assessment["dataset_execution"].values():
        if row["tested"]:
            assert row["evidence_refs"]
    for row in assessment["targets"].values():
        assert row["passed"] is None or type(row["passed"]) is bool
        if row["passed"] is not None:
            assert row["evidence_refs"]
        else:
            assert row["missing"]
    if assessment["acceptance_passed"] is True:
        assert assessment["execution_complete"]
        assert assessment["campaign_identity"]
        assert all(row["passed"] is True for row in assessment["targets"].values())


def test_mod_audit_revision_and_metadata_claims():
    import re

    data = json.loads((ROOT / "docs/mod-acceptance/evidence.json").read_text())
    assessment = data["assessment"]
    assert re.fullmatch(r"[a-f0-9]{40}", assessment["reviewed_commit"])
    metadata = json.loads((ROOT / "MOD_METADATA.json").read_text())["evidence"]
    if assessment["acceptance_passed"] is not True:
        assert metadata["qualification"] == "unqualified"
        assert metadata["performance_claims"] == []


def assert_mod_measurement_identity(assessment):
    import hashlib
    import re

    rows = [*assessment["dataset_execution"].values(), *assessment["targets"].values()]
    measured = any(
        row.get("tested") is True or row.get("passed") is not None for row in rows
    )
    if not measured:
        return
    identity = assessment["campaign_identity"]
    assert isinstance(identity, dict)
    for key in (
        "implementation",
        "host",
        "model",
        "hardware",
        "activation",
        "protocol",
    ):
        assert identity.get(key), key
    for row in rows:
        if row.get("tested") is not True and row.get("passed") is None:
            continue
        assert row["evidence_refs"]
        for ref in row["evidence_refs"]:
            assert isinstance(ref, dict)
            assert re.fullmatch(r"[a-f0-9]{64}", ref["sha256"])
            name = Path(ref["path"])
            assert not name.is_absolute() and ".." not in name.parts
            assert (
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == ref["sha256"]
            )


def test_mod_measurements_require_resolvable_evidence():
    data = json.loads((ROOT / "docs/mod-acceptance/evidence.json").read_text())
    assert_mod_measurement_identity(data["assessment"])


def test_mod_measurements_reject_arbitrary_campaign_and_missing_reference():
    import pytest

    assessment = {
        "campaign_identity": "some campaign",
        "dataset_execution": {"mmlu": {"tested": True, "evidence_refs": ["missing"]}},
        "targets": {},
    }
    with pytest.raises(AssertionError):
        assert_mod_measurement_identity(assessment)
    assessment["campaign_identity"] = dict.fromkeys(
        ("implementation", "host", "model", "hardware", "activation", "protocol"),
        "fixed",
    )
    with pytest.raises(AssertionError):
        assert_mod_measurement_identity(assessment)
    assessment["dataset_execution"]["mmlu"]["evidence_refs"] = [
        {"path": "missing-mod-evidence.json", "sha256": "0" * 64}
    ]
    with pytest.raises(FileNotFoundError):
        assert_mod_measurement_identity(assessment)


def test_mod_skill_structure_and_local_contract_references():
    import re

    folder = ROOT / ".openbitfun/skills/mod-acceptance"
    text = (folder / "SKILL.md").read_text()
    _, frontmatter, body = text.split("---", 2)
    fields = dict(line.split(": ", 1) for line in frontmatter.strip().splitlines())
    assert set(fields) == {"name", "description"}
    assert fields["name"] == folder.name
    assert fields["description"].startswith("Use when")
    assert len(frontmatter) < 1024
    for name in re.findall(r"`([^`]+\.(?:md|json))`", body):
        path = folder / name if name.startswith("evals/") else ROOT / name
        assert path.is_file(), name
    cases = json.loads((folder / "evals/cases.json").read_text())
    assert len({row["id"] for row in cases["cases"]}) == len(cases["cases"])
    assert all(row["prompt"] and row["checks"] for row in cases["cases"])


def test_combination_policy_keeps_single_operator_ab_optional():
    for name in (
        "docs/operator-benchmark-standard.md",
        ".openbitfun/skills/operator-sol-optimizer/SKILL.md",
        ".openbitfun/skills/mod-acceptance/SKILL.md",
    ):
        text = (ROOT / name).read_text()
        assert "单算子端到端 A/B 按需选测" in text
    data = json.loads((ROOT / "docs/mod-acceptance/evidence.json").read_text())
    plan = data["assessment"]["combination_plan"]
    assert plan["tested"] is False
    assert plan["single_operator_ab"] == "optional"
    assert plan["required_guards"] == ["numerical", "per_operator_attribution"]
    index = json.loads((ROOT / "DELIVERY_INDEX.json").read_text())
    entries = {e["id"]: e for e in index["operators"]}
    assert set(plan["candidates"]).isdisjoint(plan["excluded"])
    assert set(plan["candidates"]) | set(plan["excluded"]) == set(entries)
    for name, gate in plan["candidates"].items():
        assert entries[name]["gate_env"] == gate + "=1"
        assert entries[name]["l3_numerical_guard"] != "failed"
        assert plan["arms"]["off"][gate] == "0"
        assert plan["arms"]["on"][gate] == "1"
    assert set(plan["arms"]["off"]) == set(plan["arms"]["on"])
    assert all(plan["excluded"].values())
