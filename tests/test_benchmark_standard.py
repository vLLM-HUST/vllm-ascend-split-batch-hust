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
    assert "服务收益: 端到端吞吐与时延" in doc
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
    assert "passed" in l3

    l4 = data["l4_attribution"]
    assert "call_count_expected" in l4
    assert "call_count_actual" in l4

    serving = data["serving_end_to_end"]
    assert "concurrency" in serving
    assert "throughput_native_tokens_s" in serving
    assert "throughput_opt_tokens_s" in serving
