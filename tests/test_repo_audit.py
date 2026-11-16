"""索引、证据与文档引用的一致性审计（不依赖仓外归档）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import repo_audit  # noqa: E402


def test_index_evidence_and_document_references_are_consistent():
    problems: list[str] = []
    entries = repo_audit.check_index(problems)
    repo_audit.check_docs(entries, problems)
    assert not problems, "仓库交付面不一致: " + "; ".join(problems)


def test_operator_names_match_index_ids():
    entries = repo_audit.check_index([])
    names = {}
    for op_id, entry in entries.items():
        evidence = repo_audit.load(ROOT / entry["evidence"])
        if evidence.get("$schema") == "operator-benchmark-evidence/v1":
            names[op_id] = evidence.get("operator_name")
    assert names, "没有可校验的 v1 证据"
    assert all(name == op_id for op_id, name in names.items()), names


def test_non_operator_entrypoints_stay_registered_without_operator_dirs():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    plugins = project["entry-points"]["vllm.general_plugins"]
    bundles = project["entry-points"]["vllm_hust.extension_bundles"]
    non_operators = {
        "rope-fix": ("rope-fix", "org.vllm-hust.rope-fix"),
        "zerocost-wiring": ("zerocost-wiring", "org.vllm-hust.zerocost-wiring"),
        "aot-cache-guard": ("aot-cache-guard", None),
        "mlp-chunk": ("mlp-chunk", None),
    }
    index = json.loads((ROOT / "DELIVERY_INDEX.json").read_text())
    indexed = {entry["id"] for entry in index["operators"]}
    for op_id, (plugin, bundle) in non_operators.items():
        assert plugin in plugins
        if bundle:
            assert bundle in bundles
        assert not (ROOT / "ops" / op_id).exists()
        assert op_id not in indexed
