"""索引、证据与文档引用的一致性审计（不依赖仓外归档）。"""

from __future__ import annotations

import json
import re
import subprocess
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


def test_tracked_markdown_local_links_resolve():
    files = (
        subprocess.check_output(["git", "ls-files", "-z", "*.md"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    missing = []
    for name in filter(None, files):
        doc = ROOT / name
        for target in repo_audit.LINK.findall(doc.read_text()):
            path = target.partition("#")[0]
            if (
                path
                and not re.match(r"[a-z]+:|<", path)
                and not (doc.parent / path).exists()
            ):
                missing.append(f"{name}: {target}")
    assert not missing, missing


def test_only_schedulable_operators_declare_a_service_route():
    index = repo_audit.load(repo_audit.INDEX)["operators"]
    entries = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "entry-points"
    ]["vllm.general_plugins"]
    from tools.operator_tools.serving_attribution import GATES

    indexed = {row["id"]: row["gate_env"] for row in index}
    assert all(indexed[op] == gate + "=1" for op, gate in GATES.items())
    gates = set()
    for row in index:
        route = row.get("service_route")
        assert bool(route) == (row["gate_env"] != "unexposed"), row["id"]
        if not route:
            continue
        assert row["gate_env"] not in gates and row["layer"] != "rejected_exploration"
        gates.add(row["gate_env"])
        assert (
            route["scope"]
            and route["expression"]
            and route["consumer"].startswith("vllm_ascend/")
        )
        if row["layer"] == "host_fork":
            assert (
                route["reader"].startswith("vllm_ascend/")
                and "entry_point" not in route
            )
        else:
            module = route["reader"].removesuffix(".py").replace("/", ".")
            assert entries[route["entry_point"]] == f"{module}:load"


def test_combo_cannot_benchmark_an_unattributed_gate():
    import pytest

    from tools.operator_tools.serving_collect import service_routes

    config = {
        "gates": {
            "off": {"VLLM_HUST_GDN_SPLIT_BA": "0"},
            "on": {"VLLM_HUST_GDN_SPLIT_BA": "1"},
        }
    }
    with pytest.raises(ValueError, match="every enabled operator"):
        service_routes(config)
    config["attributions"] = {"gdn_split_ba": {"mode": "dedicated"}}
    assert [r["operator"] for r in service_routes(config)] == ["gdn_split_ba"]
