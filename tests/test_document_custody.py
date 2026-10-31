import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_historical_reports_are_references_not_metric_copies():
    source = json.loads((ROOT / "EVIDENCE_AVAILABILITY.json").read_text())
    assert re.fullmatch(r"[a-f0-9]{40}", source["commit"])
    prefix = (
        "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/"
        + source["commit"]
        + "/"
        + source["snapshot"]
        + "/"
    )
    archive = json.loads((ROOT / "tools/workbench-archive.json").read_text())
    archived = {row["path"] for row in archive["files"]}
    for name in source["report_reference_files"]:
        if name in archived:
            assert not (ROOT / name).exists()
            continue
        data = json.loads((ROOT / name).read_text())
        assert data["schema"] == "benchmark-document-reference/v1"
        assert data["historical_document"] == prefix + name
        assert "not a recovered raw probe" in data["document_kind"]
        assert (
            not {"local_results", "full_model_results", "resolution", "tests"}
            & data.keys()
        )


def test_unrecovered_probes_are_explicit_and_not_successful_urls():
    source = json.loads((ROOT / "EVIDENCE_AVAILABILITY.json").read_text())
    assert source["local_reference_occurrences"] == 39
    assert source["unrecovered_occurrences"] == 33
    assert source["sha256_recovered_occurrences"] == 6
    closure = json.loads(
        (ROOT / "host-proposals/GDN-qkv-norm/registration-closure.json").read_text()
    )
    assert closure["evidence"] is None
    assert closure["bundle_registration"] == "withdrawn"
    assert "null is not a passing result" in closure["evidence_availability"]["status"]
    archive = json.loads((ROOT / "tools/workbench-archive.json").read_text())
    archived = {row["path"] for row in archive["files"]}
    for name in source["reviewed_files"]:
        if name.endswith(".json") and name not in archived:
            text = (ROOT / name).read_text()
            assert "/root/包体恢复/records/probes/" not in text
            assert "/root/包体恢复/records/ladder/" not in text


def test_installation_consumers_and_boundary_are_retained():
    release = (ROOT / "docs/release.md").read_text()
    for version in ("0.1.0", "0.1.1", "0.1.2", "0.1.3", "0.2.0"):
        assert f"https://pypi.org/project/vllm-ascend-split-batch/{version}/" in release
        assert f"releases/tag/v{version}" in release
    readme = (ROOT / "README.md").read_text()
    assert "within noise" in readme
    assert "fail-closed" in readme
    assert "must not be crossed" in readme


def test_delivery_inventory_contract():
    inventory = (ROOT / "docs/delivery-inventory.md").read_text()
    assert "vllm-ascend-hust" in inventory
    assert "work/issue20-w3f2-host" in inventory
    assert "W3F2" in inventory
    assert "D4" in inventory
    assert "两层架构分工" in inventory


def test_delivery_inventory_lists_each_published_general_plugin():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    plugins = pyproject["project"]["entry-points"]["vllm.general_plugins"]
    inventory = (ROOT / "docs/delivery-inventory.md").read_text()
    for name in plugins:
        assert f"`{name}`" in inventory


def test_recovered_consumers_link_originals_without_requalifying():
    source = json.loads((ROOT / "EVIDENCE_AVAILABILITY.json").read_text())
    recovery = source["recovery_audit"]
    assert re.fullmatch(r"[a-f0-9]{40}", recovery["commit"])
    assert re.fullmatch(r"[a-f0-9]{64}", recovery["sha256"])
    gdn = json.loads(
        (ROOT / "host-proposals/GDN-qkv-norm/proposal_meta.json").read_text()
    )
    for url, name in zip(
        gdn["evidence"][:2],
        ["l0-fused-norm.json", "kernel-trace-summary.json"],
        strict=True,
    ):
        assert url.startswith("https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/")
        assert url.endswith("/" + name)
    assert "withdrawn bundle registration" in gdn["decision"]
    archive = json.loads((ROOT / "tools/workbench-archive.json").read_text())
    assert archive["source_commit"] == "185f48f3ac845833beb41bf2b8d34f7f82388128"
    assert (
        archive["archive_sha256"]
        == "003a5a296604c15585d4041395c41d8f8d6c935adb11221f27b3c6cfa2a269f7"
    )
    for path in (
        "host-proposals/OPT-delivery-workflow/proposal_meta.json",
        "tools/operator_delivery.py",
        "tests/test_operator_delivery.py",
    ):
        assert any(
            row["path"] == path and re.fullmatch(r"[a-f0-9]{64}", row["sha256"])
            for row in archive["files"]
        )
    pitfalls = (ROOT / "docs/pitfalls.md").read_text()
    assert "multibuffer-graph.json" in pitfalls
    assert "两份 separate-graph 崩溃日志仍未找回" in pitfalls
