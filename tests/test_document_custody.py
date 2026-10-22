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
    for name in source["report_reference_files"]:
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
    for name in source["reviewed_files"]:
        if name.endswith(".json"):
            text = (ROOT / name).read_text()
            assert "/root/包体恢复/records/probes/" not in text
            assert "/root/包体恢复/records/ladder/" not in text


def test_installation_consumers_and_boundary_are_retained():
    release = (ROOT / "docs/release.md").read_text()
    for version in ("0.1.0", "0.1.1", "0.1.2", "0.1.3"):
        assert f"https://pypi.org/project/vllm-ascend-split-batch/{version}/" in release
        assert f"releases/tag/v{version}" in release
    readme = (ROOT / "README.md").read_text()
    assert "within noise" in readme
    assert "fail-closed" in readme
    assert "must not be crossed" in readme


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
    delivery = json.loads(
        (ROOT / "host-proposals/OPT-delivery-workflow/proposal_meta.json").read_text()
    )
    assert delivery["motivation_evidence"].endswith(
        "/model-validation/20261018T145800Z_gdn_model_receipt.json"
    )
    pitfalls = (ROOT / "docs/pitfalls.md").read_text()
    assert "multibuffer-graph.json" in pitfalls
    assert "两份 separate-graph 崩溃日志仍未找回" in pitfalls
