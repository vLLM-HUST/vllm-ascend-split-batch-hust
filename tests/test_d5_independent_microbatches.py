# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import importlib.util
from pathlib import Path

import pytest
from _deployment import deployed_file

PATH = (
    Path(__file__).resolve().parents[1] / "tools/probe_d5_independent_microbatches.py"
)
SPEC = importlib.util.spec_from_file_location("probe_d5_independent", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_source_method_executes_full_body_with_annotations_deferred():
    source = (
        "class Host:\n    def method(self, x: DeviceTensor):\n        return x + 3\n"
    )
    method, witness = probe.load_method(source, "method", {})
    assert method(None, 4) == 7
    assert witness["line"] == 2
    assert witness["end_line"] == 3
    assert "return x + 3" in witness["source"]


@pytest.mark.parametrize("source", ["pass", "def f(): pass\nclass C:\n def f(): pass"])
def test_missing_or_ambiguous_method_is_not_silently_selected(source):
    with pytest.raises(ValueError, match="expected exactly one"):
        probe.load_method(source, "f", {})


def test_assertion_message_must_match():
    def reject():
        raise AssertionError("GDN not supported")

    assert (
        probe.expect_assert(reject, "GDN not supported")["exception"]
        == "AssertionError"
    )
    with pytest.raises(ValueError, match="unexpected assertion"):
        probe.expect_assert(reject, "different rejection")
    with pytest.raises(ValueError, match="no longer rejects"):
        probe.expect_assert(lambda: None, "GDN not supported")


def test_non_assertion_failure_is_not_a_successful_probe():
    def unexpected():
        raise AttributeError("missing fixture attribute")

    with pytest.raises(AttributeError):
        probe.expect_assert(unexpected, "GDN not supported")


def test_existing_evidence_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "result.json"
    output.write_text("preserved")
    monkeypatch.setattr(
        "sys.argv",
        [str(PATH), "--core", "/missing", "--ascend", "/missing", "--out", str(output)],
    )
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
    assert output.read_text() == "preserved"


def test_full_host_probe_when_checkout_is_available():
    root = Path("/root/vllm-workspace")
    runner = root / "vllm-ascend-hust/vllm_ascend/worker/model_runner_v1.py"
    if not deployed_file(runner):
        pytest.skip("read-only Ascend checkout not present")
    pytest.importorskip("numpy", reason="the host probe reads host sources via numpy")
    result = probe.probe(root / "vllm-hust", root / "vllm-ascend-hust")
    assert result["verdict"] == "BLOCKED_CURRENT_HOST_PATH"
    assert len(result["v1_forced_gdn_metadata"]) == 2
    assert all(not row["should_ubatch"] for row in result["v1_selection"])
    assert result["workspace"]["two_lane_manager_can_select_lane_1"]
    cases = result["splitter_interface_cases"]
    assert cases[0]["shared_request_indices"] == []
    assert cases[1]["shared_request_indices"] == [0]
