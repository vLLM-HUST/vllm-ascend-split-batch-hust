"""Tests for the pending-operator long-run plan builder."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import (  # noqa: E402
    pending_longrun,
    serving_e2e_auto,
)


def _template() -> dict:
    config = {
        "python": "/runtime/python",
        "server_command": ["/runtime/python", "-m", "vllm.entrypoints.cli.main"],
        "host_commits": {
            "vllm": "a" * 40,
            "vllm-ascend": "b" * 40,
        },
        "port": 8350,
        "model_alias": "qwen35-35b",
        "model_path": "/models/Qwen3.5-35B-A3B",
        "sharegpt_path": "/data/ShareGPT.json",
        "world_size": 2,
    }
    return {"timeout_seconds": 14400, "operators": [{"id": "seed", "config": config}]}


def test_default_plan_covers_pending_operators_without_promoting_evidence():
    plan, report = pending_longrun.build_plan(_template())
    ids = [entry["id"] for entry in plan["operators"]]
    assert ids == list(pending_longrun.DEFAULT_PENDING_OPS)
    assert "qwen_text_split_qkv_fuse" not in ids
    assert "moe_epilogue" in ids
    assert "gdn_input_norm" not in ids
    assert set(ids).isdisjoint(
        {
            "gdn_conv_pack",
            "gate_fp32",
            "gdn_split_fusion",
            "gdn_merge_inproj",
            "gdn_output_norm_fusion",
        }
    )
    assert all(
        entry.get("config", {}).get("sync_mode") == "none"
        for entry in plan["operators"]
        if "config" in entry
    )
    by_id = {entry["id"]: entry for entry in plan["operators"]}
    assert "reason" in by_id["gdn_qkv_norm"]
    assert by_id["moe_epilogue"]["config"]["gate"] == "VLLM_HUST_MOE_EPILOGUE"
    assert by_id["moe_epilogue"]["config"]["kernel"] == "_gate_mul_kernel"
    assert by_id["gdn_split_ba"]["config"]["kernel"] == "gdn_split_ba_2d_kernel"
    assert all(
        set(by_id[op]) == {"id", "reason"}
        for op in ("moe_down_residual_norm", "ffn_activation", "gdn_qkv_norm")
    )
    assert {row["mode"] for row in report} == {"skip", "full"}


def test_mode_override_can_disable_a_full_entry():
    plan, _ = pending_longrun.build_plan(_template(), modes={"gdn_split_ba": "skip"})
    entry = next(row for row in plan["operators"] if row["id"] == "gdn_split_ba")
    assert list(entry) == ["id", "reason"]


@pytest.mark.parametrize(
    "op", ["ffn_activation", "moe_down_residual_norm", "gdn_qkv_norm"]
)
@pytest.mark.parametrize("mode", ["full", "guard_only"])
def test_override_cannot_schedule_an_unexposed_gate(op, mode):
    with pytest.raises(ValueError, match="no verified gate"):
        pending_longrun.build_plan(_template(), modes={op: mode})


def test_plan_rejects_gate_belonging_to_another_operator():
    plan, _ = pending_longrun.build_plan(_template())
    plan["operators"][0]["config"]["gate"] = "VLLM_HUST_GDN_SPLIT_BA"
    with pytest.raises(ValueError, match="another operator"):
        serving_e2e_auto.validate_plan(plan)


@pytest.mark.parametrize(
    "top,nested",
    [(None, None), (None, "none"), ("none", None), ("auto", "none"), ("none", "auto")],
)
@pytest.mark.parametrize("combination", [True, False])
def test_sync_veto_never_calls_either_writer(
    tmp_path, monkeypatch, top, nested, combination
):
    plan, _ = pending_longrun.build_plan(_template())
    entry = plan["operators"][0]
    if top is not None:
        entry["sync_mode"] = top
    if nested is None:
        entry["config"].pop("sync_mode")
    else:
        entry["config"]["sync_mode"] = nested
    if combination:
        entry["combination_evidence_target"] = "must-not-write.json"

    def unexpected(*args, **kwargs):
        pytest.fail("archive-only plan attempted an evidence write")

    monkeypatch.setattr(serving_e2e_auto.serving_sync, "sync", unexpected)
    monkeypatch.setattr(serving_e2e_auto, "write_combination_evidence", unexpected)
    assert (
        serving_e2e_auto.sync_item_evidence(entry, tmp_path, logging.getLogger())
        == "not_applicable"
    )


def test_invalid_sync_mode_stops_before_collection():
    plan, _ = pending_longrun.build_plan(_template())
    plan["operators"][0]["config"]["sync_mode"] = "non"
    with pytest.raises(ValueError, match="sync_mode"):
        serving_e2e_auto.validate_plan(plan)


def test_requires_explicit_base_plan(monkeypatch):
    monkeypatch.delenv("SERVING_BASE_PLAN", raising=False)
    with pytest.raises(ValueError, match="SERVING_BASE_PLAN"):
        pending_longrun.find_template(None)


@pytest.mark.parametrize("top", [False, True])
def test_manual_promotion_requires_explicit_auto(top):
    entry = {"id": "op", "config": {}}
    (entry if top else entry["config"])["sync_mode"] = "auto"
    assert serving_e2e_auto.evidence_sync_enabled(entry)
