"""Tests for the pending-operator long-run plan builder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import pending_longrun  # noqa: E402


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
    assert by_id["moe_down_residual_norm"]["config"]["guard_only"] is True
    assert by_id["ffn_activation"]["config"]["guard_only"] is True
    assert {row["mode"] for row in report} == {"skip", "full", "guard_only"}


def test_mode_override_can_disable_a_full_entry():
    plan, _ = pending_longrun.build_plan(_template(), modes={"gdn_split_ba": "skip"})
    entry = next(row for row in plan["operators"] if row["id"] == "gdn_split_ba")
    assert list(entry) == ["id", "reason"]
