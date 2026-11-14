"""内环导出与索引登记的事实性判据。

守两条纪律（来源：docs/pitfalls.md 与 docs/operator-benchmark-standard.md §2）：

1. **不编造门控名**：调用方没给 `gate_env` 时，交付文档写"未对外暴露"，
   不得按算子名推导出代码里并不存在的 `VLLM_HUST_<NAME>`；
2. **不用 L0 推服务结论**：`DELIVERY_INDEX.json` 是对外结论面，内环导出默认
   不登记；显式登记时只写事实字段，结论一律 `pending_outer_loop`，并且
   重跑不得覆盖人工已经定过的结论字段。

这两条正是 `moe_down_residual_norm`、`ffn_activation` 与 `gdn_split_ba`
早期条目的成因，所以在这里锁死。
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.op_agent.custody import register_delivery_index
from tools.op_agent.exporter import export_operator_delivery

OP = "demo_op"
CALLSITE = "vllm_ascend/ops/demo.py:demo_forward"
WORKLOAD = {
    "primary_workload": "random",
    "stress_workload": "sharegpt",
    "cross_check_workload": "long-prefix",
    "primary_concurrency": 32,
}
KERNEL = "def optimized_op(x):\n    return x\n"


def _export(workspace: Path, **overrides):
    kwargs = {
        "op_name": OP,
        "callsite": CALLSITE,
        "baseline_latency": 8.0,
        "optimized_latency": 2.0,
        "kernel_code": KERNEL,
        "shape_facts": {"phase": "decode", "T": 16},
        "workload_spec": WORKLOAD,
        "workspace_root": str(workspace),
    }
    kwargs.update(overrides)
    return export_operator_delivery(**kwargs)


def _index(workspace: Path) -> dict:
    return json.loads((workspace / "DELIVERY_INDEX.json").read_text(encoding="utf-8"))


def test_export_without_gate_env_does_not_invent_a_switch(tmp_path):
    """没给门控变量时交付文档写"未对外暴露"，不出现推导出来的环境变量名。"""
    out_dir = Path(_export(tmp_path))
    for path in out_dir.iterdir():
        assert "VLLM_HUST_" not in path.read_text(encoding="utf-8"), (
            f"{path.name} 里出现了推导出来的门控变量名"
        )
    assert "未对外暴露" in (out_dir / "README.md").read_text(encoding="utf-8")
    assert "未对外暴露" in (out_dir / "REPORT.md").read_text(encoding="utf-8")


def test_export_does_not_touch_delivery_index_by_default(tmp_path):
    """内环导出默认不写索引：索引变更必须是显式动作。"""
    index_path = tmp_path / "DELIVERY_INDEX.json"
    index_path.write_text(
        json.dumps({"schema": "operator-delivery-index/v1", "operators": []}),
        encoding="utf-8",
    )
    _export(tmp_path)
    assert _index(tmp_path)["operators"] == []


def test_explicit_registration_writes_pending_conclusions(tmp_path):
    """显式登记也只写事实：L0 再快也不得写成已集成或服务更快。"""
    (tmp_path / "DELIVERY_INDEX.json").write_text(
        json.dumps({"schema": "operator-delivery-index/v1", "operators": []}),
        encoding="utf-8",
    )
    _export(tmp_path, register_entry=True)

    entry = _index(tmp_path)["operators"][0]
    assert entry["id"] == OP
    assert entry["gate_env"] == "unexposed"
    assert entry["layer"] == "unclassified"
    assert entry["status"] == "l0_measured_pending_outer_loop"
    assert entry["serving_status"] == "pending_outer_loop"
    assert entry["default_enabled"] is False


def test_registration_preserves_human_conclusions_on_rerun(tmp_path):
    """重跑内环不得把人工定过的结论字段改回去。"""
    human_entry = {
        "id": OP,
        "display_name": f"{OP} ({CALLSITE})",
        "layer": "host_fork",
        "directory": f"ops/{OP}",
        "readme": f"ops/{OP}/README.md",
        "report": f"ops/{OP}/REPORT.md",
        "evidence": f"ops/{OP}/evidence.json",
        "status": "integrated_host_fork_measured",
        "default_enabled": False,
        "gate_env": "VLLM_HUST_DEMO=1",
        "l3_numerical_guard": "passed",
        "serving_status": "within_noise",
    }
    (tmp_path / "DELIVERY_INDEX.json").write_text(
        json.dumps(
            {"schema": "operator-delivery-index/v1", "operators": [human_entry]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _export(tmp_path, register_entry=True, layer="unclassified", gate_env="")

    entry = _index(tmp_path)["operators"][0]
    for key, value in (
        ("layer", "host_fork"),
        ("status", "integrated_host_fork_measured"),
        ("serving_status", "within_noise"),
        ("gate_env", "VLLM_HUST_DEMO=1"),
        ("l3_numerical_guard", "passed"),
    ):
        assert entry[key] == value, f"{key} 被内环重跑覆盖成了 {entry[key]!r}"


def test_registration_reports_missing_index(tmp_path):
    """没有索引文件时返回 False，不静默创建一份。"""
    assert register_delivery_index(OP, CALLSITE, workspace_root=str(tmp_path)) is False
    assert not (tmp_path / "DELIVERY_INDEX.json").exists()
