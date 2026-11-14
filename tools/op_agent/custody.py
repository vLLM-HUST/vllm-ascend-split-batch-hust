#!/usr/bin/env python3
"""Lightweight custody and anti-drift fingerprinting for operator delivery."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

#: 该算子还没有对外门控开关时的写法（`DELIVERY_INDEX.json` 的既有词汇）。
UNEXPOSED_GATE = "unexposed"

#: 工具新登记时的占位归属层；必须在合入前改成真实层，否则门禁会拦。
UNCLASSIFIED_LAYER = "unclassified"

#: 内环只做了 L0 时的诚实状态与服务状态。
L0_PENDING_STATUS = "l0_measured_pending_outer_loop"
PENDING_SERVING_STATUS = "pending_outer_loop"

#: 更新已有条目时保留的人工结论字段：内环重跑不得改写对外结论。
PRESERVED_ON_UPDATE = (
    "layer",
    "status",
    "serving_status",
    "l3_numerical_guard",
    "gate_env",
    "default_enabled",
)


def calculate_dir_fingerprint(dir_path: str) -> dict[str, str]:
    """Calculate sha256 checksums of all evidence files in an operator directory."""
    fingerprints: dict[str, str] = {}
    if not os.path.isdir(dir_path):
        return fingerprints

    for root, _, files in os.walk(dir_path):
        for f in sorted(files):
            if f.endswith(".sha256"):
                continue
            full_p = os.path.join(root, f)
            rel_p = os.path.relpath(full_p, dir_path)
            try:
                with open(full_p, "rb") as fp:
                    content = fp.read()
                fingerprints[rel_p] = hashlib.sha256(content).hexdigest()
            except OSError:
                continue

    return fingerprints


def write_dir_checksum(dir_path: str) -> str:
    """Generate and write a checksum.sha256 file inside the operator directory."""
    fps = calculate_dir_fingerprint(dir_path)
    checksum_file = os.path.join(dir_path, "checksum.sha256")
    lines = [f"{sha}  {fname}" for fname, sha in sorted(fps.items())]
    with open(checksum_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return checksum_file


def verify_dir_checksum(dir_path: str) -> tuple[bool, str]:
    """Verify that current files match recorded checksums (anti-drift gate)."""
    checksum_file = os.path.join(dir_path, "checksum.sha256")
    if not os.path.isfile(checksum_file):
        return False, f"Missing checksum file: {checksum_file}"

    recorded: dict[str, str] = {}
    with open(checksum_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                recorded[parts[1]] = parts[0]

    current = calculate_dir_fingerprint(dir_path)

    # Compare
    missing = set(recorded) - set(current)
    if missing:
        return False, f"Evidence drift: missing files {sorted(missing)}"

    modified = [k for k in recorded if recorded[k] != current.get(k)]
    if modified:
        return False, f"Evidence drift: modified files {sorted(modified)}"

    return True, "All files matched checksums cleanly."


def register_delivery_index(
    op_name: str,
    callsite: str,
    gate_env: str = UNEXPOSED_GATE,
    layer: str = UNCLASSIFIED_LAYER,
    l3_passed: bool = True,
    workspace_root: str | None = None,
) -> bool:
    """Register or update an operator entry with facts only.

    结论字段（`layer` / `status` / `serving_status` / `gate_env` /
    `l3_numerical_guard`）一律不推断：局部 L0 加速比不能推出服务收益，
    算子名也推不出真实门控变量。

    - 新条目：写保守值 `layer=unclassified`、`status=l0_measured_pending_outer_loop`、
      `serving_status=pending_outer_loop`、`gate_env=unexposed`；`layer` 必须在合入前
      由人工改成真实归属，`tests/test_delivery_status_consistency.py` 会拦占位值。
    - 已有条目：保留人工已定的结论字段，只刷新目录/证据路径与 `display_name`。
    """
    if workspace_root is None:
        workspace_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    index_path = os.path.join(workspace_root, "DELIVERY_INDEX.json")
    if not os.path.isfile(index_path):
        return False

    try:
        with open(index_path, encoding="utf-8") as f:
            data = json.load(f)

        operators = data.setdefault("operators", [])
        entry: dict[str, Any] = {
            "id": op_name,
            "display_name": f"{op_name} ({callsite})",
            "layer": layer,
            "directory": f"ops/{op_name}",
            "readme": f"ops/{op_name}/README.md",
            "report": f"ops/{op_name}/REPORT.md",
            "evidence": f"ops/{op_name}/evidence.json",
            "status": L0_PENDING_STATUS,
            "default_enabled": False,
            "gate_env": gate_env or UNEXPOSED_GATE,
            "l3_numerical_guard": "passed" if l3_passed else "failed",
            "serving_status": PENDING_SERVING_STATUS,
        }

        idx = next((i for i, op in enumerate(operators) if op.get("id") == op_name), -1)
        if idx >= 0:
            # 人工已经定过的结论字段不得被内环重跑覆盖
            current = operators[idx]
            for key in PRESERVED_ON_UPDATE:
                if key in current:
                    entry[key] = current[key]
            operators[idx] = entry
        else:
            operators.append(entry)

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return True
    except Exception:
        return False
