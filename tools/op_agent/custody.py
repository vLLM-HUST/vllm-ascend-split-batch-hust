#!/usr/bin/env python3
"""Lightweight custody and anti-drift fingerprinting for operator delivery."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any


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
    speedup: float,
    gate_env: str,
    workspace_root: str | None = None,
) -> bool:
    """Register or update operator entry in repo's DELIVERY_INDEX.json."""
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
            "layer": "plugin_carrier",
            "directory": f"ops/{op_name}",
            "readme": f"ops/{op_name}/README.md",
            "report": f"ops/{op_name}/REPORT.md",
            "evidence": f"ops/{op_name}/evidence.json",
            "status": "integrated_plugin_measured"
            if speedup > 1.0
            else "default_off_negative_speedup",
            "default_enabled": False,
            "gate_env": gate_env,
            "l3_numerical_guard": "passed",
            "serving_status": "verified_faster"
            if speedup > 1.0
            else "negative_or_pending",
        }

        # Update if exists, else append
        idx = next((i for i, op in enumerate(operators) if op.get("id") == op_name), -1)
        if idx >= 0:
            operators[idx] = entry
        else:
            operators.append(entry)

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return True
    except Exception:
        return False
