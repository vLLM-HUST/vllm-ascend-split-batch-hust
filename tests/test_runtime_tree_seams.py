"""CPU-only guards for host seam assertions and runtime-patch retirement.

Tests here verify static host interface contracts without touching /vllm-workspace
or importing vllm.
"""

from __future__ import annotations

import json
import os
import pathlib
import py_compile
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
RT = HERE.parent / "tools" / "runtime-tree"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


@pytest.mark.host_tree
def test_indexed_service_routes_resolve_on_the_target_host():
    from tools.operator_tools import host_runtime, serving_collect

    package = (
        pathlib.Path(os.environ["VLLM_ASCEND_HUST_ROOT"]).resolve() / "vllm_ascend"
    )
    entries = json.loads((HERE.parent / "DELIVERY_INDEX.json").read_text())["operators"]
    for entry in entries:
        if entry["gate_env"] == "unexposed":
            assert "service_route" not in entry
            continue
        (route,) = serving_collect.service_routes(
            {"gate": entry["gate_env"].split("=")[0]}
        )
        assert host_runtime.check_service_route(route, package)["identity"] == route


def test_seam_checker_compiles_and_selftests():
    checker = RT / "check-tree-seams.py"
    assert checker.is_file(), checker
    py_compile.compile(str(checker), doraise=True)
    assert checker.stat().st_mode & 0o111, "部署脚本须保留可执行位"
    proc = _run([str(checker), "--selftest"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_seam_table_schema():
    table = json.loads((RT / "tree-seam-assertions.json").read_text())
    entries = table["seams"]
    assert isinstance(entries, list) and entries
    for e in entries:
        assert e["kind"] in ("symbol", "text"), e
        assert e.get("consumer") and e.get("call_token"), e
        if e["kind"] == "symbol":
            assert e.get("definer") and e.get("symbol"), e


def test_runtime_patching_script_is_retired():
    """Runtime tree patching script is retired; modifications live in host fork."""
    assert not (RT / "apply-runtime-patches.sh").exists()
    assert not (RT / "portable_runtime.py").exists()
    assert not (RT / "patch-manifest.json").exists()
