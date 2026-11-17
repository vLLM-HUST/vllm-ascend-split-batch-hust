# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

"""Boundary guard for ``zerocost_wiring`` (see boundary spec §3.5).

The module rewrites already-imported host *method attributes*. That is allowed.
Rewriting host source text through a meta-path hook is the banned class, so the
three conditions that keep this module on the allowed side are pinned here: the
gates keep it inert, an uninstall path exists, and nothing touches the disk
sources.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "vllm_ascend_split_batch"
MODULE = SRC / "zerocost_wiring.py"


def _tree() -> ast.Module:
    return ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))


def _toplevel_names() -> set[str]:
    return {
        node.name
        for node in _tree().body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_install_returns_early_when_no_gate_is_enabled() -> None:
    """Both gates off must mean nothing is installed during load."""
    fn = next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "install"
    )
    early_returns = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Name)
        and node.test.operand.id == "wanted"
        and any(isinstance(stmt, ast.Return) for stmt in node.body)
    ]
    assert early_returns, "install() must return before patching when no gate is on"


def test_uninstall_is_exposed() -> None:
    """The exception is only valid while the rewrite can be rolled back."""
    assert "uninstall" in _toplevel_names()


def test_no_meta_path_source_transform() -> None:
    """Meta-path hooks rewrite source text for every importer: banned class."""
    text = MODULE.read_text(encoding="utf-8")
    for banned in ("MetaPathFinder", "find_spec", "exec_module", "SourceFileLoader"):
        assert banned not in text, f"{banned} puts this module in the banned class"


def test_rewrites_attributes_not_host_files() -> None:
    """Allowed side of the line: setattr on imported objects, no file edits."""
    text = MODULE.read_text(encoding="utf-8")
    assert "setattr" in text
    for banned in ("shutil", "write_text", "apply-runtime-patches"):
        assert banned not in text, f"{banned} suggests on-disk host patching"
