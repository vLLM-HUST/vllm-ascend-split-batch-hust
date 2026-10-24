# SPDX-License-Identifier: Apache-2.0

"""Host-binding policy for the test suite and the repository root.

Three test kinds are allowed: CPU-only, ``host_tree`` marked (deselected on
the dependency-free CI job), or host-probing behind ``_deployment`` guards.
The fourth kind -- "runs everywhere, explodes only off this host" -- turned
all three CI legs red on PR #19: a module-level ``Path.is_file()`` on an
unreadable ``/root`` path raises ``PermissionError`` and takes the whole
collection down before any test runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
HOST_ROOTS = ("/root/", "/usr/local/Ascend", "/vllm-workspace/")
GUARDS = {"deployed_file", "deployed_dir"}
PROBES = {
    "is_file",
    "is_dir",
    "exists",
    "stat",
    "glob",
    "rglob",
    "read_text",
    "read_bytes",
}


def _has_host_literal(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and n.value.startswith(HOST_ROOTS)
        for n in ast.walk(node)
    )


def _host_names(tree: ast.Module) -> set[str]:
    """Module-level names whose value is or contains a host absolute path."""
    names = set()
    for stmt in tree.body:
        targets = []
        value = None
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        if value is None or not _has_host_literal(value):
            continue
        names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _host_probes(tree: ast.Module, names: set[str]) -> list[tuple[int, str, bool]]:
    """(lineno, attribute, at_import_time) for each probe of a host path."""
    found = []
    import_time_nodes = {
        id(node)
        for stmt in tree.body
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(stmt)
    }
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in PROBES
        ):
            continue
        chain = node.func.value
        hits = _has_host_literal(chain) or any(
            isinstance(n, ast.Name) and n.id in names for n in ast.walk(chain)
        )
        if hits:
            found.append((node.lineno, node.func.attr, id(node) in import_time_nodes))
    return found


def _guarded(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "_deployment":
            return any(alias.name in GUARDS for alias in node.names)
    return False


def _violations() -> list[str]:
    problems = []
    for path in sorted(TESTS.glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        marked = "host_tree" in source
        guarded = _guarded(tree)
        probes = _host_probes(tree, _host_names(tree))
        if not probes or marked:
            continue
        if not guarded:
            problems.append(
                f"{path.name}: host path probe at line(s) "
                f"{sorted({p[0] for p in probes})} without host_tree marker or "
                "_deployment guard"
            )
        for lineno, attr, at_import_time in probes:
            if at_import_time:
                problems.append(
                    f"{path.name}: import-time .{attr}() at line {lineno} raises "
                    "PermissionError on the CI runner before collection"
                )
    return problems


def test_every_host_probe_is_marked_or_guarded():
    problems = _violations()
    assert not problems, (
        "host-binding policy violation(s): an unguarded probe fails the whole "
        "CI collection, not just itself. Either add the `host_tree` marker "
        "(deselected on CI) or guard with `_deployment.deployed_file/"
        "deployed_dir`:\n  " + "\n  ".join(problems)
    )


def test_repository_root_carries_no_measurement_dumps():
    """Ladder-window dumps belong in the evidence store, not the code repo."""
    offenders = [
        p.name
        for p in sorted(REPO.iterdir())
        if p.is_file() and p.suffix == ".json" and p.name[:4].isdigit()
    ]
    assert not offenders, (
        "measurement dumps at repository root (move to the evidence store and "
        "keep only a pointer): " + ", ".join(offenders)
    )
