from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def ignored_paths(event):
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    block = re.search(rf"^  {event}:\n(.*?)(?=^  \w|^#)", text, re.M | re.S)
    assert block is not None
    lines = re.findall(r"^      - (.+)$", block[1], re.M)
    return [ast.literal_eval(line) for line in lines]


@pytest.mark.parametrize("event", ["push", "pull_request"])
@pytest.mark.parametrize(
    ("paths", "skip"),
    [
        (["README.md", "docs/agent-guide.md"], True),
        (["docs/evidence/run/result.json", "docs/evidence/run/check.log"], True),
        (["docs/evidence/result.json"], True),
        (["docs/evidence/run/result.json", "src/plugin.py"], False),
        (["docs/evidence/run/reproduce.py"], False),
        (["tests/test_planner.py"], False),
        (["pyproject.toml"], False),
        (["requirements.txt"], False),
        ([".github/workflows/ci.yml"], False),
        (["bundle/manifest.json"], False),
        (["tools/bench-ladder.sh"], False),
    ],
)
def test_ci_filters_only_documentation_and_results(tmp_path, event, paths, skip):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("\n".join(ignored_paths(event)) + "\n")
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "check-ignore", "--no-index", "--stdin"],
        input="\n".join(paths) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    assert (set(result.stdout.splitlines()) == set(paths)) is skip


def test_release_and_manual_full_validation_remain_available():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "  workflow_dispatch:" in ci
    assert 'python: ["3.10", "3.12", "3.14"]' in ci
    assert 'pytest -q -m "not host_tree" --junitxml=' in ci
    assert "continue-on-error" not in ci
    publish = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "paths-ignore" not in publish
    for command in (
        'pytest -q -m "not host_tree"',
        "ruff check .",
        "ruff format --check .",
        "python -m build",
        "python -m twine check dist/*",
    ):
        assert f"- run: {command}" in publish
