# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/operator_delivery.py"
spec = importlib.util.spec_from_file_location("operator_delivery", TOOL)
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True,
        ).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    (tmp_path / "kernel.py").write_text("old\n")
    (tmp_path / "other.py").write_text("unrelated\n")
    git("add", "kernel.py", "other.py")
    git("commit", "-qm", "base")
    return tmp_path, git, git("rev-parse", "HEAD")


def test_historical_commit_is_not_delivery(repo):
    path, _, base = repo
    assert delivery.collect(path)["state"] == "UNSCOPED"
    report = delivery.collect(path, base, ["kernel.py"])
    assert report["state"] == "INCOMPLETE"
    assert report["remote"] == "NOT_VERIFIED"
    assert report["acceptance"] == "NOT_CHECKED"


def test_scoped_commit_ignores_unrelated_dirty(repo):
    path, git, base = repo
    (path / "kernel.py").write_text("new\n")
    assert delivery.collect(path, base, ["kernel.py"])["state"] == "INCOMPLETE"
    git("add", "kernel.py")
    assert delivery.collect(path, base, ["kernel.py"])["state"] == "INCOMPLETE"
    git("commit", "-qm", "task")
    (path / "other.py").write_text("user work\n")
    before = git("status", "--porcelain")
    report = delivery.collect(path, base, ["kernel.py"])
    assert report["state"] == "COMMITTED_LOCAL"
    assert report["remote"] == "NOT_VERIFIED"
    assert git("status", "--porcelain") == before
    assert delivery.collect(path, base, ["missing.py"])["state"] == "INCOMPLETE"
    (path / "kernel.py").write_text("unfinished\n")
    assert delivery.collect(path, base, ["kernel.py"])["state"] == "INCOMPLETE"


@pytest.mark.parametrize("scope", [".", "..", "../x", "/tmp/x", ":(all)", "*.py"])
def test_reject_unbounded_scope(repo, scope):
    path, _, base = repo
    with pytest.raises(ValueError):
        delivery.collect(path, base, [scope])


def test_untracked_scoped_file_blocks(repo):
    path, git, base = repo
    (path / "impl").mkdir()
    (path / "impl/a.py").write_text("ok")
    git("add", "impl/a.py")
    git("commit", "-qm", "implementation")
    (path / "impl/b.py").write_text("not committed")
    assert delivery.collect(path, base, ["impl"])["state"] == "INCOMPLETE"


def test_non_ancestor_rejected(repo):
    path, git, base = repo
    git("checkout", "-qb", "side")
    (path / "kernel.py").write_text("side")
    git("commit", "-qam", "side")
    side = git("rev-parse", "HEAD")
    git("checkout", "-q", base)
    with pytest.raises(ValueError):
        delivery.collect(path, side, ["kernel.py"])


def test_cli_unscoped_is_not_successful_check(repo):
    path, _, _ = repo
    result = subprocess.run(
        [sys.executable, str(TOOL), "--repo", str(path), "--check", "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 3
    assert json.loads(result.stdout)["state"] == "UNSCOPED"


def test_not_a_repo_is_error(tmp_path):
    result = subprocess.run(
        [sys.executable, str(TOOL), "--repo", str(tmp_path), "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 4
    assert json.loads(result.stdout)["state"] == "ERROR"
