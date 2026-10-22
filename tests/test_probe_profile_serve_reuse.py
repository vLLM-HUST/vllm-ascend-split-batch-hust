import importlib.util
import os
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/probe_profile_serve_reuse.py"
spec = importlib.util.spec_from_file_location("reuse_probe", SOURCE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize(
    "profile,plain,expected",
    [
        ([100, 100], [100, 100], "supports-reuse"),
        ([100.4, 100.4], [100, 100], "supports-reuse"),
        ([98, 98], [100, 100], "rejects-reuse"),
        ([102, 102], [100, 100], "rejects-reuse"),
        ([100.7, 100.7], [100, 100], "inconclusive"),
        ([90, 100], [100, 100], "inconclusive"),
        ([100, 100], [90, 100], "inconclusive"),
    ],
)
def test_classification(profile, plain, expected):
    assert probe.classify(profile, plain) == expected


def test_linux_identity_and_ancestry():
    info = probe.identity(os.getpid())
    assert info["ppid"] == os.getppid()
    assert info["pgid"] == os.getpgrp()
    assert info["start_ticks"] > 0
    assert probe.descendant(os.getpid(), os.getppid())
    assert not probe.descendant(1, os.getpid())


def test_evidence_is_append_only(tmp_path):
    output = tmp_path / "evidence.json"
    probe.dump(output, {"first": True})
    with pytest.raises(FileExistsError):
        probe.dump(output, {"replacement": True})
    assert "replacement" not in output.read_text()


def test_cleanup_refuses_recycled_pid(tmp_path, monkeypatch):
    runner = probe.Probe(tmp_path)
    runner.owner = {"pid": 123, "start_ticks": 1}
    monkeypatch.setattr(
        probe, "identity", lambda pid: {"pid": pid, "start_ticks": 2, "pgid": pid}
    )
    monkeypatch.setattr(probe.os, "killpg", lambda *args: pytest.fail("unowned kill"))
    with pytest.raises(RuntimeError, match="replaced process"):
        runner.stop("profile")


def test_protocol_has_no_collection_or_broad_cleanup():
    source = SOURCE.read_text()
    assert "start_profile" not in source
    assert "pkill" not in source
    assert "['stop']" not in source
    assert 'for arm in ("profile", "plain"):' in source
