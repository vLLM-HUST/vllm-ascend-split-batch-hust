import ast
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FETCH = load("benchmark_evidence", "tools/benchmark_evidence.py")
PATHS = load("evidence_paths", "bench/evidence_paths.py")


@pytest.mark.parametrize(
    "name",
    [
        "bench_w31_e2e.py",
        "bench_fisampling_refresh_e2e.py",
        "bench_w31_gelu.py",
        "bench_w31_gelu_graph.py",
    ],
)
def test_active_harness_uses_external_output_guard(name):
    tree = ast.parse((ROOT / "bench" / name).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(node.func, ast.Name) and node.func.id == "output_directory"
        for node in calls
    )


def test_only_reviewed_source_remains_in_evidence_directories():
    source = json.loads((ROOT / "EVIDENCE_SOURCES.json").read_text())
    files = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    candidates = {
        p
        for p in files
        if p.startswith(
            (
                "docs/evidence/",
                "bench/results/",
                "host-proposals/GDN-qkv-norm/validation/",
            )
        )
    }
    retained = source["retained_source_paths"]
    assert len({r["path"] for r in retained}) == len(retained)
    assert candidates == {r["path"] for r in retained}
    for row in retained:
        name, digest = row["path"], row["sha256"]
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "x/../escape", "x//ambiguous", ""]
)
def test_unsafe_download_path(path):
    with pytest.raises(ValueError):
        FETCH.relative_path(path)


def test_output_requires_explicit_external_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("SPLIT_BATCH_RESULTS_DIR", raising=False)
    with pytest.raises(ValueError):
        PATHS.output_directory()
    for path in ["relative", str(ROOT / "bench/results")]:
        monkeypatch.setenv("SPLIT_BATCH_RESULTS_DIR", path)
        with pytest.raises(ValueError):
            PATHS.output_directory()
    monkeypatch.setenv("SPLIT_BATCH_RESULTS_DIR", str(tmp_path))
    assert PATHS.output_directory() == tmp_path


def test_output_rejects_symlink_into_repository(monkeypatch, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(ROOT, target_is_directory=True)
    monkeypatch.setenv("SPLIT_BATCH_RESULTS_DIR", str(link / "results"))
    with pytest.raises(ValueError):
        PATHS.output_directory()


def test_wrong_hash_and_in_repository_cache_rejected():
    with pytest.raises(ValueError):
        FETCH.verify_bytes(b"changed", {"bytes": 0, "sha256": "bad"})
    with pytest.raises(ValueError):
        FETCH.fetch(ROOT / "cache")


def test_corrupt_cache_is_not_overwritten(monkeypatch, tmp_path):
    original = b""
    manifest = {
        "source_commit": "fixture",
        "files": [
            {
                "path": "empty.json",
                "bytes": 0,
                "sha256": hashlib.sha256(original).hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(FETCH, "load_manifest", lambda: ("unused/", manifest))
    (tmp_path / "empty.json").write_bytes(b"bad")
    with pytest.raises(ValueError):
        FETCH.fetch(tmp_path)
    assert (tmp_path / "empty.json").read_bytes() == b"bad"


def test_empty_failure_record_is_not_repaired(monkeypatch, tmp_path):
    manifest = {
        "source_commit": "fixture",
        "files": [
            {
                "path": "empty.json",
                "bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
    }
    monkeypatch.setattr(FETCH, "load_manifest", lambda: ("unused/", manifest))
    monkeypatch.setattr(FETCH, "download", lambda url: b"")
    assert FETCH.fetch(tmp_path)["verified_files"] == 1
    assert (tmp_path / "empty.json").read_bytes() == b""


def test_cache_symlink_escape_rejected(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (cache / "nested").symlink_to(outside, target_is_directory=True)
    manifest = {
        "source_commit": "fixture",
        "files": [{"path": "nested/run.json", "bytes": 0, "sha256": "unused"}],
    }
    monkeypatch.setattr(FETCH, "load_manifest", lambda: ("unused/", manifest))
    with pytest.raises(ValueError, match="escapes"):
        FETCH.fetch(cache)
    assert not list(outside.iterdir())
