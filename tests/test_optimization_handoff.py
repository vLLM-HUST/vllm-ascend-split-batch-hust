"""A clean checkout can select a design task, but never schedule a service."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT))


def run_next(archive):
    return subprocess.run(
        [sys.executable, "-m", "tools.operator_tools.next", "--archive", str(archive)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_clean_checkout_without_archive(tmp_path):
    run = run_next(tmp_path / "absent")
    assert run.returncode == 2, run.stderr
    assert "Recurrent" in run.stdout and "1/2" in run.stdout
    assert "Chunk" in run.stdout and "gdn_split_ba" in run.stdout
    assert "只可静态设计" in run.stdout and "final-receipt.json" in run.stdout
    assert list(tmp_path.iterdir()) == []


def test_longrun_launcher_requires_operator_before_making_run_dir(tmp_path):
    import os

    run_dir = tmp_path / "run"
    proc = subprocess.run(
        ["bash", ROOT / "workloads/run_pending_longrun.sh"],
        env={
            **os.environ,
            "PY": sys.executable,
            "RUN_DIR": str(run_dir),
            "SERVING_BASE_PLAN": "",
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2 and "operator" in proc.stderr
    assert not run_dir.exists()


def test_wrong_receipt_does_not_unlock_replay(tmp_path):
    (tmp_path / "final-receipt.json").write_text("{}")
    run = run_next(tmp_path)
    assert run.returncode == 1 and "SHA256" in run.stderr


def test_verify_linked_sources_and_host_identity(tmp_path):
    from tools.operator_tools import next as handoff

    record = json.loads(handoff.RECORD.read_text())
    hosts = record["host_commits"]

    def put(name, obj):
        data = json.dumps(obj).encode()
        (tmp_path / name).write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    (tmp_path / "exit-code").write_text("1\n")
    trace = put("trace.json", [])
    result = put("result.json", {"source_sha256": {"trace.json": trace}})
    preflight = put(
        "host_preflight.json",
        {
            "returncode": 0,
            "stdout": json.dumps({k: {"revision": v} for k, v in hosts.items()}),
        },
    )
    receipt = {
        "host_commits": hosts,
        "analysis_exit": 0,
        "raw_collection_completed": True,
        "collector_exit": 1,
        "source_hashes": {"result.json": result, "host_preflight.json": preflight},
    }
    record["receipt_sha256"] = put("final-receipt.json", receipt)
    handoff.verify(record, tmp_path)
    with pytest.raises(ValueError, match="宿主"):
        handoff.verify(
            {**record, "host_commits": {**hosts, "vllm": "a" * 40}}, tmp_path
        )
    put("trace.json", [1])
    with pytest.raises(ValueError, match="trace.json"):
        handoff.verify(record, tmp_path)
    (tmp_path / "trace.json").unlink()
    with pytest.raises(FileNotFoundError, match="trace.json"):
        handoff.verify(record, tmp_path)
