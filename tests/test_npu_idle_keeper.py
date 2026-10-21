from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/npu_idle_keeper.py"
spec = importlib.util.spec_from_file_location("npu_idle_keeper", SCRIPT)
keeper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keeper)

MAPPING = """
NPU ID Chip ID Chip Logic ID Chip Name
3 0 0 Ascend 910B2
3 1 - Mcu
6 0 1 Ascend 910B2
"""
EMPTY = """| NPU Chip | Process id | Process name | Process memory(MB) |
| No running processes found in NPU 3 |
| No running processes found in NPU 6 |
"""


def test_mapping_and_empty_processes():
    assert keeper.parse_mapping(MAPPING) == {3: 0, 6: 1}
    assert keeper.parse_mapping("No devices found") == {}
    assert keeper.parse_processes(EMPTY, {3, 6}) == set()


def test_process_table_includes_unknown_namespace_process():
    text = EMPTY.replace(
        "| No running processes found in NPU 3 |",
        "| 3 0 | 123 | Get pid name failed | 400 |",
    )
    assert keeper.parse_processes(text, {3, 6}) == {123}


@pytest.mark.parametrize("text", ["garbled", "Process id\n", EMPTY.splitlines()[0]])
def test_bad_process_table_fails_closed(text):
    with pytest.raises(ValueError):
        keeper.parse_processes(text, {3, 6})


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Aicore Usage Rate(%) : 0\nAivector Usage Rate(%) : 19", 19),
        ("Aicore Usage Rate(%) : 2", 2),
    ],
)
def test_usage(text, expected):
    assert keeper.parse_usage(text) == expected


@pytest.mark.parametrize(
    "text", ["", "Aicore Usage Rate(%) : N/A", "Aicore Usage Rate(%) : 101"]
)
def test_bad_usage_fails_closed(text):
    with pytest.raises(ValueError):
        keeper.parse_usage(text)


def test_busy_checks_own_pid_foreign_pid_and_utilization(monkeypatch):
    row = EMPTY.replace(
        "No running processes found in NPU 3", "3 0 | 123 | python | 400"
    )
    monkeypatch.setattr(
        keeper, "smi", lambda *a: row if not a else "Aicore Usage Rate(%) : 0"
    )
    assert keeper.busy_reason({3: 0, 6: 1}, {123}, 15) == ""
    assert "123" in keeper.busy_reason({3: 0, 6: 1}, set(), 15)
    monkeypatch.setattr(
        keeper, "smi", lambda *a: EMPTY if not a else "Aicore Usage Rate(%) : 16"
    )
    assert "utilization" in keeper.busy_reason({3: 0, 6: 1}, set(), 15)


def test_controller_yields_releases_waits_then_resumes(tmp_path, monkeypatch):
    clock = [0.0]
    reasons = iter(["", "", "foreign process", "", "", "", ValueError("bad")])
    workers = []

    class Worker:
        pid = 99999999
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    class Event:
        iterations = 0

        def is_set(self):
            return self.iterations == 7

        def wait(self, seconds):
            clock[0] += seconds
            self.iterations += 1

    def reason(*_):
        result = next(reasons)
        if isinstance(result, Exception):
            raise result
        return result

    def start(*_):
        worker = Worker()
        workers.append((clock[0], worker))
        return worker

    monkeypatch.setattr(keeper.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(keeper, "smi", lambda *_: MAPPING)
    monkeypatch.setattr(keeper, "busy_reason", reason)
    monkeypatch.setattr(keeper, "start_worker", start)
    args = SimpleNamespace(
        lock=str(tmp_path / "pid"),
        max_seconds=0,
        idle_seconds=2,
        poll_seconds=2,
        busy_percent=15,
    )
    assert keeper.run(args, Event()) == 0
    assert [start for start, _ in workers] == [2, 8]
    assert all(worker.returncode == 0 for _, worker in workers)


def test_no_npu_never_starts_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(keeper, "smi", lambda *_: "No devices found")
    monkeypatch.setattr(
        keeper, "start_worker", lambda *_: pytest.fail("unexpected NPU")
    )
    args = SimpleNamespace(lock=str(tmp_path / "pid"))
    assert keeper.run(args, threading.Event()) == 0


def test_worker_forced_cleanup_only_targets_owned_child():
    actions = []

    class Worker:
        pid = 123

        def poll(self):
            return None

        def terminate(self):
            actions.append("terminate")

        def kill(self):
            actions.append("kill")

        def wait(self, timeout):
            if actions[-1] != "kill":
                raise subprocess.TimeoutExpired("worker", timeout)
            actions.append("reaped")

    keeper.stop_worker(Worker())
    assert actions == ["terminate", "kill", "reaped"]


def test_control_ignores_stale_pid_file(tmp_path):
    lock = tmp_path / "pid"
    lock.write_text(str(__import__("os").getpid()))
    assert keeper.control(SimpleNamespace(lock=str(lock), action="stop")) == 0
    assert keeper.control(SimpleNamespace(lock=str(lock), action="status")) == 1


def test_relative_path_status_and_stop(tmp_path):
    lock = tmp_path / "supervisor.pid"
    script = str(SCRIPT.relative_to(SCRIPT.parents[1]))
    fake_smi = tmp_path / "npu-smi"
    fake_smi.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"print({MAPPING!r} if '-m' in sys.argv else {EMPTY!r} "
        "if len(sys.argv) == 2 else 'Aicore Usage Rate(%) : 0')\n"
    )
    fake_smi.chmod(0o755)
    child = subprocess.Popen(
        [sys.executable, script, "run", "--lock", str(lock), "--idle-seconds", "3600"],
        cwd=SCRIPT.parents[1],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            if lock.exists() and lock.read_text() == str(child.pid):
                break
            time.sleep(0.05)
        else:
            pytest.fail("supervisor did not start")
        assert keeper.control(SimpleNamespace(lock=str(lock), action="status")) == 0
        assert keeper.control(SimpleNamespace(lock=str(lock), action="stop")) == 0
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)


@pytest.mark.parametrize(
    "flags",
    [
        ["--pulse-ms", "100"],
        ["--period-seconds", "0"],
        ["--period-seconds", ".01"],
        ["--poll-seconds", "0"],
    ],
)
def test_cli_rejects_unbounded_workload(flags):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "run", *flags],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2


@pytest.mark.parametrize("enabled, expected", [("0", "stop"), ("1", "run")])
def test_startup_hook_uses_disable_switch_and_detaches(tmp_path, enabled, expected):
    patch = SCRIPT.with_name("start-sshd-npu-keeper.patch").read_text()
    block = "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    syntax = subprocess.run(
        ["bash", "-n"], input=block, text=True, capture_output=True, timeout=5
    )
    assert syntax.returncode == 0, syntax.stderr
    tools = tmp_path / "tools"
    tools.mkdir()
    events = tmp_path / "events"
    fake = tools / "npu_idle_keeper.py"
    fake.write_text(
        "import pathlib, sys\n"
        "if sys.argv[1] == 'status': raise SystemExit(1)\n"
        f"with pathlib.Path({str(events)!r}).open('a') as f: "
        "f.write(sys.argv[1] + '\\n')\n"
    )
    fake_smi = tmp_path / "npu-smi"
    fake_smi.write_text("#!/bin/sh\nexit 0\n")
    fake_smi.chmod(0o755)
    result = subprocess.run(
        ["bash", "-c", block],
        capture_output=True,
        text=True,
        timeout=5,
        env={
            **os.environ,
            "HUSK": str(tmp_path),
            "PY": sys.executable,
            "LOG_DIR": str(tmp_path),
            "NPU_IDLE_KEEPALIVE": enabled,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
        },
    )
    assert result.returncode == 0, result.stderr
    for _ in range(100):
        if events.exists() and expected in events.read_text().splitlines():
            break
        time.sleep(0.02)
    else:
        pytest.fail("startup action did not run")
    assert events.read_text().splitlines() == [expected]
