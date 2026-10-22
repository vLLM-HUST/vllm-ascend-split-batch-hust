#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

LOG = logging.getLogger("npu-idle-keeper")
SCRIPT = Path(__file__).resolve()
DEFAULT_LOCK = "/root/log/npu-idle-keeper.pid"


def smi(*args: str) -> str:
    return subprocess.check_output(
        ["npu-smi", "info", *args], text=True, stderr=subprocess.STDOUT, timeout=5
    )


def parse_mapping(text: str) -> dict[int, int]:
    devices = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+Ascend\s+.+", line)
        if match:
            card, _chip, logical = map(int, match.groups())
            devices[card] = logical
    return devices


def parse_processes(text: str, cards: set[int]) -> set[int]:
    if "Process id" not in text:
        raise ValueError("missing NPU process table")
    table = text.split("Process id", 1)[1]
    seen = set()
    pids = set()
    for line in table.splitlines():
        empty = re.search(r"No running processes found in NPU (\d+)", line)
        row = re.fullmatch(
            r"\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*", line
        )
        if empty:
            seen.add(int(empty[1]))
        elif row:
            card, _chip, pid = map(int, row.groups())
            seen.add(card)
            if card in cards:
                pids.add(pid)
    if not cards.issubset(seen):
        raise ValueError(f"incomplete NPU process table: expected {cards}, got {seen}")
    return pids


def parse_usage(text: str) -> int:
    rates = re.findall(r"Ai(?:core|vector) Usage Rate\(%\)\s*:\s*(\d+)", text)
    if not rates:
        raise ValueError("missing NPU utilization")
    values = list(map(int, rates))
    if any(value > 100 for value in values):
        raise ValueError("invalid NPU utilization")
    return max(values)


def busy_reason(devices: dict[int, int], own_pids: set[int], threshold: int) -> str:
    other = parse_processes(smi(), set(devices)) - own_pids
    if other:
        return f"other NPU processes: {sorted(other)}"
    for card in devices:
        usage = parse_usage(smi("-t", "usages", "-i", str(card)))
        if usage >= threshold:
            return f"NPU {card} utilization {usage}% >= {threshold}%"
    return ""


def worker_pids(worker: subprocess.Popen | None) -> set[int]:
    if worker is None:
        return set()
    pids = {worker.pid}
    try:
        text = Path(f"/proc/{worker.pid}/status").read_text()
        match = re.search(r"^NSpid:\s*(.+)$", text, re.MULTILINE)
        if match:
            pids.update(map(int, match[1].split()))
    except (OSError, ValueError):
        pass
    return pids


def stop_worker(worker: subprocess.Popen | None) -> None:
    if worker is None:
        return
    if worker.poll() is None:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)
    LOG.info("worker pid=%d stopped and reaped", worker.pid)


def start_worker(args: argparse.Namespace, devices: dict[int, int]) -> subprocess.Popen:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices.values()))
    return subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "worker",
            "--pulse-ms",
            str(args.pulse_ms),
            "--period-seconds",
            str(args.period_seconds),
            "--parent-pid",
            str(os.getpid()),
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )


def run(args: argparse.Namespace, stopped: threading.Event) -> int:
    lock_path = Path(args.lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info("already running")
            return 0
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()))
        lock.flush()
        devices = parse_mapping(smi("-m"))
        if not devices:
            LOG.info("no allocated NPU; skipped")
            return 0
        LOG.info("supervisor pid=%d devices=%s", os.getpid(), devices)
        worker = None
        idle_since = None
        retry_after = 0.0
        state = None
        deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
        try:
            while not stopped.is_set():
                if deadline is not None and time.monotonic() >= deadline:
                    break
                now = time.monotonic()
                if worker is not None and worker.poll() is not None:
                    LOG.error(
                        "worker exited rc=%s; retry in 60 seconds", worker.returncode
                    )
                    worker = None
                    idle_since = None
                    retry_after = now + 60
                try:
                    reason = busy_reason(
                        devices, worker_pids(worker), args.busy_percent
                    )
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    reason = f"monitor unavailable: {exc}"
                if reason:
                    if state != reason:
                        LOG.info("sleeping: %s", reason)
                    stop_worker(worker)
                    worker = None
                    idle_since = None
                    state = reason
                else:
                    if idle_since is None:
                        idle_since = now
                        LOG.info("idle; waiting %.1fs", args.idle_seconds)
                    if (
                        worker is None
                        and now >= retry_after
                        and now - idle_since >= args.idle_seconds
                    ):
                        worker = start_worker(args, devices)
                        LOG.info("worker pid=%d started", worker.pid)
                    state = "idle"
                stopped.wait(args.poll_seconds)
        finally:
            stop_worker(worker)
        return 0


def workload(args: argparse.Namespace, stopped: threading.Event) -> int:
    import ctypes

    if not args.parent_pid:
        raise ValueError("worker requires a supervisor")
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot install parent-death signal")
    if os.getppid() != args.parent_pid:
        return 1

    import torch
    import torch_npu  # noqa: F401

    tensors = []
    for device in range(torch.npu.device_count()):
        torch.npu.set_device(device)
        left = torch.ones((512, 512), dtype=torch.float16, device=f"npu:{device}")
        right = torch.ones_like(left)
        out = torch.empty_like(left)
        torch.mm(left, right, out=out)
        torch.npu.synchronize(device)
        tensors.append((device, left, right, out))
    if not tensors:
        raise RuntimeError("no NPU visible to worker")
    LOG.info(
        "worker ready: %d devices, 512x512 fp16, %.1fms/device/%.2fs",
        len(tensors),
        args.pulse_ms,
        args.period_seconds,
    )
    while not stopped.is_set():
        begin = time.monotonic()
        for device, left, right, out in tensors:
            if stopped.is_set():
                break
            torch.npu.set_device(device)
            until = time.monotonic() + args.pulse_ms / 1000
            while not stopped.is_set() and time.monotonic() < until:
                torch.mm(left, right, out=out)
                torch.npu.synchronize(device)
        stopped.wait(max(0, args.period_seconds - (time.monotonic() - begin)))
    return 0


def control(args: argparse.Namespace) -> int:
    try:
        with Path(args.lock).open("r+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pid = int(lock.read().strip())
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                cwd = Path(f"/proc/{pid}/cwd").resolve()
                paths = [(cwd / os.fsdecode(arg)).resolve() for arg in cmd[1:] if arg]
                if SCRIPT not in paths or b"run" not in cmd:
                    raise ValueError("PID does not belong to this supervisor") from None
                if args.action == "stop":
                    os.kill(pid, signal.SIGTERM)
                    LOG.info("stop requested for supervisor pid=%d", pid)
                else:
                    LOG.info("supervisor running pid=%d", pid)
                return 0
    except FileNotFoundError:
        pass
    LOG.info("not running")
    return 0 if args.action == "stop" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "worker", "status", "stop", "check"))
    parser.add_argument("--lock", default=DEFAULT_LOCK)
    parser.add_argument("--poll-seconds", type=float, default=2)
    parser.add_argument("--idle-seconds", type=float, default=15)
    parser.add_argument("--busy-percent", type=int, default=15)
    parser.add_argument("--pulse-ms", type=float, default=10)
    parser.add_argument("--period-seconds", type=float, default=1)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()
    if not (0 < args.poll_seconds <= 60 and 0 <= args.idle_seconds <= 3600):
        parser.error("invalid polling/idle interval")
    if not (1 <= args.busy_percent <= 100 and args.max_seconds >= 0):
        parser.error("invalid utilization threshold/runtime limit")
    if not (0 < args.period_seconds <= 60 and 0 < args.pulse_ms <= 20):
        parser.error("invalid pulse/period")
    if args.pulse_ms / 1000 / args.period_seconds > 0.05:
        parser.error("per-device duty cycle must not exceed 5%")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())
    try:
        if args.action in ("status", "stop"):
            return control(args)
        if args.action == "worker":
            return workload(args, stopped)
        if args.action == "check":
            devices = parse_mapping(smi("-m"))
            reason = (
                busy_reason(devices, set(), args.busy_percent) if devices else "no NPU"
            )
            LOG.info("devices=%s state=%s", devices, reason or "idle")
            return 0
        return run(args, stopped)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        LOG.error("stopped safely: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
