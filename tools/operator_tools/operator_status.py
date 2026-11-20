"""Derive operator progress from artifacts and optional filesystem events."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as exc:
        return {"read_error": str(exc)}


def process_alive(owner, run):
    if not owner or not isinstance(owner.get("pid"), int):
        return False
    try:
        data = (Path("/proc") / str(owner["pid"]) / "cmdline").read_bytes()
        return b"serving_longrun.py" in data and str(run).encode() in data
    except OSError:
        return False


def operator_state(directory, entry, progress):
    result = {"id": entry["id"], "status": "pending"}
    if "reason" in entry:
        return {**result, "status": "not_run", "reason": entry["reason"]}
    failure = read_json(directory / "failure.json")
    l3 = read_json(directory / "serving_l3.json")
    l4 = read_json(directory / "l4.json")
    summary = read_json(directory / "summary.json")
    result.update(numerical=l3, attribution=l4, performance=summary)
    if failure:
        result.update(status="failed", reason=failure)
    elif (l3 and l3.get("passed") is False) or (l4 and l4.get("passed") is False):
        result["status"] = "failed"
    elif summary:
        result["status"] = (
            "collected_pending_review"
            if l3 and l3.get("passed") is True and l4 and l4.get("passed") is True
            else "incomplete_evidence"
        )
    elif progress.get("finished"):
        result["status"] = progress["status"]
    elif directory.exists():
        result["status"] = "running"
    stages = (
        sorted(
            (
                p
                for p in directory.iterdir()
                if p.is_dir() and p.name.startswith(("guard_", "bench_"))
            ),
            key=lambda p: p.stat().st_mtime,
        )
        if directory.exists()
        else []
    )
    if stages:
        stage = stages[-1]
        result["stage"] = stage.name
        log = stage / "server.log"
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 4096))
                lines = handle.read().decode(errors="replace").splitlines()
            result["latest_log"] = lines[-1] if lines else ""
            result["log_age_seconds"] = round(max(0, time.time() - log.stat().st_mtime))
            result["log_path"] = str(log)
        result["stage_exit"] = read_json(stage / "exit.json")
    return result


def collect(run):
    plan = read_json(run / "plan.json")
    if not plan or "operators" not in plan:
        raise ValueError(f"Missing or invalid plan: {run}")
    progress = read_json(run / "progress.json") or []
    if not isinstance(progress, list):
        raise ValueError(f"Invalid progress: {run}")
    previous = {entry["id"]: entry for entry in progress}
    operators = [
        operator_state(run / e["id"], e, previous.get(e["id"], {}))
        for e in plan["operators"]
    ]
    completion = read_json(run / "completion.json")
    alive = process_alive(read_json(run / "owner.json"), run)
    failures = any(o["status"] in {"failed", "incomplete_evidence"} for o in operators)
    stop = read_json(run / "stop-request.json")
    if stop and not alive:
        status = "stopped_by_request"
        for op in operators:
            if op["status"] in {"running", "pending"}:
                op.update(
                    status=(
                        "stopped_by_request" if op["status"] == "running" else "not_run"
                    ),
                    reason=stop.get("reason", "Stopped by request"),
                )
    elif completion:
        status = (
            "failed"
            if completion.get("error")
            else "finished_with_failures"
            if failures
            else "finished_pending_review"
        )
    elif not alive:
        status = "interrupted_or_missing_process"
    else:
        status = (
            "running"
            if any(o["status"] == "running" for o in operators)
            else "waiting_or_between_operators"
        )
    return {
        "run": str(run),
        "status": status,
        "process_alive": alive,
        "operators": operators,
    }


def refresh(runs, output=None, sync_evidence=False, standalone=()):
    report = {"updated_unix": time.time(), "runs": [collect(p) for p in runs]}
    if sync_evidence:
        import sys

        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.insert(0, root)
        from tools.operator_tools.serving_sync import sync

        candidates = [
            (op["id"], Path(run["run"]) / op["id"], op)
            for run in report["runs"]
            for op in run["operators"]
            if op["status"] == "collected_pending_review"
        ]
        for operator, directory in standalone:
            item = {"id": operator, "status": "collected_pending_review"}
            report["runs"].append(
                {"run": str(directory), "status": "standalone", "operators": [item]}
            )
            candidates.append((operator, directory, item))
        for operator, directory, item in candidates:
            try:
                item["sync"] = sync(directory, operator)
            except (ValueError, OSError, KeyError, TypeError, StopIteration) as exc:
                item["sync"] = f"blocked: {exc}"
    lines = []
    for run in report["runs"]:
        lines.append(f"{run['run']}: {run['status']}")
        for op in run["operators"]:
            line = f"  {op['id']}: {op['status']}"
            if op.get("stage"):
                line += f" | {op['stage']} | log age {op.get('log_age_seconds', '?')}s"
            if op.get("sync"):
                line += f" | evidence: {op['sync']}"
            if op.get("reason"):
                reason = op["reason"]
                if not isinstance(reason, str):
                    reason = json.dumps(reason, ensure_ascii=False)
                line += f" | reason: {reason}"
            lines.append(line)
    text = "\n".join(lines) + "\n"
    if output:
        for target, content in (
            (output, json.dumps(report, indent=2, ensure_ascii=False) + "\n"),
            (output.with_suffix(".txt"), text),
        ):
            temp = target.with_suffix(target.suffix + ".tmp")
            temp.write_text(content)
            temp.replace(target)
    return text


def watch(runs, output, sync_evidence=False, standalone=()):
    # Linux inotify blocks until artifacts change; no timed polling or NPU access.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    watched = set()
    try:
        while True:
            for run in runs:
                for directory in [run, *run.glob("*"), *run.glob("*/*")]:
                    if directory.is_dir() and directory not in watched:
                        wd = libc.inotify_add_watch(fd, os.fsencode(directory), 0x188)
                        if wd < 0:
                            raise OSError(ctypes.get_errno(), f"watch {directory}")
                        watched.add(directory)
            print(refresh(runs, output, sync_evidence, standalone), flush=True)
            if all((run / "completion.json").exists() for run in runs):
                return
            events = os.read(fd, 65536)
            offset = 0
            while offset < len(events):
                _, mask, _, length = struct.unpack_from("iIII", events, offset)
                if mask & 0x4000:
                    raise RuntimeError(
                        "inotify queue overflow; restart status observer"
                    )
                offset += 16 + length
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--sync-evidence", action="store_true")
    parser.add_argument(
        "--standalone", action="append", default=[], metavar="OPERATOR=ARCHIVE"
    )
    args = parser.parse_args()
    standalone = [
        (op, Path(path).resolve())
        for op, path in (value.split("=", 1) for value in args.standalone)
    ]
    runs = [Path(p).resolve() for p in args.run]
    if args.output and any(args.output.resolve().is_relative_to(p) for p in runs):
        parser.error("Status output must be outside watched runs")
    if args.watch:
        if not args.output:
            parser.error("--watch requires --output")
        watch(runs, args.output.resolve(), args.sync_evidence, standalone)
    else:
        print(refresh(runs, args.output, args.sync_evidence, standalone), end="")


if __name__ == "__main__":
    main()
