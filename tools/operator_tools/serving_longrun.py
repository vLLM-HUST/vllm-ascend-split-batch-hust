"""Serial serving collection without automatic delivery evidence promotion.

Superseded by `serving_e2e_auto.py` (resumable, multi-gate, evidence sync); kept
only because `operator_status` and the CPU test suite still load it.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import select
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.operator_tools import serving_collect as collector  # noqa: E402


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate(plan):
    entries = plan["operators"]
    expected = {
        o["id"]
        for o in json.loads((ROOT / "DELIVERY_INDEX.json").read_text())["operators"]
    }
    names = [entry["id"] for entry in entries]
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("Plan must account for every indexed operator exactly once")
    for entry in entries:
        if ("config" in entry) == ("reason" in entry):
            raise ValueError(
                "Each operator needs either config or an explicit exclusion reason"
            )
        if "reason" in entry and not entry["reason"].strip():
            raise ValueError("Empty exclusion reason")
    if not 0 < plan["timeout_seconds"] <= 86400:
        raise ValueError("Invalid per-operator timeout")


def timeout_handler(signum, frame):
    raise TimeoutError("Operator collection exceeded its time budget")


def interrupt_handler(signum, frame):
    raise KeyboardInterrupt("Long run interrupted")


def wait_previous(pid, timeout):
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        if not select.select([fd], [], [], timeout)[0]:
            raise TimeoutError(
                "Previous collector did not finish; refusing concurrent NPU use"
            )
    finally:
        os.close(fd)


def execute(plan, output):
    results = []
    for entry in plan["operators"]:
        record = {"id": entry["id"], "started": time.time(), "status": "running"}
        results.append(record)
        write(output / "progress.json", results)
        if "reason" in entry:
            record.update(status="not_run", reason=entry["reason"])
        else:
            directory = output / entry["id"]
            directory.mkdir()
            config = entry["config"]
            collector.save(directory / "config.json", config)
            signal.alarm(plan["timeout_seconds"])
            try:
                collector.run(config, directory)
                record["status"] = (
                    "numerical_only_attribution_pending"
                    if config.get("guard_only", False)
                    else "collected_pending_review"
                )
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                collector.save(directory / "failure.json", record)
            finally:
                signal.alarm(0)
            record["artifacts"] = {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.glob("*.json")
            }
        record["finished"] = time.time()
        write(output / "progress.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    validate(plan)
    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Raw output must be outside the repository")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    write(output / "plan.json", plan)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.signal(signal.SIGTERM, interrupt_handler)
    outcome = {"status": "running", "pid": os.getpid(), "started": time.time()}
    try:
        write(
            output / "owner.json", {"pid": os.getpid(), "started": outcome["started"]}
        )
        outcome["status"] = "waiting" if args.wait_pid else "running"
        write(output / "status.json", outcome)
        if args.wait_pid:
            wait_previous(args.wait_pid, 86400)
        with (ROOT / ".openbitfun/serving-longrun.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            outcome["status"] = "running"
            write(output / "status.json", outcome)
            results = execute(plan, output)
            outcome.update(
                status="finished_with_failures"
                if any(r["status"] == "failed" for r in results)
                else "finished_pending_review",
                results=results,
            )
    except BaseException as exc:
        outcome["status"] = "failed"
        outcome["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        outcome["finished"] = time.time()
        write(output / "completion.json", outcome)
        write(output / "status.json", outcome)


if __name__ == "__main__":
    main()
