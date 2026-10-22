from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import suppress
from pathlib import Path

RECORDS = Path("/root/包体恢复/records/ladder")
PIDFILE = Path("/tmp/ladder-serve.pid")
MODEFILE = Path("/tmp/ladder-serve-mode.json")


def drift(first, second):
    return 100 * (second / first - 1)


def classify(profile, plain):
    if any(abs(drift(*values)) > 0.5 for values in (profile, plain)):
        return "inconclusive"
    delta = abs(drift(plain[1], profile[1]))
    return (
        "supports-reuse"
        if delta <= 0.5
        else "rejects-reuse"
        if delta >= 1
        else "inconclusive"
    )


def identity(pid):
    text = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "state": text[0],
        "ppid": int(text[1]),
        "pgid": int(text[2]),
        "start_ticks": int(text[19]),
    }


def descendant(pid, ancestor):
    for _ in range(32):
        if pid == ancestor:
            return True
        if pid <= 1:
            break
        pid = identity(pid)["ppid"]
    return False


def dump(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


class Probe:
    def __init__(self, output):
        self.output = output
        self.deadline = time.monotonic() + 40 * 60 - 60
        self.owner = None
        self.active = None
        self.env = dict(
            os.environ,
            ASCEND_RT_VISIBLE_DEVICES="0,1",
            VLLM_HUST_GATE_FP32="1",
            VLLM_HUST_GDN_MERGE_INPROJ="1",
            LADDER_L2_LOAD="synthetic",
        )
        self.env.pop("LADDER_SERVE_EXTRA", None)
        self.env.pop("LADDER_MODEL", None)

    def command(self, name, args, startup=False):
        remaining = min(900, self.deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("NPU budget exhausted")
        with (self.output / f"{name}.log").open("x") as log:
            proc = subprocess.Popen(
                ["opt", "bench", *args],
                env=self.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.active = proc
            until = time.monotonic() + remaining
            try:
                while proc.poll() is None:
                    if startup and self.owner is None and PIDFILE.exists():
                        pid = int(PIDFILE.read_text().strip())
                        info = identity(pid)
                        if not descendant(pid, proc.pid):
                            raise RuntimeError("Startup PID is not our descendant")
                        if info["pgid"] == pid:
                            self.owner = info
                            dump(self.output / f"{name}-owner.json", info)
                    if time.monotonic() >= until:
                        raise TimeoutError(name)
                    time.sleep(0.2)
                dump(self.output / f"{name}.exit.json", {"exit": proc.returncode})
                if proc.returncode:
                    raise RuntimeError(f"{name} exited {proc.returncode}")
                if startup:
                    if not self.owner or PIDFILE.read_text().strip() != str(
                        self.owner["pid"]
                    ):
                        raise RuntimeError("Startup ownership missing or replaced")
                    current = identity(self.owner["pid"])
                    if current["start_ticks"] != self.owner["start_ticks"] or current[
                        "state"
                    ] in ("Z", "X"):
                        raise RuntimeError("Startup PID identity lost")
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                self.active = None

    def stop(self, arm):
        if not self.owner:
            return
        owner = self.owner
        try:
            current = identity(owner["pid"])
        except FileNotFoundError:
            current = None
        if current is not None:
            if (
                current["start_ticks"] != owner["start_ticks"]
                or current["pgid"] != owner["pid"]
            ):
                raise RuntimeError("Refusing cleanup of a replaced process")
            os.killpg(owner["pid"], signal.SIGTERM)
            time.sleep(15)
            with suppress(ProcessLookupError):
                os.killpg(owner["pid"], signal.SIGKILL)
        if PIDFILE.exists() and PIDFILE.read_text().strip() == str(owner["pid"]):
            PIDFILE.unlink()
            MODEFILE.unlink(missing_ok=True)
        dump(
            self.output / f"{arm}-cleanup.json",
            {"owned_identity": owner, "stopped": True},
        )
        self.owner = None

    def artifact(self, name, step):
        before = set(RECORDS.glob(f"*-{step}.json"))
        self.command(name, [step])
        added = set(RECORDS.glob(f"*-{step}.json")) - before
        if len(added) != 1:
            raise RuntimeError(f"Ambiguous {name} artifacts: {added}")
        path = added.pop()
        dump(self.output / f"{name}-artifact.json", {"path": str(path)})
        return path

    def run(self):
        if PIDFILE.exists():
            raise RuntimeError("Existing PID file; refusing to take ownership")
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", 8000)) == 0:
                raise RuntimeError("Port already in use")
        artifacts, values = {}, {}
        arm = "profile"
        try:
            for arm in ("profile", "plain"):
                args = (
                    ["serve", "--profile", "light"] if arm == "profile" else ["serve"]
                )
                self.command(f"{arm}-serve", args, startup=True)
                log = Path("/tmp/ladder-serve-log").read_text().strip()
                self.command(f"{arm}-carriers", ["carriers", log])
                paths = [self.artifact(f"{arm}-l2-{i}", "l2") for i in (1, 2)]
                reports = [json.loads(path.read_text()) for path in paths]
                if any(
                    doc.get("completed") != 128 or doc.get("failed") != 0
                    for doc in reports
                ):
                    raise RuntimeError("Incomplete L2 requests")
                values[arm] = [doc["output_throughput"] for doc in reports]
                artifacts[arm] = {
                    "serve_log": log,
                    "l2": [str(p) for p in paths],
                    "l3": str(self.artifact(f"{arm}-l3", "l3")),
                }
                self.stop(arm)
                if abs(drift(*values[arm])) > 0.5:
                    dump(
                        self.output / "collection-result.json",
                        {
                            "status": "inconclusive",
                            "reason": f"{arm} L2 outside declared convergence band",
                            "values": values,
                            "artifacts": artifacts,
                        },
                    )
                    return
            self.command(
                "l3-compare",
                ["compare", artifacts["profile"]["l3"], artifacts["plain"]["l3"]],
            )
            dump(
                self.output / "collection-result.json",
                {
                    "status": classify(values["profile"], values["plain"]),
                    "values": values,
                    "artifacts": artifacts,
                    "profile_vs_plain_second_pct": drift(
                        values["plain"][1], values["profile"][1]
                    ),
                    "boundary": "Collection only; pairing facts await review",
                },
            )
        finally:
            self.stop(arm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe = Probe(args.output)
    try:
        probe.run()
    except Exception as error:
        dump(args.output / "collection-failure.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
