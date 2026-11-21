"""Resumable nohup runner for single-operator and multi-gate serving E2E runs.

Archive-only by default. Repository writes require explicit sync_mode="auto";
"none" at either item or config level vetoes all writes.

Supports stage-level checkpoint/resume across server restarts:
  - Reuses completed stages (`guard_off`, `guard_on`, `bench_0_off`, `bench_1_on`,
    `bench_2_on`, `bench_3_off`) when their exit/artifact checks pass; deletes and
    reruns only interrupted or incomplete stages.
  - Supports both single-operator runs (`gate` string, synced via `serving_sync.sync`
    into `ops/<op>/evidence.json` + `DELIVERY_INDEX.json` + `custody.py`) and
    multi-gate combination runs (`gates: {"off": {...}, "on": {...}}`, optionally
    written into `docs/mod-acceptance/evidence.json` `combination_plan`).
  - Acquires `.openbitfun/serving-longrun.lock` so two collectors never contend for
    the same NPUs/ports.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import (  # noqa: E402
    custody,
    serving_attribution,
    serving_sync,  # noqa: E402
)
from tools.operator_tools import serving_collect as collector  # noqa: E402
from tools.operator_tools.serving_loop import (  # noqa: E402
    FIXED_L3_PROMPTS,
    compare_greedy_outputs,
    parse_op_statistic_csv,
    summarize_serving_ab_runs,
)
from tools.operator_tools.workloads import (  # noqa: E402
    WORKLOAD_CATALOG,
    build_bench_serve_cmd,
    select_workloads_for_operator,
)

DOTDIR = ROOT / ".openbitfun"
DEFAULT_CHECKPOINT = DOTDIR / "e2e-run-state.json"
DEFAULT_LOGFILE = DOTDIR / "e2e-run.log"
DEFAULT_STATUSFILE = DOTDIR / "e2e-run-status.txt"
LOCKFILE = DOTDIR / "serving-longrun.lock"

BENCH_ORDER: tuple[tuple[int, str], ...] = (
    (0, "off"),
    (1, "on"),
    (2, "on"),
    (3, "off"),
)


def atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class JsonLineHandler(logging.Handler):
    def __init__(self, logfile: Path) -> None:
        super().__init__()
        self.logfile = logfile

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "msg": self.format(record),
        }
        with suppress(OSError):
            self.logfile.parent.mkdir(parents=True, exist_ok=True)
            with self.logfile.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def setup_logger(logfile: Path) -> logging.Logger:
    logger = logging.getLogger("serving_e2e_auto")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(stream)
    logger.addHandler(JsonLineHandler(logfile))
    return logger


def write_status_file(status_path: Path, state: dict[str, Any]) -> None:
    lines = [
        f"pid: {os.getpid()}",
        f"updated: {utc_now()}",
        f"output_root: {state.get('output_root', '')}",
        f"overall_status: {state.get('status', 'running')}",
        "",
    ]
    for item_id, rec in state.get("items", {}).items():
        stage = rec.get("current_stage") or rec.get("last_stage") or "-"
        sync_state = rec.get("evidence_sync", "-")
        extra = f" | error={rec['error']}" if rec.get("error") else ""
        lines.append(
            f"  {item_id}: {rec.get('status', '?')} | "
            f"stage={stage} | sync={sync_state}{extra}"
        )
    with suppress(OSError):
        status_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = status_path.with_suffix(status_path.suffix + ".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(status_path)


def save_state(checkpoint_path: Path, status_path: Path, state: dict[str, Any]) -> None:
    state["updated_ts"] = utc_now()
    atomic_write(checkpoint_path, state)
    write_status_file(status_path, state)


def evidence_sync_enabled(entry: dict[str, Any]) -> bool:
    # Early plans stored sync_mode on the item; neither spelling may override a veto.
    modes = [obj["sync_mode"] for obj in (entry, entry["config"]) if "sync_mode" in obj]
    if any(mode not in ("auto", "none") for mode in modes):
        raise ValueError(f"{entry['id']}: invalid sync_mode {modes!r}")
    return "auto" in modes and "none" not in modes


def validate_plan(plan: dict[str, Any]) -> None:
    items = plan.get("items") or plan.get("operators")
    if not isinstance(items, list) or not items:
        raise ValueError("plan must define a non-empty 'items' or 'operators' list")
    timeout = plan.get("timeout_seconds", 14400)
    if not isinstance(timeout, (int, float)) or not (0 < timeout <= 86400):
        raise ValueError("timeout_seconds must be in (0, 86400]")
    seen: set[str] = set()
    for entry in items:
        item_id = entry.get("id")
        if not item_id or item_id in seen:
            raise ValueError(f"Invalid or duplicate item id: {item_id!r}")
        seen.add(item_id)
        if ("config" in entry) == ("reason" in entry):
            raise ValueError(
                f"Item {item_id!r}: specify exactly one of 'config' or 'reason'"
            )
        if "reason" in entry and not str(entry["reason"]).strip():
            raise ValueError(f"Item {item_id!r}: empty skip reason")
        if "config" in entry:
            evidence_sync_enabled(entry)
            cfg = entry["config"]
            has_single = isinstance(cfg.get("gate"), str) and bool(cfg["gate"])
            has_multi = isinstance(cfg.get("gates"), dict) and set(cfg["gates"]) == {
                "off",
                "on",
            }
            if has_single and [
                r["operator"] for r in collector.service_routes(cfg)
            ] != [item_id]:
                raise ValueError(f"Item {item_id!r}: gate belongs to another operator")
            if has_single == has_multi:
                raise ValueError(
                    f"Item {item_id!r}: config needs either 'gate' or "
                    "'gates'={'off','on'}"
                )


def clean_exit(stage_dir: Path) -> bool:
    exit_path = stage_dir / "exit.json"
    if not exit_path.is_file():
        return False
    try:
        info = read_json(exit_path)
    except Exception:
        return False
    if "collection_completed" in info:
        return (
            info.get("collection_completed") is True
            and info.get("returncode_before_shutdown") is None
            and info.get("shutdown_requested") is True
        )
    return info.get("returncode") == 0


def guard_stage_complete(config: dict[str, Any], stage_dir: Path, arm: str) -> bool:
    if not stage_dir.is_dir() or not clean_exit(stage_dir):
        return False
    try:
        inv = read_json(stage_dir / "invocation.json")
        if config.get("gate"):
            if inv.get("gate") != ("1" if arm == "on" else "0"):
                return False
        else:
            expected_gates = {k: str(v) for k, v in config["gates"][arm].items()}
            if inv.get("gates") != expected_gates:
                return False
        responses = read_json(stage_dir / "responses.json")
        greedy = read_json(stage_dir / "greedy.json")
        if len(responses) != len(FIXED_L3_PROMPTS):
            return False
        if not all(
            r.get("usage", {}).get("completion_tokens") == 128
            and len(r.get("choices", [])) == 1
            for r in responses
        ):
            return False
        if [r["choices"][0]["text"] for r in responses] != greedy:
            return False
        world_size = int(config["world_size"])
        op_csvs = list((stage_dir / "profile").rglob("op_statistic.csv"))
        if len(op_csvs) != world_size:
            return False
        if config.get("attribution") or config.get("attributions"):
            kd_csvs = list((stage_dir / "profile").rglob("kernel_details.csv"))
            if len(kd_csvs) != world_size:
                return False
    except Exception:
        return False
    return True


def bench_stage_complete(
    config: dict[str, Any],
    stage_dir: Path,
    arm: str,
    pairing: dict[str, Any],
) -> bool:
    if not stage_dir.is_dir() or not clean_exit(stage_dir):
        return False
    try:
        inv = read_json(stage_dir / "invocation.json")
        if config.get("gate"):
            if inv.get("gate") != ("1" if arm == "on" else "0"):
                return False
        else:
            expected_gates = {k: str(v) for k, v in config["gates"][arm].items()}
            if inv.get("gates") != expected_gates:
                return False
        for workload, profile in WORKLOAD_CATALOG.items():
            target = stage_dir / f"{workload}.json"
            if not target.is_file():
                return False
            collector.validate_benchmark(target, profile.default_num_prompts)
            winv = read_json(stage_dir / f"{workload}_invocation.json")
            cmd = build_bench_serve_cmd(
                workload,
                config["port"],
                config["model_alias"],
                str(stage_dir),
                target.name,
                concurrency=pairing["primary_concurrency"]
                if workload == pairing["primary_workload"]
                else profile.default_concurrency,
                dataset_path=config["sharegpt_path"]
                if workload == "sharegpt"
                else None,
            )
            expected_cmd = [
                config["python"],
                "-m",
                "vllm.entrypoints.cli.main",
                *cmd[1:],
                "--tokenizer",
                config["model_path"],
                "--seed",
                "42",
            ]
            if winv != {"command": expected_cmd, "returncode": 0}:
                return False
    except Exception:
        return False
    return True


@contextmanager
def multi_gate_server(
    config: dict[str, Any], output: Path, arm: str, profile: bool = False
):
    output.mkdir(parents=True, exist_ok=False)
    port = config["port"]
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} already occupied")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for key in list(env):
        if key.startswith("VLLM_HUST_"):
            env.pop(key)
    env.update(config.get("environment", {}))
    for fixed_off_key in config.get("fixed_off", []):
        env[fixed_off_key] = "0"
    arm_gates = {k: str(v) for k, v in config["gates"][arm].items()}
    env.update(arm_gates)
    cmd = list(config["server_command"])
    if profile:
        cmd += config.get("guard_server_args", [])
        cmd += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(output / "profile"),
                    "torch_profiler_with_stack": False,
                    "torch_profiler_record_shapes": bool(
                        config.get("attribution") or config.get("attributions")
                    ),
                    "ignore_frontend": True,
                }
            ),
        ]
    collector.save(
        output / "invocation.json",
        {
            "command": cmd,
            "arm": arm,
            "gates": arm_gates,
            "fixed_off": list(config.get("fixed_off", [])),
        },
    )
    with (output / "server.log").open("w") as srv_log:
        proc = subprocess.Popen(
            cmd,
            cwd=output,
            env=env,
            stdout=srv_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        collector.save(output / "owner.json", {"pid": proc.pid})
        collection_completed = False
        try:
            deadline = time.monotonic() + config.get("startup_timeout", 1200)
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited {proc.returncode}: {output}")
                try:
                    collector.request(port, "/health")
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Server startup: {output}") from None
                    time.sleep(2)
            yield env
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server exited before collection finished: {output}"
                )
            collection_completed = True
        finally:
            before_shutdown = proc.poll()
            forced_kill = False
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                forced_kill = True
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            collector.save(
                output / "exit.json",
                {
                    "returncode": proc.returncode,
                    "returncode_before_shutdown": before_shutdown,
                    "shutdown_requested": before_shutdown is None,
                    "forced_kill": forced_kill,
                    "collection_completed": collection_completed,
                },
            )


def server_ctx(config: dict[str, Any], output: Path, arm: str, profile: bool = False):
    if config.get("gate"):
        return collector.server(config, output, arm, profile=profile)
    return multi_gate_server(config, output, arm, profile=profile)


def run_guard_stage(config: dict[str, Any], stage_dir: Path, arm: str) -> list[str]:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    with server_ctx(config, stage_dir, arm, profile=True):
        for _ in range(10):
            collector.completion(config, "Explain RMSNorm briefly.", 16)
        responses = [collector.completion(config, p, 128) for p in FIXED_L3_PROMPTS]
        collector.save(stage_dir / "responses.json", responses)
        texts = [r["choices"][0]["text"] for r in responses]
        collector.save(stage_dir / "greedy.json", texts)
        collector.request(config["port"], "/start_profile", {})
        try:
            response = collector.completion(config, "Explain a binary heap.", 16)
            collector.save(stage_dir / "profile_response.json", response)
        finally:
            collector.request(config["port"], "/stop_profile", {})
    collector.analyse_profile(config, stage_dir)
    return texts


def verify_multi_attribution(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Verify per-candidate attribution in a combined multi-gate guard run."""
    specs = config.get("attributions", {})
    world_size = int(config["world_size"])
    kd_arms: dict[str, dict[str, list[dict[str, str]]]] = {}
    op_arms: dict[str, dict[str, Path]] = {}
    sources: list[dict[str, str]] = []

    for arm in ("off", "on"):
        kd_devices: dict[str, list[dict[str, str]]] = {}
        for path in sorted(
            (output / f"guard_{arm}" / "profile").rglob("kernel_details.csv")
        ):
            with path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            ids = {row["Device_id"] for row in rows}
            if len(ids) != 1:
                raise ValueError(f"Expected 1 device in {path}, got {ids}")
            dev = ids.pop()
            kd_devices[dev] = rows
            sources.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        op_devices: dict[str, Path] = {}
        for path in sorted(
            (output / f"guard_{arm}" / "profile").rglob("op_statistic.csv")
        ):
            with path.open(newline="") as fh:
                ids = {row["Device_id"] for row in csv.DictReader(fh)}
            if len(ids) != 1:
                raise ValueError(f"Expected 1 device in {path}, got {ids}")
            dev = ids.pop()
            op_devices[dev] = path
            sources.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        if len(kd_devices) != world_size or len(op_devices) != world_size:
            raise ValueError("Incomplete per-device profiler CSVs")
        kd_arms[arm] = kd_devices
        op_arms[arm] = op_devices

    if kd_arms["off"].keys() != kd_arms["on"].keys():
        raise ValueError("OFF/ON device sets differ")

    per_candidate: dict[str, Any] = {}
    for cand_id, rule in specs.items():
        mode = rule["mode"]
        if mode == "dedicated":
            dev_reports = {
                dev: serving_attribution.evaluate(
                    cand_id, kd_arms["off"][dev], kd_arms["on"][dev]
                )
                for dev in sorted(kd_arms["off"])
            }
            per_candidate[cand_id] = {
                "mode": "dedicated",
                "passed": all(r["passed"] for r in dev_reports.values()),
                "devices": dev_reports,
            }
        elif mode == "kernel_count":
            kernel = rule["kernel"]
            expected_on = int(rule["expected_per_rank_on"])
            dev_reports = {}
            for dev in sorted(op_arms["off"]):
                off_stat = parse_op_statistic_csv(str(op_arms["off"][dev]), kernel)
                on_stat = parse_op_statistic_csv(str(op_arms["on"][dev]), kernel)
                ok = (
                    off_stat["call_count_actual"] == 0
                    and on_stat["call_count_actual"] == expected_on
                )
                dev_reports[dev] = {
                    "passed": ok,
                    "expected_on": expected_on,
                    "off_actual": off_stat["call_count_actual"],
                    "on_actual": on_stat["call_count_actual"],
                }
            per_candidate[cand_id] = {
                "mode": "kernel_count",
                "kernel": kernel,
                "passed": all(r["passed"] for r in dev_reports.values()),
                "devices": dev_reports,
            }
        else:
            raise ValueError(f"Unsupported attribution rule mode: {mode}")

    overall = all(c["passed"] for c in per_candidate.values())
    result = {
        "passed": overall,
        "protocol": "single-request-16-tokens",
        "candidates": per_candidate,
        "sources": sources,
    }
    collector.save(output / "l4.json", result)
    if not overall:
        failed = [k for k, v in per_candidate.items() if not v["passed"]]
        raise ValueError(f"Combined L4 attribution failed for: {failed}")
    return result


def run_bench_stage(
    config: dict[str, Any],
    stage_dir: Path,
    arm: str,
    pairing: dict[str, Any],
) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    with server_ctx(config, stage_dir, arm, profile=False) as env:
        for _ in range(10):
            collector.completion(config, "Explain RMSNorm briefly.", 16)
        for workload, profile in WORKLOAD_CATALOG.items():
            target = stage_dir / f"{workload}.json"
            command = build_bench_serve_cmd(
                workload,
                config["port"],
                config["model_alias"],
                str(stage_dir),
                target.name,
                concurrency=pairing["primary_concurrency"]
                if workload == pairing["primary_workload"]
                else profile.default_concurrency,
                dataset_path=config["sharegpt_path"]
                if workload == "sharegpt"
                else None,
            )
            cmd = [
                config["python"],
                "-m",
                "vllm.entrypoints.cli.main",
                *command[1:],
                "--tokenizer",
                config["model_path"],
                "--seed",
                "42",
            ]
            with (stage_dir / f"{workload}.log").open("w") as wlog:
                subprocess.run(
                    cmd,
                    cwd=stage_dir,
                    env=env,
                    stdout=wlog,
                    stderr=subprocess.STDOUT,
                    timeout=2400,
                    check=True,
                )
            collector.save(
                stage_dir / f"{workload}_invocation.json",
                {"command": cmd, "returncode": 0},
            )
            collector.validate_benchmark(target, profile.default_num_prompts)


def collect_item_resumable(
    item_id: str,
    config: dict[str, Any],
    item_dir: Path,
    item_state: dict[str, Any],
    flush_cb,
    logger: logging.Logger,
) -> None:
    item_dir.mkdir(parents=True, exist_ok=True)
    if not Path(config["sharegpt_path"]).is_file():
        raise FileNotFoundError(config["sharegpt_path"])

    pairing = select_workloads_for_operator(
        config.get("phase", "decode"), config.get("max_active_rows")
    )
    reuse = any(item_dir.glob("guard_*")) or any(item_dir.glob("bench_*"))
    if reuse and read_json(item_dir / "config.json") != config:
        raise ValueError("Collection config changed; use a new archive")
    collector.verify_host(config, item_dir, reuse=reuse)
    collector.save(item_dir / "config.json", config)
    with suppress(FileNotFoundError):
        (item_dir / "failure.json").unlink()

    texts: dict[str, list[str]] = {}
    for arm in ("off", "on"):
        stage_name = f"guard_{arm}"
        stage_dir = item_dir / stage_name
        item_state["current_stage"] = stage_name
        flush_cb()
        if guard_stage_complete(config, stage_dir, arm):
            logger.info("[%s] Reusing completed %s", item_id, stage_name)
            texts[arm] = read_json(stage_dir / "greedy.json")
        else:
            logger.info("[%s] Running %s", item_id, stage_name)
            texts[arm] = run_guard_stage(config, stage_dir, arm)
        item_state["last_stage"] = stage_name
        flush_cb()

    guard = compare_greedy_outputs(texts["off"], texts["on"])
    collector.save(item_dir / "serving_l3.json", guard)
    if not guard["passed"]:
        raise ValueError(f"Serving L3 failed for {item_id}: {guard}")

    if config.get("guard_only", False):
        item_state["current_stage"] = None
        return

    if config.get("gate"):
        collector.verify_profiles(config, item_dir)
    elif config.get("attributions"):
        verify_multi_attribution(config, item_dir)

    collector.save(item_dir / "workload_pairing.json", pairing)

    if not config.get("guard_and_l4_only", False):
        paths: dict[str, dict[str, list[str]]] = {
            w: {"off": [], "on": []} for w in WORKLOAD_CATALOG
        }
        for idx, arm in BENCH_ORDER:
            stage_name = f"bench_{idx}_{arm}"
            stage_dir = item_dir / stage_name
            item_state["current_stage"] = stage_name
            flush_cb()
            if bench_stage_complete(config, stage_dir, arm, pairing):
                logger.info("[%s] Reusing completed %s", item_id, stage_name)
            else:
                logger.info("[%s] Running %s", item_id, stage_name)
                run_bench_stage(config, stage_dir, arm, pairing)
            for workload in WORKLOAD_CATALOG:
                paths[workload][arm].append(str(stage_dir / f"{workload}.json"))
            item_state["last_stage"] = stage_name
            flush_cb()

        summaries = {
            w: summarize_serving_ab_runs(p["off"], p["on"], w) for w, p in paths.items()
        }
        collector.save(item_dir / "summary.json", summaries)

    item_state["current_stage"] = None
    flush_cb()


def write_combination_evidence(
    item_id: str,
    config: dict[str, Any],
    item_dir: Path,
    target_rel: str,
    logger: logging.Logger,
) -> str:
    mod_path = ROOT / target_rel
    data = read_json(mod_path)
    plan_block = data["assessment"]["combination_plan"]

    l3 = read_json(item_dir / "serving_l3.json")
    l4 = read_json(item_dir / "l4.json") if (item_dir / "l4.json").is_file() else None
    summary = (
        read_json(item_dir / "summary.json")
        if (item_dir / "summary.json").is_file()
        else None
    )

    files_hashed: dict[str, str] = {}
    for rel in ("config.json", "host_preflight.json", "serving_l3.json"):
        p = item_dir / rel
        if p.is_file():
            files_hashed[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    for optional_rel in ("l4.json", "workload_pairing.json", "summary.json"):
        p = item_dir / optional_rel
        if p.is_file():
            files_hashed[optional_rel] = hashlib.sha256(p.read_bytes()).hexdigest()

    entry_ref = {"archive": item_dir.resolve().as_uri(), "files": files_hashed}
    retained = [
        r
        for r in plan_block.get("retained_experiments", [])
        if r.get("archive") != entry_ref["archive"]
    ]
    retained.append(entry_ref)
    plan_block["retained_experiments"] = retained

    runs_map = plan_block.setdefault("runs", {})
    run_record: dict[str, Any] = {
        "archive": item_dir.resolve().as_uri(),
        "arms": copy.deepcopy(config["gates"]),
        "fixed_off": list(config.get("fixed_off", [])),
        "host_commits": copy.deepcopy(config["host_commits"]),
        "numerical_guard": l3,
    }
    if l4 is not None:
        run_record["attribution"] = {
            "passed": l4["passed"],
            "protocol": l4.get("protocol"),
            "candidates": {
                k: {"mode": v["mode"], "passed": v["passed"]}
                for k, v in l4.get("candidates", {}).items()
            },
        }
    if summary is not None:
        primary_wid = plan_block.get("primary_workload", "random")
        primary = copy.deepcopy(summary[primary_wid])
        if primary["status"] == "negative_speedup":
            primary["status"] = "measured_no_gain"
        run_record["serving_ab"] = {
            "tested": True,
            "primary_workload": primary_wid,
            "status": primary["status"],
            "speedup": primary["speedup"],
            "gen_rate_off": primary["gen_rate_off"],
            "gen_rate_on": primary["gen_rate_on"],
            "intra_group_spread_pct": primary["intra_group_spread_pct"],
            "mean_step_off_ms": primary["mean_step_off_ms"],
            "mean_step_on_ms": primary["mean_step_on_ms"],
            "workloads": summary,
        }
        if config.get("promote_combination_primary", False):
            plan_block["tested"] = True
            plan_block["status"] = primary["status"]
            plan_block["primary_result"] = run_record["serving_ab"]

    runs_map[item_id] = run_record
    atomic_write(mod_path, data)
    logger.info("[%s] Updated combination_plan in %s", item_id, target_rel)
    return "combination_recorded"


def sync_item_evidence(
    entry: dict[str, Any], item_dir: Path, logger: logging.Logger
) -> str:
    item_id = entry["id"]
    config = entry["config"]
    if not evidence_sync_enabled(entry):
        return "not_applicable"
    if config.get("guard_only") or config.get("guard_and_l4_only"):
        if entry.get("combination_evidence_target"):
            return write_combination_evidence(
                item_id, config, item_dir, entry["combination_evidence_target"], logger
            )
        return "not_applicable"

    if config.get("gate"):
        res = serving_sync.sync(item_dir, item_id, root=ROOT)
        ok, reason = custody.verify_dir_checksum(str(ROOT / "ops" / item_id))
        if not ok:
            raise RuntimeError(f"Custody verify failed for {item_id}: {reason}")
        logger.info("[%s] serving_sync result: %s", item_id, res)
        return res

    target_rel = entry.get(
        "combination_evidence_target", "docs/mod-acceptance/evidence.json"
    )
    return write_combination_evidence(item_id, config, item_dir, target_rel, logger)


def run_offline_checks(python_exe: str, logger: logging.Logger) -> None:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    res = subprocess.run(
        [python_exe, "-m", "pytest", "@tests/offline-checks.args"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"offline-checks failed:\n{res.stdout[-2000:]}\n{res.stderr[-500:]}"
        )
    logger.info("offline-checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Path to plan JSON")
    parser.add_argument(
        "--output", required=True, help="Archive directory outside repository root"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint JSON path",
    )
    parser.add_argument(
        "--logfile", type=Path, default=DEFAULT_LOGFILE, help="JSONL log path"
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=DEFAULT_STATUSFILE,
        help="Human-readable status file path",
    )
    parser.add_argument(
        "--offline-python",
        default=str(ROOT / ".openbitfun/mod-acceptance-venv/bin/python"),
        help="Python interpreter with pytest for post-run offline checks",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint / partial stage directories",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record item failure and proceed to remaining items",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = read_json(plan_path)
    validate_plan(plan)

    output_root = Path(args.output).resolve()
    if output_root == ROOT or ROOT in output_root.parents:
        parser.error("--output must be outside the repository root")

    checkpoint_path = args.checkpoint.resolve()
    status_path = args.status_file.resolve()
    logger = setup_logger(args.logfile.resolve())

    DOTDIR.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.is_file():
        state = read_json(checkpoint_path)
        if state.get("items") and state.get("output_root") != str(output_root):
            parser.error("Checkpoint belongs to another output archive")
        if state.get("items") and not args.resume:
            parser.error(
                f"Checkpoint already exists at {checkpoint_path}; "
                "pass --resume to continue"
            )
    else:
        state = {"items": {}}

    state.update(
        pid=os.getpid(),
        plan_path=str(plan_path),
        output_root=str(output_root),
        status="running",
    )
    state.setdefault("started_ts", utc_now())

    def flush() -> None:
        save_state(checkpoint_path, status_path, state)
        atomic_write(output_root / "e2e-progress.json", state)

    def handle_sigterm(signum: int, frame: object) -> None:
        logger.warning("Received signal %d; flushing checkpoint before exit", signum)
        state["status"] = "interrupted"
        flush()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    items = plan.get("items") or plan["operators"]
    timeout = int(plan.get("timeout_seconds", 14400))
    failed_items: list[str] = []
    any_synced = False

    with LOCKFILE.open("w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archived_plan = output_root / "plan.json"
        if archived_plan.exists() and read_json(archived_plan) != plan:
            raise ValueError("Archive belongs to another plan; use a new archive")
        if state.get("items") and not archived_plan.exists():
            raise ValueError("Checkpoint lacks its archived plan; use a new archive")
        atomic_write(archived_plan, plan)
        atomic_write(
            output_root / "owner.json", {"pid": os.getpid(), "started": utc_now()}
        )
        flush()
        for entry in items:
            item_id = entry["id"]
            rec = state["items"].setdefault(item_id, {})

            if "reason" in entry:
                rec.update(
                    status="skipped",
                    reason=entry["reason"],
                    finished_ts=utc_now(),
                )
                flush()
                continue

            if (
                rec.get("status") == "done"
                and (output_root / item_id / "serving_l3.json").is_file()
            ):
                logger.info("[%s] Already marked done in checkpoint; skipping", item_id)
                continue

            item_dir = output_root / item_id
            rec.update(
                status="running",
                output_dir=str(item_dir),
                started_ts=rec.get("started_ts") or utc_now(),
                error=None,
            )
            flush()

            def alarm_handler(signum: int, frame: object, _id: str = item_id) -> None:
                raise TimeoutError(f"Item {_id!r} exceeded timeout {timeout}s")

            prev_alarm = signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(timeout)
            try:
                collect_item_resumable(
                    item_id, entry["config"], item_dir, rec, flush, logger
                )
                sync_res = sync_item_evidence(entry, item_dir, logger)
                if sync_res in {"synced", "already_synced", "combination_recorded"}:
                    any_synced = True
                rec.update(
                    status="done",
                    evidence_sync=sync_res,
                    finished_ts=utc_now(),
                )
                flush()
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                logger.error("[%s] Failed: %s", item_id, msg)
                rec.update(
                    status="failed",
                    error=msg,
                    finished_ts=utc_now(),
                )
                with contextlib.suppress(OSError):
                    collector.save(
                        item_dir / "failure.json",
                        {"id": item_id, "error": msg, "ts": utc_now()},
                    )
                flush()
                failed_items.append(item_id)
                if not args.continue_on_error:
                    break
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_alarm)

        if any_synced and Path(args.offline_python).is_file():
            try:
                run_offline_checks(args.offline_python, logger)
                state["offline_checks"] = "passed"
            except Exception as exc:
                logger.error("Post-run offline checks failed: %s", exc)
                state["offline_checks"] = f"failed: {exc}"
                failed_items.append("offline_checks")

        state["status"] = "failed" if failed_items else "complete"
        state["failed_items"] = failed_items
        state["finished_ts"] = utc_now()
        flush()
        atomic_write(output_root / "completion.json", state)

    if failed_items:
        sys.exit(1)


if __name__ == "__main__":
    main()
