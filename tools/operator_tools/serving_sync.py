"""Promote complete serving collections; never launch a server or modify raw data."""

from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import json
import re
from pathlib import Path

from tools.operator_tools import custody, serving_attribution, serving_collect
from tools.operator_tools.serving_loop import (
    FIXED_L3_PROMPTS,
    compare_greedy_outputs,
    parse_op_statistic_csv,
    summarize_serving_ab_runs,
)
from tools.operator_tools.workloads import (
    WORKLOAD_CATALOG,
    build_bench_serve_cmd,
    select_workloads_for_operator,
)

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".sync-tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def validate(directory, entry):
    files = {}

    def read(relative):
        path = directory / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load(path)

    require(not (directory / "failure.json").exists(), "Collection failed")
    config = read("config.json")
    require(entry["gate_env"] == config["gate"] + "=1", "Gate mismatch")
    require(entry["default_enabled"] is False, "Expected default-off")
    require(entry["layer"] != "rejected_exploration", "Rejected operator")
    require(not config.get("guard_only"), "Guard-only collection")
    hosts = config.get("host_commits", {})
    require(set(hosts) == {"vllm", "vllm-ascend"}, "Missing host commits")
    require(
        all(re.fullmatch(r"[a-f0-9]{40}", c) for c in hosts.values()),
        "Invalid host commits",
    )
    preflight = read("host_preflight.json")
    require(preflight["returncode"] == 0, "Host preflight failed")
    observed = json.loads(preflight["stdout"])
    require(
        all(observed[n]["revision"] == c for n, c in hosts.items()), "Host mismatch"
    )
    require(
        preflight["argv"][0] == config["python"] == config["server_command"][0],
        "Interpreter mismatch",
    )
    texts = {}
    for stage, arm in [("guard_off", "off"), ("guard_on", "on")] + [
        (f"bench_{i}_{arm}", arm) for i, arm in enumerate(("off", "on", "on", "off"))
    ]:
        invocation = read(f"{stage}/invocation.json")
        require(invocation["gate"] == ("1" if arm == "on" else "0"), "Arm mismatch")
        command = invocation["command"]
        require(
            command[: len(config["server_command"])] == config["server_command"],
            "Server command mismatch",
        )
        exit_info = read(f"{stage}/exit.json")
        if "collection_completed" in exit_info:
            require(
                exit_info["collection_completed"] is True
                and exit_info["returncode_before_shutdown"] is None
                and exit_info["shutdown_requested"] is True,
                "Incomplete server stage",
            )
        else:
            require(exit_info.get("returncode") == 0, "Legacy server exit is not clean")
        if stage.startswith("guard"):
            responses = read(f"{stage}/responses.json")
            require(len(responses) == len(FIXED_L3_PROMPTS), "Incomplete fixed prompts")
            require(
                all(
                    r["usage"]["completion_tokens"] == 128 and len(r["choices"]) == 1
                    for r in responses
                ),
                "Incomplete response",
            )
            texts[arm] = [r["choices"][0]["text"] for r in responses]
            require(
                texts[arm] == read(f"{stage}/greedy.json"), "Greedy artifact mismatch"
            )
    l3 = compare_greedy_outputs(texts["off"], texts["on"])
    require(
        l3["passed"] is True and l3 == read("serving_l3.json"), "Numerical guard failed"
    )
    l4 = read("l4.json")
    if config.get("attribution"):
        require(config["attribution"] == entry["id"], "Attribution operator mismatch")
        actual = serving_attribution.verify(config, directory)
        require(
            actual == l4 and actual["passed"] is True, "Dedicated attribution failed"
        )
        for source in actual["sources"]:
            files[str(Path(source["path"]).relative_to(directory))] = source["sha256"]
    else:
        audited = {
            "gdn_output_norm_fusion": ("_output_norm_kernel", 30),
            "moe_epilogue": ("_gate_mul_kernel", 40),
        }
        require(
            (config.get("kernel"), config.get("attention_layers"))
            == audited.get(entry["id"]),
            "Unaudited kernel rule",
        )
        require(config["world_size"] == 2, "Unaudited device count")
        expected = config["attention_layers"] * 16
        device_sets = []
        for arm in ("off", "on"):
            devices = set()
            paths = sorted(
                (directory / f"guard_{arm}/profile").rglob("op_statistic.csv")
            )
            require(len(paths) == config["world_size"], "Missing rank profile")
            for path in paths:
                with path.open(newline="") as handle:
                    ids = {row["Device_id"] for row in csv.DictReader(handle)}
                require(len(ids) == 1 and not ids & devices, "Duplicate device profile")
                devices.update(ids)
                files[str(path.relative_to(directory))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                count = parse_op_statistic_csv(str(path), config["kernel"])[
                    "call_count_actual"
                ]
                require(
                    count == (expected if arm == "on" else 0), "Kernel count mismatch"
                )
            device_sets.append(devices)
        require(device_sets[0] == device_sets[1], "OFF/ON devices differ")
        require(l4["passed"] is True, "Attribution failed")
        l4.update(
            call_count_expected=expected * config["world_size"],
            call_count_actual=expected * config["world_size"],
        )
    l4.update(status="completed", verified=True)
    pairing = select_workloads_for_operator(
        config.get("phase", "decode"), config.get("max_active_rows")
    )
    require(read("workload_pairing.json") == pairing, "Workload pairing mismatch")
    summaries = {}
    for workload, profile in WORKLOAD_CATALOG.items():
        paths = {"off": [], "on": []}
        for i, arm in enumerate(("off", "on", "on", "off")):
            stage = directory / f"bench_{i}_{arm}"
            target = stage / f"{workload}.json"
            read(str(target.relative_to(directory)))
            serving_collect.validate_benchmark(target, profile.default_num_prompts)
            invocation = read(f"{stage.name}/{workload}_invocation.json")
            cmd = build_bench_serve_cmd(
                workload,
                config["port"],
                config["model_alias"],
                str(stage),
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
            require(
                invocation == {"command": expected_cmd, "returncode": 0},
                "Benchmark invocation mismatch",
            )
            paths[arm].append(str(target))
        summaries[workload] = summarize_serving_ab_runs(
            paths["off"], paths["on"], workload
        )
    require(read("summary.json") == summaries, "Summary differs from raw results")
    primary = copy.deepcopy(summaries[pairing["primary_workload"]])
    primary.update(
        diagnostic_only=False,
        applicability="current",
        scope="operator serving A/B; not MOD acceptance",
        workloads=summaries,
    )
    if primary["status"] == "negative_speedup":
        primary["status"] = "measured_no_gain"
    return {
        "l3_numerical_guard": l3,
        "l4_attribution": l4,
        "serving_end_to_end": primary,
        "serving_sync": {
            "archive": directory.as_uri(),
            "files": files,
            "host_commits": hosts,
            "completed_ns": (directory / "summary.json").stat().st_mtime_ns,
        },
    }


def sync(directory, operator, root=ROOT):
    directory = directory.resolve()
    root = root.resolve()
    require(not directory.is_relative_to(root), "Raw data must be outside repository")
    lock_dir = root / ".openbitfun"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / "serving-sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        journal = lock_dir / "serving-sync-journal.json"
        require(
            not journal.exists(), "Interrupted sync: restore journal before retrying"
        )
        index_path = root / "DELIVERY_INDEX.json"
        index = load(index_path)
        entry = next(e for e in index["operators"] if e["id"] == operator)
        evidence_path = root / entry["evidence"]
        evidence = load(evidence_path)
        require(
            evidence.get("$schema") == "operator-benchmark-evidence/v1",
            "Unsupported evidence schema",
        )
        ok, reason = custody.verify_dir_checksum(str(evidence_path.parent))
        require(ok, reason)
        result = validate(directory, entry)
        previous = evidence.get("serving_sync")
        if previous == result["serving_sync"]:
            return "already_synced"
        if previous:
            require(
                previous["archive"] != result["serving_sync"]["archive"],
                "Previously synced raw data changed",
            )
            require(
                previous["completed_ns"] < result["serving_sync"]["completed_ns"],
                "Older collection cannot replace newer evidence",
            )
        evidence.setdefault("serving_history", []).append(
            {
                "index": copy.deepcopy(entry),
                "evidence": {
                    k: copy.deepcopy(v)
                    for k, v in evidence.items()
                    if k != "serving_history"
                },
            }
        )
        evidence.update(result)
        entry.update(
            l3_numerical_guard="passed",
            serving_status=result["serving_end_to_end"]["status"],
        )
        status = (
            "default_off_serving_no_gain"
            if entry["serving_status"] == "measured_no_gain"
            else "integrated_host_fork_measured"
            if entry["layer"] == "host_fork"
            else "integrated_plugin_measured"
        )
        entry["status"] = evidence["status"] = status
        checksum = evidence_path.parent / "checksum.sha256"
        backups = {
            str(p.relative_to(root)): p.read_text()
            for p in (index_path, evidence_path, checksum)
        }
        atomic(journal, backups)
        try:
            atomic(evidence_path, evidence)
            custody.write_dir_checksum(str(evidence_path.parent))
            atomic(index_path, index)
            require(
                custody.verify_dir_checksum(str(evidence_path.parent))[0],
                "Checksum verification failed",
            )
        except BaseException:
            for path, content in backups.items():
                (root / path).write_text(content)
            journal.unlink()
            raise
        # A hard interruption leaves the recovery journal in place.
        journal.unlink()
        return "synced"
