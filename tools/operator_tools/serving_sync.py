"""Promote complete serving collections; never launch a server or modify raw data."""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import re
import sys
from pathlib import Path

from tools.operator_tools import (
    custody,
    host_runtime,
    serving_attribution,
    serving_collect,
)
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


def validate_route(entry, observed):
    require(entry.get("service_route"), "Operator has no reviewed service route")
    receipt = observed.get("service_routes", {}).get(entry["id"], {})
    expected = host_runtime.route_identity(entry)
    identity = dict(receipt.get("identity", {}))
    plugin_hash = identity.pop("reviewed_plugin_sha256", None)
    require(identity == expected, "Implementation owner or consumer mismatch")
    hashes = receipt.get("sha256", {})
    require(
        set(hashes) == {expected["reader"], expected["consumer"]}
        and all(re.fullmatch(r"[a-f0-9]{64}", h) for h in hashes.values()),
        "Missing source fingerprints",
    )
    if expected["layer"] == "plugin_carrier":
        require(
            plugin_hash == hashes[expected["reader"]], "Plugin fingerprint mismatch"
        )
    return receipt


def timing_cohort(config, concurrency, order):
    # Only record explicit timing flags. Guard flags and runtime defaults may differ.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int)
    parser.add_argument("--max-model-len")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction
    )
    flags, _ = parser.parse_known_args(config["server_command"][1:])
    length = flags.max_model_len
    return {
        "model": config["model_path"],
        "tensor_parallel_size": flags.tensor_parallel_size,
        "max_model_len": int(length) if length and length.isdecimal() else length,
        "server_mode": "eager" if flags.enforce_eager else None,
        "prefix_caching": flags.enable_prefix_caching,
        "concurrency": concurrency,
        "order": order,
    }


def validate(directory, entry, *, legacy_replay=False):
    files = {}

    def read(relative):
        path = directory / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load(path)

    require(not (directory / "failure.json").exists(), "Collection failed")
    config = read("config.json")
    require(
        entry["gate_env"] == config["gate"] + "=1",
        f"Gate mismatch: index={entry['gate_env']!r}, collection={config['gate']}=1",
    )
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
    receipt = None
    if not legacy_replay or "service_routes" in observed:
        receipt = validate_route(entry, observed)
    texts = {}
    order = ["off", "on", "on", "off"]
    for stage, arm in [("guard_off", "off"), ("guard_on", "on")] + [
        (f"bench_{i}_{arm}", arm) for i, arm in enumerate(order)
    ]:
        invocation = read(f"{stage}/invocation.json")
        require(invocation["gate"] == ("1" if arm == "on" else "0"), "Arm mismatch")
        command = invocation["command"]
        require(
            command[: len(config["server_command"])] == config["server_command"],
            "Server command mismatch",
        )
        if stage.startswith("bench_"):
            require(command == config["server_command"], "Timing server command drift")
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
            "gdn_split_ba": ("gdn_split_ba_2d_kernel", 30),
        }
        require(
            (config.get("kernel"), config.get("attention_layers"))
            == audited.get(entry["id"]),
            "Unaudited kernel rule",
        )
        require(config["world_size"] == 2, "Unaudited device count")
        expected = config["attention_layers"] * 16
        device_sets = []
        profiles = {"off": [], "on": []}
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
                stats = parse_op_statistic_csv(str(path), config["kernel"])
                profiles[arm].append({"path": str(path), **stats})
                require(
                    stats["call_count_actual"] == (expected if arm == "on" else 0),
                    "Kernel count mismatch",
                )
            device_sets.append(devices)
        require(device_sets[0] == device_sets[1], "OFF/ON devices differ")
        require(
            l4["passed"] is True
            and l4.get("profiles") == profiles
            and l4.get("expected_per_rank_on") == expected,
            "Attribution differs from per-rank profiles",
        )
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
    concurrency = {}
    for workload, profile in WORKLOAD_CATALOG.items():
        paths = {"off": [], "on": []}
        concurrency[workload] = (
            pairing["primary_concurrency"]
            if workload == pairing["primary_workload"]
            else profile.default_concurrency
        )
        for i, arm in enumerate(order):
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
                concurrency=concurrency[workload],
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
        "serving_cohort": timing_cohort(config, concurrency, order),
        "serving_sync": {
            "archive": directory.as_uri(),
            "files": files,
            "host_commits": hosts,
            **({"service_route": receipt} if receipt is not None else {}),
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
            require(
                all(evidence.get(k) == v for k, v in result.items())
                and entry["serving_status"] == result["serving_end_to_end"]["status"],
                "Stored evidence or index differs from archive replay",
            )
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


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, required=True, help="completed runner archive"
    )
    parser.add_argument("--operator", required=True, help="exact delivery index ID")
    parser.add_argument(
        "--apply", action="store_true", help="promote reviewed evidence"
    )
    args = parser.parse_args(argv)
    try:
        run = args.run.resolve(strict=True)
        operator = args.operator
        entry = next(
            (
                e
                for e in load(root / "DELIVERY_INDEX.json")["operators"]
                if e["id"] == operator
            ),
            None,
        )
        require(entry is not None, "Unknown operator")
        directory = (run / operator).resolve(strict=True)
        require(
            directory.parent == run and not directory.is_relative_to(root),
            "Expected external operator archive",
        )
        completion = load(run / "completion.json")
        require(
            completion["status"] == "complete"
            and completion["items"][operator]["status"] == "done",
            "Run or item incomplete",
        )
        plan = load(run / "plan.json")
        configs = [
            e.get("config")
            for e in plan.get("items", plan.get("operators", []))
            if e["id"] == operator
        ]
        require(
            configs == [load(directory / "config.json")], "Collection differs from plan"
        )
        if args.apply:
            action = sync(directory, operator, root)
            evidence = load(root / entry["evidence"])
        else:
            action = "preview"
            evidence = validate(directory, entry)
        print(
            json.dumps(
                {
                    "action": action,
                    "operator": operator,
                    "cohort": evidence["serving_cohort"],
                    "workloads": evidence["serving_end_to_end"]["workloads"],
                },
                indent=2,
            )
        )
    except (KeyError, OSError, ValueError, TypeError) as exc:
        print(f"Backfill blocked: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
