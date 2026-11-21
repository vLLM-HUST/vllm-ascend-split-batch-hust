"""Replay a synthetic completed collection without a service or NPU."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.operator_tools import custody
from tools.operator_tools import serving_sync as sync
from tools.operator_tools.serving_loop import (
    FIXED_L3_PROMPTS,
    compare_greedy_outputs,
    summarize_serving_ab_runs,
)
from tools.operator_tools.workloads import (
    WORKLOAD_CATALOG,
    build_bench_serve_cmd,
    select_workloads_for_operator,
)

OP = "gdn_split_ba"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def collection(tmp_path):
    root, run = tmp_path / "repo", tmp_path / "archive"
    directory = run / OP
    dataset = tmp_path / "data.json"
    dataset.write_text("[]")
    index = sync.load(sync.ROOT / "DELIVERY_INDEX.json")
    entry = next(e.copy() for e in index["operators"] if e["id"] == OP)
    entry.update(
        status="l0_measured_pending_outer_loop", serving_status="pending_outer_loop"
    )
    save(root / "DELIVERY_INDEX.json", {"operators": [entry]})
    save(root / entry["evidence"], {"$schema": "operator-benchmark-evidence/v1"})
    custody.write_dir_checksum(str((root / entry["evidence"]).parent))
    gate = entry["gate_env"].removesuffix("=1")
    server = [
        "python",
        "serve",
        "model",
        "--tensor-parallel-size",
        "2",
        "--max-model-len=4096",
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]
    hosts = {"vllm": "a" * 40, "vllm-ascend": "b" * 40}
    cfg = dict(
        gate=gate,
        python="python",
        server_command=server,
        host_commits=hosts,
        world_size=2,
        kernel="gdn_split_ba_2d_kernel",
        attention_layers=30,
        model_path="model",
        model_alias="demo",
        port=8350,
        sharegpt_path=str(dataset),
        sync_mode="none",
    )
    save(directory / "config.json", cfg)
    save(run / "plan.json", {"operators": [{"id": OP, "config": cfg}]})
    save(
        run / "completion.json",
        {"status": "complete", "items": {OP: {"status": "done"}}},
    )
    route = dict(
        identity=sync.host_runtime.route_identity(entry),
        sha256={entry["service_route"][k]: "c" * 64 for k in ("reader", "consumer")},
    )
    observed = {k: {"revision": v} for k, v in hosts.items()} | {
        "service_routes": {OP: route}
    }
    save(
        directory / "host_preflight.json",
        dict(argv=["python"], returncode=0, stdout=json.dumps(observed)),
    )
    texts = ["matched"] * len(FIXED_L3_PROMPTS)
    save(directory / "serving_l3.json", compare_greedy_outputs(texts, texts))
    save(directory / "workload_pairing.json", select_workloads_for_operator("decode"))
    order = ["off", "on", "on", "off"]
    samples = {w: {"off": [], "on": []} for w in WORKLOAD_CATALOG}
    for stage in [
        "guard_off",
        "guard_on",
        *(f"bench_{i}_{arm}" for i, arm in enumerate(order)),
    ]:
        arm = stage.split("_")[-1]
        dest = directory / stage
        save(dest / "invocation.json", dict(command=server, gate=str(int(arm == "on"))))
        save(dest / "exit.json", {"returncode": 0})
        if stage.startswith("guard"):
            save(
                dest / "responses.json",
                [
                    {"choices": [{"text": x}], "usage": {"completion_tokens": 128}}
                    for x in texts
                ],
            )
            save(dest / "greedy.json", texts)
            for dev in (0, 1):
                p = dest / "profile" / str(dev) / "op_statistic.csv"
                p.parent.mkdir(parents=True)
                p.write_text(
                    "Device_id,OP Type,Count,Total Time(us)\n"
                    f"{dev},gdn_split_ba_2d_kernel,{480 if arm == 'on' else 0},3\n"
                )
            continue
        for w, profile in WORKLOAD_CATALOG.items():
            target = dest / f"{w}.json"
            rate = (
                {"random": 101, "sharegpt": 110, "long-prefix": 90}[w]
                if arm == "on"
                else 100
            )
            save(
                target,
                dict(
                    completed=profile.default_num_prompts,
                    failed=0,
                    output_throughput=rate,
                    mean_tpot_ms=2,
                ),
            )
            cli = build_bench_serve_cmd(
                w,
                8350,
                "demo",
                str(dest),
                target.name,
                concurrency=32,
                dataset_path=str(dataset) if w == "sharegpt" else None,
            )
            save(
                dest / f"{w}_invocation.json",
                dict(
                    command=[
                        "python",
                        "-m",
                        "vllm.entrypoints.cli.main",
                        *cli[1:],
                        "--tokenizer",
                        "model",
                        "--seed",
                        "42",
                    ],
                    returncode=0,
                ),
            )
            samples[w][arm].append(str(target))
    save(
        directory / "summary.json",
        {
            w: summarize_serving_ab_runs(x["off"], x["on"], w)
            for w, x in samples.items()
        },
    )
    sync.serving_collect.verify_profiles(cfg, directory)
    return root, run


def contents(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_backfill_preview_apply_and_repeat(collection, capsys):
    root, run = collection
    args = ["--run", str(run), "--operator", OP]
    before = contents(root)
    raw = contents(run)
    assert sync.main(args, root=root) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["action"] == "preview" and contents(root) == before
    assert {w: v["status"] for w, v in preview["workloads"].items()} == {
        "random": "within_noise",
        "sharegpt": "verified_faster",
        "long-prefix": "negative_speedup",
    }
    assert preview["cohort"] == dict(
        model="model",
        tensor_parallel_size=2,
        max_model_len=4096,
        server_mode="eager",
        prefix_caching=False,
        concurrency={w: 32 for w in WORKLOAD_CATALOG},
        order=["off", "on", "on", "off"],
    )
    assert sync.main(args + ["--apply"], root=root) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "synced"
    entry = sync.load(root / "DELIVERY_INDEX.json")["operators"][0]
    e = sync.load(root / entry["evidence"])
    assert (
        e["serving_cohort"] == preview["cohort"]
        and e["serving_end_to_end"]["workloads"] == preview["workloads"]
    )
    assert (
        entry["serving_status"] == "within_noise" and entry["default_enabled"] is False
    )
    assert (
        len(e["serving_history"]) == 1
        and custody.verify_dir_checksum(str(root / entry["directory"]))[0]
    )
    written = contents(root)
    assert sync.main(args + ["--apply"], root=root) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "already_synced"
    assert written == contents(root) and raw == contents(run)


@pytest.mark.parametrize(
    "damage",
    ["running", "summary", "missing_arm", "bench_command", "route", "rank_count"],
)
def test_backfill_rejects_without_writes(collection, damage):
    root, run = collection
    d = run / OP
    if damage == "running":
        save(
            run / "completion.json",
            {"status": "running", "items": {OP: {"status": "running"}}},
        )
    elif damage == "rank_count":
        p = d / "l4.json"
        j = sync.load(p)
        j["profiles"]["on"][1]["call_count_actual"] = 0
        save(p, j)
    elif damage == "summary":
        save(d / "summary.json", {})
    elif damage == "missing_arm":
        (d / "bench_3_off/random.json").unlink()
    elif damage == "bench_command":
        p = d / "bench_2_on/invocation.json"
        j = sync.load(p)
        j["command"].append("--enable-prefix-caching")
        save(p, j)
    else:
        p = d / "host_preflight.json"
        j = sync.load(p)
        v = json.loads(j["stdout"])
        v["service_routes"] = {}
        j["stdout"] = json.dumps(v)
        save(p, j)
    before = contents(root)
    assert sync.main(["--run", str(run), "--operator", OP, "--apply"], root=root) != 0
    assert (root / "DELIVERY_INDEX.json").read_bytes() == before[
        Path("DELIVERY_INDEX.json")
    ]
    assert contents(root / "ops") == {
        p.relative_to("ops"): b for p, b in before.items() if p.parts[0] == "ops"
    }


def test_audit_and_repeat_reject_changed_cohort(collection, monkeypatch):
    from tools.operator_tools import repo_audit

    root, run = collection
    args = ["--run", str(run), "--operator", OP, "--apply"]
    assert sync.main(args, root=root) == 0
    entry = sync.load(root / "DELIVERY_INDEX.json")["operators"][0]
    path = root / entry["evidence"]
    e = sync.load(path)
    e["serving_cohort"]["prefix_caching"] = True
    save(path, e)
    custody.write_dir_checksum(str(path.parent))
    monkeypatch.setattr(repo_audit, "ROOT", root)
    problems = []
    assert repo_audit.check_archives({OP: entry}, problems) == 1
    assert any("serving_cohort" in problem for problem in problems)
    before = contents(root)
    assert sync.main(args, root=root) != 0
    assert contents(root) == before
