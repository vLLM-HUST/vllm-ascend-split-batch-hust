"""Collect real serving guards and balanced A/B runs from owned server processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import host_runtime  # noqa: E402
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

#: `config["gate"]` 只能是**裸变量名**（形如 `VLLM_HUST_GATE_FP32`）。
#: 带赋值（`VLLM_HUST_GATE_FP32=1`）会被当成变量名整串注入环境，没人读它，
#: 于是 on/off 两臂实际完全相同，最后只能产出"噪声内"的假结论。
GATE_ENV_NAME = re.compile(r"VLLM_HUST_[A-Z0-9_]+")


def gate_env_name(config):
    """校验并返回 `config["gate"]` 的变量名；配置里没有该键时返回 None。"""
    if "gate" not in config:
        return None
    gate = config["gate"]
    if not isinstance(gate, str) or not GATE_ENV_NAME.fullmatch(gate):
        raise ValueError(
            "config['gate'] 必须是纯变量名（形如 VLLM_HUST_GATE_FP32），"
            f"不要带 =1；当前取值 {gate!r}"
        )
    return gate


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def request(port, route, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        data = response.read()
    return json.loads(data) if data else None


def completion(config, prompt, tokens):
    result = request(
        config["port"],
        "/v1/completions",
        {
            "model": config["model_alias"],
            "prompt": prompt,
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": tokens,
            "ignore_eos": True,
        },
    )
    if len(result.get("choices", [])) != 1:
        raise ValueError(f"Invalid completion response: {result}")
    if result.get("usage", {}).get("completion_tokens") != tokens:
        raise ValueError(f"Incomplete generation: {result}")
    return result


@contextmanager
def server(config, output, arm, profile=False):
    gate = gate_env_name(config)
    if gate is None:
        raise ValueError(
            "config['gate'] 必填: 目标开关的变量名"
            "（形如 VLLM_HUST_GATE_FP32，不要带 =1）"
        )
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
    env[gate] = "1" if arm == "on" else "0"
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
                    "torch_profiler_record_shapes": bool(config.get("attribution")),
                    "ignore_frontend": True,
                }
            ),
        ]
    save(output / "invocation.json", {"command": cmd, "gate": env[gate]})
    with (output / "server.log").open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        save(output / "owner.json", {"pid": proc.pid})
        collection_completed = False
        try:
            deadline = time.monotonic() + config.get("startup_timeout", 900)
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited {proc.returncode}: {output}")
                try:
                    request(port, "/health")
                    break
                except (urllib.error.URLError, TimeoutError):
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
            # Only signal the process group created by this context.
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                forced_kill = True
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            save(
                output / "exit.json",
                {
                    "returncode": proc.returncode,
                    "returncode_before_shutdown": before_shutdown,
                    "shutdown_requested": before_shutdown is None,
                    "forced_kill": forced_kill,
                    "collection_completed": collection_completed,
                },
            )


PROFILE_PROMPT = "Explain a binary heap."

# Roughly four characters per token for the filler sentence below; only used to
# size a shared prefix that is long enough to fill a cache block.
_PREFIX_FILLER = "The quick brown fox jumps over the lazy dog while counting. "


def build_shared_prefix(tokens: int) -> str:
    if tokens <= 0:
        return ""
    return _PREFIX_FILLER * max(1, math.ceil(tokens * 4 / len(_PREFIX_FILLER)))


def build_profile_prompts(count: int, shared_prefix: str) -> list[str]:
    """Prompts for the profile window of one collection run.

    A shared prefix is the part a prefix cache can reuse, so measuring it needs
    at least two requests inside the same profile window, each with a distinct
    suffix. With no prefix and a single request this returns the original probe
    prompt, which keeps the default run identical to what it was.
    """
    if count <= 1 and not shared_prefix:
        return [PROFILE_PROMPT]
    return [
        f"{shared_prefix}Now explain variant {index} of a binary heap."
        for index in range(max(1, count))
    ]


def run_profile_requests(config, prompts, tokens, concurrency):
    """Send the profile-window requests, optionally several at a time."""
    if concurrency <= 1 or len(prompts) <= 1:
        return [completion(config, prompt, tokens) for prompt in prompts]
    workers = min(concurrency, len(prompts))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda p: completion(config, p, tokens), prompts))


def collect_guard(config, output, arm):
    """Run the guard probe sequence and analyse the captured profile.

    Token counts and prompt count come from ``config["collection_shape"]`` when
    present, so a run can target a different output-length regime without
    editing this function. The defaults preserve the original shape.
    """
    shape = config.get("collection_shape") or {}
    warmup_tokens = int(shape.get("warmup_tokens") or 16)
    l3_tokens = int(shape.get("l3_tokens") or 128)
    profile_tokens = int(shape.get("profile_tokens") or 16)
    warmup_repeats = int(shape.get("warmup_repeats") or 10)
    profile_requests = int(shape.get("profile_requests") or 1)
    concurrency = int(shape.get("concurrency") or 1)
    shared_prefix = shape.get("shared_prefix") or ""
    if not shared_prefix:
        shared_prefix = build_shared_prefix(int(shape.get("shared_prefix_tokens") or 0))
    with server(config, output, arm, profile=True):
        for _ in range(warmup_repeats):
            completion(config, "Explain RMSNorm briefly.", warmup_tokens)
        responses = [completion(config, p, l3_tokens) for p in FIXED_L3_PROMPTS]
        save(output / "responses.json", responses)
        texts = [r["choices"][0]["text"] for r in responses]
        save(output / "greedy.json", texts)
        prompts = build_profile_prompts(profile_requests, shared_prefix)
        request(config["port"], "/start_profile", {})
        try:
            profiled = run_profile_requests(
                config, prompts, profile_tokens, concurrency
            )
            if len(profiled) == 1:
                save(output / "profile_response.json", profiled[0])
            else:
                save(output / "profile_responses.json", profiled)
        finally:
            request(config["port"], "/stop_profile", {})
    analyse_profile(config, output)
    return texts


def analyse_profile(config, output):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(config.get("environment", {}))
    cmd = [
        config["python"],
        "-c",
        "import sys; from torch_npu.profiler.profiler import analyse; "
        "analyse(sys.argv[1])",
        str(output / "profile"),
    ]
    with (output / "analyse.log").open("w") as log:
        subprocess.run(
            cmd,
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )


def verify_profiles(config, output):
    if config.get("attribution"):
        from tools.operator_tools.serving_attribution import verify

        result = verify(config, output)
        save(output / "l4.json", result)
        if not result["passed"]:
            raise ValueError(
                "Dedicated L4 attribution failed; performance runs refused"
            )
        return
    reports = {}
    for arm in ("off", "on"):
        paths = sorted((output / f"guard_{arm}" / "profile").rglob("op_statistic.csv"))
        if len(paths) != config["world_size"]:
            raise ValueError(f"Expected per-rank profiles, found {paths}")
        reports[arm] = [
            {"path": str(p), **parse_op_statistic_csv(str(p), config["kernel"])}
            for p in paths
        ]
    expected = config["attention_layers"] * 16
    passed = all(r["call_count_actual"] == 0 for r in reports["off"]) and all(
        r["call_count_actual"] == expected for r in reports["on"]
    )
    result = {"passed": passed, "expected_per_rank_on": expected, "profiles": reports}
    save(output / "l4.json", result)
    if not passed:
        raise ValueError("L4 attribution failed; performance runs refused")


def validate_benchmark(path, expected):
    data = json.loads(Path(path).read_text())
    if (
        type(data.get("completed")) is not int
        or data["completed"] != expected
        or type(data.get("failed")) is not int
        or data["failed"] != 0
    ):
        raise ValueError(f"Incomplete benchmark: {path}")
    for key in ("output_throughput", "mean_tpot_ms"):
        value = data.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"Invalid benchmark metrics: {path}")


def service_routes(config):
    single = gate_env_name(config)
    if single:
        gates = {single}
    elif "gates" in config:
        off, on = (config["gates"][arm] for arm in ("off", "on"))
        if (
            off.keys() != on.keys()
            or any(str(v) != "0" for v in off.values())
            or any(str(v) not in ("0", "1") for v in on.values())
        ):
            raise ValueError("Combination must compare matching OFF/ON gates")
        gates = {k for k, v in on.items() if str(v) == "1"}
        if not gates:
            raise ValueError("Combination has no enabled gate")
    else:
        return []  # Identity-only probe.
    index = json.loads((ROOT / "DELIVERY_INDEX.json").read_text())
    rows = [
        e
        for e in index["operators"]
        if e.get("service_route") and e["gate_env"] in {g + "=1" for g in gates}
    ]
    if (
        len(rows) != len(gates)
        or {e["gate_env"].removesuffix("=1") for e in rows} != gates
    ):
        raise ValueError(f"Unknown, ambiguous or inactive service gate: {gates}")
    if (
        "gates" in config
        and not config.get("guard_only")
        and set(config.get("attributions", {})) != {e["id"] for e in rows}
    ):
        raise ValueError("Combination requires attribution for every enabled operator")
    routes = [host_runtime.route_identity(e) for e in rows]
    for route in routes:
        if route["layer"] == "plugin_carrier":
            route["reviewed_plugin_sha256"] = hashlib.sha256(
                (ROOT / "src" / route["reader"]).read_bytes()
            ).hexdigest()
    return routes


def verify_host(config, output, *, reuse=False):
    if config["server_command"][0] != config["python"]:
        raise ValueError("server_command must use the configured Python interpreter")
    # Use the same explicit dependency environment as the server.
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "PYTHONPATH" and not k.startswith("VLLM_HUST_")
    }
    env.update(config.get("environment", {}))
    probe = subprocess.run(
        [
            config["python"],
            "-c",
            "import importlib.util,json,sys; "
            "s=importlib.util.spec_from_file_location('host_runtime',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print(json.dumps(m.check_host_runtime(*json.loads(sys.argv[2]))))",
            str(ROOT / "tools/operator_tools/host_runtime.py"),
            json.dumps(
                [
                    config.get("host_commits", {}),
                    service_routes(config),
                ]
            ),
        ],
        env=env,
        cwd=output,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if reuse:
        probe.check_returncode()
        previous = json.loads((output / "host_preflight.json").read_text())
        if previous["returncode"] or json.loads(previous["stdout"]) != json.loads(
            probe.stdout
        ):
            raise ValueError(
                "Runtime identity or service route changed; use a new archive"
            )
        return  # Keep the receipt that actually preceded the original stages.
    save(
        output / "host_preflight.json",
        {
            "argv": probe.args,
            "returncode": probe.returncode,
            "stdout": probe.stdout,
            "stderr": probe.stderr,
        },
    )
    probe.check_returncode()


def run(config, output):
    gate_env_name(config)
    pairing = select_workloads_for_operator(
        config.get("phase", "decode"), config.get("max_active_rows")
    )
    if not Path(config["sharegpt_path"]).is_file():
        raise FileNotFoundError(config["sharegpt_path"])
    verify_host(config, output)
    save(output / "config.json", config)
    texts = {}
    for arm in ("off", "on"):
        print(f"Collecting guard {arm}", flush=True)
        texts[arm] = collect_guard(config, output / f"guard_{arm}", arm)
    guard = compare_greedy_outputs(texts["off"], texts["on"])
    save(output / "serving_l3.json", guard)
    if not guard["passed"]:
        raise ValueError("Serving L3 failed; performance runs refused")
    if config.get("guard_only", False):
        return
    verify_profiles(config, output)
    save(output / "workload_pairing.json", pairing)
    paths = {w: {a: [] for a in ("off", "on")} for w in WORKLOAD_CATALOG}
    for index, arm in enumerate(("off", "on", "on", "off")):
        directory = output / f"bench_{index}_{arm}"
        print(f"Collecting benchmarks {index} {arm}", flush=True)
        with server(config, directory, arm) as env:
            for _ in range(10):
                completion(config, "Explain RMSNorm briefly.", 16)
            for workload in paths:
                target = directory / f"{workload}.json"
                profile = WORKLOAD_CATALOG[workload]
                command = build_bench_serve_cmd(
                    workload,
                    config["port"],
                    config["model_alias"],
                    str(directory),
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
                with (directory / f"{workload}.log").open("w") as log:
                    subprocess.run(
                        cmd,
                        cwd=directory,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=1800,
                        check=True,
                    )
                save(
                    directory / f"{workload}_invocation.json",
                    {"command": cmd, "returncode": 0},
                )
                validate_benchmark(target, profile.default_num_prompts)
                paths[workload][arm].append(str(target))
    summaries = {
        w: summarize_serving_ab_runs(p["off"], p["on"], w) for w, p in paths.items()
    }
    save(output / "summary.json", summaries)
    print(json.dumps(summaries, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate")
    parser.add_argument("--port", type=int)
    parser.add_argument("--model")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    # 入口即校验开关名: 带 `=1` 的写法会被整串当变量名注入, on/off 两臂实际相同。
    gate_env_name(config)
    for key, expected in (
        ("gate", args.gate),
        ("port", args.port),
        ("model_alias", args.model),
    ):
        if expected is not None and config[key] != expected:
            parser.error(f"Config {key} does not match requested value {expected!r}")
    output = Path(args.output).resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Raw collection output must be outside the repository")
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(config, output)
    except Exception as exc:
        save(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
