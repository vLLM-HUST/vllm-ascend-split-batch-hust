"""Collect real serving guards and balanced A/B runs from owned server processes."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.op_agent.serving_loop import (  # noqa: E402
    FIXED_L3_PROMPTS,
    compare_greedy_outputs,
    parse_op_statistic_csv,
    summarize_serving_ab_runs,
)


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
    env[config["gate"]] = "1" if arm == "on" else "0"
    cmd = list(config["server_command"])
    if profile:
        cmd += [
            "--profiler-config",
            json.dumps(
                {
                    "profiler": "torch",
                    "torch_profiler_dir": str(output / "profile"),
                    "torch_profiler_with_stack": False,
                    "ignore_frontend": True,
                }
            ),
        ]
    save(output / "invocation.json", {"command": cmd, "gate": env[config["gate"]]})
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
        finally:
            # Only signal the process group created by this context.
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
            save(output / "exit.json", {"returncode": proc.returncode})


def collect_guard(config, output, arm):
    with server(config, output, arm, profile=True):
        for _ in range(10):
            completion(config, "Explain RMSNorm briefly.", 16)
        responses = [completion(config, p, 128) for p in FIXED_L3_PROMPTS]
        save(output / "responses.json", responses)
        texts = [r["choices"][0]["text"] for r in responses]
        save(output / "greedy.json", texts)
        request(config["port"], "/start_profile", {})
        try:
            response = completion(config, "Explain a binary heap.", 16)
            save(output / "profile_response.json", response)
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
    if data.get("completed") != expected or data.get("failed") != 0:
        raise ValueError(f"Incomplete benchmark: {path}")
    if data.get("output_throughput", 0) <= 0:
        raise ValueError(f"Invalid benchmark metrics: {path}")


def run(config, output):
    save(output / "config.json", config)
    texts = {}
    for arm in ("off", "on"):
        print(f"Collecting guard {arm}", flush=True)
        texts[arm] = collect_guard(config, output / f"guard_{arm}", arm)
    guard = compare_greedy_outputs(texts["off"], texts["on"])
    save(output / "serving_l3.json", guard)
    if not guard["passed"]:
        raise ValueError("Serving L3 failed; performance runs refused")
    verify_profiles(config, output)
    paths = {w: {a: [] for a in ("off", "on")} for w in ("random", "sharegpt")}
    for index, arm in enumerate(("off", "on", "on", "off")):
        directory = output / f"bench_{index}_{arm}"
        print(f"Collecting benchmarks {index} {arm}", flush=True)
        with server(config, directory, arm) as env:
            for _ in range(10):
                completion(config, "Explain RMSNorm briefly.", 16)
            for workload in paths:
                target = directory / f"{workload}.json"
                cmd = [
                    config["python"],
                    "-m",
                    "vllm.entrypoints.cli.main",
                    "bench",
                    "serve",
                    "--model",
                    config["model_alias"],
                    "--tokenizer",
                    config["model_path"],
                    "--port",
                    str(config["port"]),
                    "--seed",
                    "42",
                    "--dataset-name",
                    workload,
                    "--num-prompts",
                    "128",
                    "--max-concurrency",
                    "16",
                    "--save-result",
                    "--result-dir",
                    str(directory),
                    "--result-filename",
                    target.name,
                ]
                if workload == "random":
                    cmd += [
                        "--random-input-len",
                        "512",
                        "--random-output-len",
                        "256",
                        "--ignore-eos",
                    ]
                else:
                    cmd += ["--dataset-path", config["sharegpt_path"]]
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
                validate_benchmark(target, 128)
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
    for key, expected in (
        ("gate", args.gate),
        ("port", args.port),
        ("model_alias", args.model),
    ):
        if expected is not None and config[key] != expected:
            parser.error(f"Config {key} does not match requested value {expected!r}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(config, output)
    except Exception as exc:
        save(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
