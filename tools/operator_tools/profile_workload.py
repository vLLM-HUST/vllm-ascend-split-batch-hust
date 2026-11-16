"""Profile one workload shape against the unmodified runtime (gate off).

Purpose: produce a device-time profile under a chosen workload and graph mode so
hotspot selection is based on the execution phase being optimized, not on a
fixed short prompt. This path performs no gate comparison and writes no operator
evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import serving_collect  # noqa: E402

WORKLOADS = {
    # 2048-token input, short generation: chunked prefill dominates.
    "long-prefix": {"prompt_tokens": 2048, "max_tokens": 32},
    # Short input, long generation: decode dominates.
    "decode": {"prompt_tokens": 32, "max_tokens": 128},
    # Controlled synthetic pair from the workload catalog.
    "random": {"prompt_tokens": 512, "max_tokens": 256},
    # Real conversation distribution; prompts come from the dataset.
    "sharegpt": {"dataset": "sharegpt", "max_tokens": 128},
}

DEFAULT_SHAREGPT = "/models/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"

_FILLER = (
    "The quick brown fox jumps over the lazy dog while the river keeps "
    "carrying sediment downstream past the old mill and the narrow bridge. "
)


def build_prompt(tokens: int) -> str:
    """Build a deterministic prompt of roughly `tokens` model tokens.

    English text tokenizes near 4 characters per token; the multiplier keeps the
    result comfortably above the requested length so the prefill bucket is
    reached rather than approximated from below.
    """
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    repeats = max(1, (tokens * 4) // len(_FILLER) + 1)
    return _FILLER * repeats


def build_sharegpt_prompts(path: str, count: int) -> list[str]:
    """Take the first human turn of the first `count` conversations."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as handle:
        data = json.load(handle)
    prompts: list[str] = []
    for item in data:
        for turn in item.get("conversations") or []:
            if turn.get("from") in ("human", "user") and turn.get("value"):
                prompts.append(turn["value"])
                break
        if len(prompts) >= count:
            break
    if len(prompts) < count:
        raise ValueError(f"only {len(prompts)} usable prompts in {path}")
    return prompts[:count]


def profile_run(
    config: dict,
    output: Path,
    workload: str,
    prompts: list[str],
    concurrency: int,
    warmup: int,
) -> dict:
    spec = WORKLOADS[workload]
    max_tokens = spec["max_tokens"]

    serve_dir = output / "serve"
    with serving_collect.server(config, serve_dir, "off", profile=True) as env:
        # Graph capture happens on the first requests; keep it outside the
        # profiled window so the trace reflects replay, not capture.
        for _ in range(warmup):
            serving_collect.completion(config, prompts[0][:256], 8)

        serving_collect.request(config["port"], "/start_profile", {})
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = list(
                    pool.map(
                        lambda prompt: serving_collect.completion(
                            config, prompt, max_tokens
                        ),
                        prompts,
                    )
                )
        finally:
            serving_collect.request(config["port"], "/stop_profile", {})
        serving_collect.save(
            output / "responses.json",
            {
                "count": len(results),
                "workload": workload,
                "gate": env[config["gate"]],
            },
        )
    serving_collect.analyse_profile(config, serve_dir)
    return {"workload": workload, "prompts": len(prompts), "concurrency": concurrency}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="long-prefix")
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--cudagraph-mode",
        default="FULL_AND_PIECEWISE",
        help="CUDAGraphMode for --compilation-config; 'none' keeps eager.",
    )
    parser.add_argument("--no-cudagraph", action="store_true")
    parser.add_argument("--sharegpt-path", default=DEFAULT_SHAREGPT)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    config["gate"] = os.environ.get(
        "PROFILE_GATE", config.get("gate", "VLLM_HUST_GDN_SPLIT_BA")
    )
    config["server_command"] = [
        str(config["port"]) if part == "__PORT__" else part
        for part in config["server_command"]
    ]
    config["guard_server_args"] = ["--no-enable-prefix-caching"]
    if not args.no_cudagraph:
        config["guard_server_args"] += [
            "--compilation-config",
            json.dumps({"cudagraph_mode": args.cudagraph_mode}),
        ]

    output = args.output
    if output.exists():
        raise SystemExit(f"output already exists: {output}")
    spec = WORKLOADS[args.workload]
    if spec.get("dataset") == "sharegpt":
        prompts = build_sharegpt_prompts(args.sharegpt_path, args.prompts)
        prompt_source = args.sharegpt_path
    else:
        prompts = [build_prompt(spec["prompt_tokens"]) for _ in range(args.prompts)]
        prompt_source = "synthetic"
    output.mkdir(parents=True)
    serving_collect.save(
        output / "plan.json",
        {
            "workload": args.workload,
            "spec": spec,
            "prompts": len(prompts),
            "prompt_source": prompt_source,
            "concurrency": args.concurrency,
            "guard_server_args": config["guard_server_args"],
        },
    )
    print(
        json.dumps(
            profile_run(
                config,
                output,
                args.workload,
                prompts,
                args.concurrency,
                args.warmup,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
