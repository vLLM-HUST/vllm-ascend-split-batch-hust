#!/usr/bin/env python3
"""Standard serving workload catalog, characteristics, and operator fitness mapping.

Models the three canonical container workloads (`random`, `sharegpt`,
`long-prefix`) and maps operator shape/phase traits to primary
and dynamic-shape stress workloads without hardcoding large dataset files
inside the Git repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkloadProfile:
    """Structured specification and tensor-shape behavior of a serving workload."""

    workload_id: str
    dataset_kind: str
    summary: str
    prompt_len_profile: str
    output_len_profile: str
    batch_token_behavior: str
    bottleneck_focus: str
    default_num_prompts: int
    default_concurrency: int
    ignore_eos: bool
    recommended_for_phases: tuple[str, ...]
    candidate_dataset_paths: tuple[str, ...] = ()
    extra_bench_args: tuple[str, ...] = ()


WORKLOAD_CATALOG: dict[str, WorkloadProfile] = {
    "random": WorkloadProfile(
        workload_id="random",
        dataset_kind="builtin_synthetic",
        summary=(
            "Controlled fixed-length input/output synthetic workload with zero "
            "length variance and minimal run-to-run noise."
        ),
        prompt_len_profile="fixed (default 512 or 1024 tokens)",
        output_len_profile="fixed (default 256 tokens, ignore_eos=True)",
        batch_token_behavior=(
            "Steady-state decode locks onto the concurrency bucket (e.g. T=16 or T=32) "
            "after initial ramp-up; maximizes signal-to-noise ratio for A/B comparison."
        ),
        bottleneck_focus=(
            "Steady-state decode HBM bandwidth, vector epilogue, and launch cost"
        ),
        default_num_prompts=128,
        default_concurrency=32,
        ignore_eos=True,
        recommended_for_phases=(
            "decode",
            "decode_small_batch",
            "decode_batched",
            "all",
        ),
        extra_bench_args=(
            "--dataset-name",
            "random",
            "--random-input-len",
            "512",
            "--random-output-len",
            "256",
            "--ignore-eos",
        ),
    ),
    "sharegpt": WorkloadProfile(
        workload_id="sharegpt",
        dataset_kind="real_conversation_distribution",
        summary=(
            "Real multi-turn conversation workload with skewed long-tail input/output "
            "lengths under continuous batching."
        ),
        prompt_len_profile="skewed variable distribution (64 to 1024 tokens)",
        output_len_profile="skewed variable distribution with natural EOS completion",
        batch_token_behavior=(
            "Highly dynamic step token count T across 1..2048 as requests arrive and "
            "finish asynchronously and chunked prefill interleaves with decode."
        ),
        bottleneck_focus=(
            "Dynamic shape robustness, Triton JIT recompilation regressions, "
            "tail-block masking, and row-count gate hit ratios under mixed steps"
        ),
        default_num_prompts=256,
        default_concurrency=32,
        ignore_eos=False,
        recommended_for_phases=("decode", "prefill", "mixed", "all"),
        candidate_dataset_paths=(
            "/data/vllm-hust-benchmark/scripts/traces/evoscientist-workload-sharegpt.json",
            "/vllm-workspace/bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
            "/root/vllm-workspace/bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
        ),
        extra_bench_args=("--dataset-name", "sharegpt"),
    ),
    "long-prefix": WorkloadProfile(
        workload_id="long-prefix",
        dataset_kind="builtin_synthetic_long_context",
        summary=(
            "Long-context and shared-prefix prefill-dominant workload with large "
            "per-step token counts."
        ),
        prompt_len_profile="long input (2048 to 8192 tokens, optional shared prefix)",
        output_len_profile="short generation (128 tokens)",
        batch_token_behavior=(
            "Frequent large-T chunked prefill steps (T in 512..2048) dominated by "
            "matrix multiplication and wide tensor split/cat/rearrange copies."
        ),
        bottleneck_focus=(
            "Prefill projection merge, wide tensor unpack/rearrange, and KV reuse"
        ),
        default_num_prompts=128,
        default_concurrency=32,
        ignore_eos=True,
        recommended_for_phases=("prefill", "prefill_heavy"),
        extra_bench_args=(
            "--dataset-name",
            "random",
            "--random-input-len",
            "2048",
            "--random-output-len",
            "128",
            "--ignore-eos",
        ),
    ),
}


def resolve_local_dataset_path(
    workload_id: str, env_override: str | None = None
) -> str | None:
    """Locate an offline dataset file on the container filesystem if available."""
    if env_override and os.path.isfile(env_override):
        return env_override

    env_key = f"VLLM_HUST_DATASET_{workload_id.upper().replace('-', '_')}"
    from_env = os.environ.get(env_key)
    if from_env and os.path.isfile(from_env):
        return from_env

    profile = WORKLOAD_CATALOG.get(workload_id)
    if profile is None:
        return None
    for candidate in profile.candidate_dataset_paths:
        if os.path.isfile(candidate):
            return candidate
    return None


def select_workloads_for_operator(
    phase: str,
    max_active_rows: int | None = None,
) -> dict[str, Any]:
    """Recommend primary A/B workload and dynamic-shape stress workload for an op.

    Separates controlled steady-state comparison (`primary_workload`) from
    dynamic-batch / long-tail robustness verification (`stress_workload`).
    """
    norm_phase = phase.lower().strip()
    if norm_phase in ("prefill", "prefill_heavy"):
        primary = "long-prefix"
        concurrency = 32
        rationale = (
            "Prefill-oriented operator: use `long-prefix` (2048 in / 128 out) so "
            "large-T prefill steps dominate active device time."
        )
    elif max_active_rows is not None and max_active_rows <= 32:
        primary = "random"
        concurrency = min(max_active_rows, 16)
        rationale = (
            f"Row-gated decode operator (active up to {max_active_rows} rows): use "
            f"controlled `random` at concurrency={concurrency} to stay inside the "
            "enabled decode bucket."
        )
    else:
        primary = "random"
        concurrency = 32
        rationale = (
            "General decode/epilogue operator: use controlled `random` (512 in / "
            "256 out @ concurrency=32) to lock steady-state batch size and minimize "
            "run-to-run variance."
        )

    return {
        "primary_workload": primary,
        "primary_concurrency": concurrency,
        "stress_workload": "sharegpt",
        "rationale": rationale,
    }


def build_bench_serve_cmd(
    workload_id: str,
    port: int,
    model: str,
    result_dir: str,
    result_filename: str,
    num_prompts: int | None = None,
    concurrency: int | None = None,
    dataset_path: str | None = None,
) -> list[str]:
    """Build reproducible `vllm bench serve` CLI command for a registered workload."""
    if workload_id not in WORKLOAD_CATALOG:
        valid = ", ".join(sorted(WORKLOAD_CATALOG))
        raise KeyError(f"Unknown workload '{workload_id}'. Valid workloads: {valid}")

    prof = WORKLOAD_CATALOG[workload_id]
    n_prompts = num_prompts if num_prompts is not None else prof.default_num_prompts
    max_conc = concurrency if concurrency is not None else prof.default_concurrency
    resolved_path = resolve_local_dataset_path(workload_id, dataset_path)

    cmd = [
        "vllm",
        "bench",
        "serve",
        "--model",
        model,
        "--port",
        str(port),
        *prof.extra_bench_args,
    ]
    if resolved_path is not None:
        cmd.extend(["--dataset-path", resolved_path])

    cmd.extend(
        [
            "--num-prompts",
            str(n_prompts),
            "--max-concurrency",
            str(max_conc),
            "--save-result",
            "--result-dir",
            result_dir,
            "--result-filename",
            result_filename,
        ]
    )
    return cmd
