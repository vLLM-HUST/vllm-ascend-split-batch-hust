#!/usr/bin/env python3
"""Outer-loop serving harness for whole-network L3, L4 attribution, and E2E A/B.

Decoupled from the sub-second inner operator loop (`optimizer_loop.py`):
- Inner loop optimizes kernel compute, tensor stride contracts, and dynamic-shape
  zero-recompilation in isolation (L3 + L0).
- Outer loop (`serving_loop.py`) evaluates whole-network integration:
  1. Serving greedy numerical consistency across fixed prompts (Serving L3).
  2. Profiler `op_statistic.csv` kernel call-count verification and all-kernel
     share / signal-to-noise gating (L4).
  3. Workload-aware A/B serving comparison (`random`, `sharegpt`,
     `long-prefix`) and structured evidence write-back with checksum refresh.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

from tools.op_agent.custody import write_dir_checksum
from tools.op_agent.workloads import (
    WORKLOAD_CATALOG,
    build_bench_serve_cmd,
    select_workloads_for_operator,
)

FIXED_L3_PROMPTS: tuple[str, ...] = (
    "Explain the difference between paging and segmentation in operating systems.",
    "Derive the time complexity of binary heap construction from an unsorted array.",
    "Write a concise Python function to compute prefix sums and explain edge cases.",
    "Compare write-through and write-back CPU caches under multi-core contention.",
    "Summarize how TCP congestion control transitions from slow start to avoidance.",
    "Explain why RMSNorm avoids mean centering compared with standard LayerNorm.",
    "Describe two-phase locking in relational databases and how deadlocks arise.",
    "Give a step-by-step trace of topological sorting using Kahn's algorithm.",
)


def compare_greedy_outputs(
    baseline_outputs: list[str],
    candidate_outputs: list[str],
) -> dict[str, Any]:
    """Compare greedy outputs (`temperature=0.0`) item-by-item for exact match."""
    if len(baseline_outputs) != len(candidate_outputs):
        return {
            "passed": False,
            "matched": 0,
            "total": max(len(baseline_outputs), len(candidate_outputs)),
            "mismatched_indices": [],
            "error": (
                f"Length mismatch: baseline={len(baseline_outputs)} "
                f"vs candidate={len(candidate_outputs)}"
            ),
        }

    mismatches = [
        idx
        for idx, (b_text, c_text) in enumerate(
            zip(baseline_outputs, candidate_outputs, strict=True)
        )
        if b_text != c_text
    ]
    total = len(baseline_outputs)
    matched = total - len(mismatches)
    return {
        "passed": len(mismatches) == 0 and total > 0,
        "matched": matched,
        "total": total,
        "match_summary": f"{matched}/{total}",
        "mismatched_indices": mismatches,
    }


def parse_op_statistic_csv(
    csv_path: str,
    target_kernel_substr: str,
    expected_calls: int | None = None,
) -> dict[str, Any]:
    """Parse MindStudio / torch_npu `op_statistic.csv` for L4 call attribution."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"op_statistic.csv not found: {csv_path}")

    total_time_us = 0.0
    matched_calls = 0
    matched_time_us = 0.0
    matched_names: list[str] = []

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_name = (
                row.get("OP Type") or row.get("Op Name") or row.get("Name") or ""
            ).strip()
            count_raw = row.get("Count") or row.get("Calls") or "0"
            total_raw = (
                row.get("Total Time(us)")
                or row.get("Total Time (us)")
                or row.get("TotalTime(us)")
                or "0"
            )
            try:
                count_val = int(float(count_raw))
                time_val = float(total_raw)
            except ValueError:
                continue

            total_time_us += time_val
            if target_kernel_substr and target_kernel_substr in op_name:
                matched_calls += count_val
                matched_time_us += time_val
                matched_names.append(op_name)

    hit_verified = matched_calls > 0 and (
        expected_calls is None or matched_calls == expected_calls
    )
    share_pct = (
        round((matched_time_us / total_time_us) * 100.0, 4)
        if total_time_us > 0
        else 0.0
    )
    avg_us = round(matched_time_us / matched_calls, 2) if matched_calls > 0 else None

    return {
        "target_kernel": target_kernel_substr,
        "matched_op_types": sorted(set(matched_names)),
        "call_count_expected": expected_calls,
        "call_count_actual": matched_calls,
        "hit_verified": hit_verified,
        "target_total_ms": round(matched_time_us / 1000.0, 3),
        "target_avg_us": avg_us,
        "all_kernel_total_ms": round(total_time_us / 1000.0, 3),
        "target_share_pct": share_pct,
    }


def evaluate_e2e_snr_gate(
    net_saved_ms: float,
    all_kernel_total_ms: float,
    serving_noise_floor_pct: float = 1.5,
) -> dict[str, Any]:
    """Assess whether L4 operator savings exceed the serving noise floor.

    Prevents misinterpreting sub-0.5% Amdahl share shifts as real serving wins
    or losses when run-to-run serving variance is ~1-2%.
    """
    if all_kernel_total_ms <= 0:
        return {
            "expected_all_kernel_delta_pct": 0.0,
            "noise_floor_pct": serving_noise_floor_pct,
            "snr_ratio": 0.0,
            "recommend_full_ab": False,
            "verdict_hint": "invalid_denominator",
        }

    delta_pct = (net_saved_ms / all_kernel_total_ms) * 100.0
    snr = abs(delta_pct) / max(serving_noise_floor_pct, 1e-6)
    recommend_ab = snr >= 0.5
    hint = (
        "proceed_to_ab"
        if recommend_ab
        else "within_noise_expected_due_to_small_all_kernel_share"
    )
    return {
        "expected_all_kernel_delta_pct": round(delta_pct, 3),
        "noise_floor_pct": serving_noise_floor_pct,
        "snr_ratio": round(snr, 2),
        "recommend_full_ab": recommend_ab,
        "verdict_hint": hint,
    }


def summarize_serving_ab_runs(
    off_result_paths: list[str],
    on_result_paths: list[str],
    workload_id: str,
    noise_threshold_pct: float = 1.5,
) -> dict[str, Any]:
    """Aggregate `vllm bench serve` JSON outputs for baseline (off) vs enabled (on)."""
    if not off_result_paths or not on_result_paths:
        raise ValueError("Both off_result_paths and on_result_paths must be non-empty.")

    def _load_metrics(paths: list[str]) -> list[dict[str, float]]:
        rows = []
        for p in paths:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            rows.append(
                {
                    "output_rate": float(data.get("output_throughput", 0.0)),
                    "mean_step_ms": float(data.get("mean_tpot_ms", 0.0)),
                    "median_step_ms": float(data.get("median_tpot_ms", 0.0)),
                    "mean_first_token_ms": float(data.get("mean_ttft_ms", 0.0)),
                }
            )
        return rows

    off_rows = _load_metrics(off_result_paths)
    on_rows = _load_metrics(on_result_paths)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals)

    def _spread_pct(vals: list[float]) -> float:
        avg = _mean(vals)
        if avg <= 0 or len(vals) < 2:
            return 0.0
        return ((max(vals) - min(vals)) / avg) * 100.0

    off_rate = _mean([r["output_rate"] for r in off_rows])
    on_rate = _mean([r["output_rate"] for r in on_rows])
    off_step = _mean([r["mean_step_ms"] for r in off_rows])
    on_step = _mean([r["mean_step_ms"] for r in on_rows])

    off_spread = _spread_pct([r["output_rate"] for r in off_rows])
    on_spread = _spread_pct([r["output_rate"] for r in on_rows])
    effective_noise = max(noise_threshold_pct, off_spread, on_spread)

    rate_delta_pct = ((on_rate - off_rate) / off_rate) * 100.0 if off_rate > 0 else 0.0
    step_delta_pct = ((on_step - off_step) / off_step) * 100.0 if off_step > 0 else 0.0
    speedup = round(on_rate / off_rate, 4) if off_rate > 0 else 1.0

    if abs(rate_delta_pct) <= effective_noise:
        status = "within_noise"
    elif rate_delta_pct > 0:
        status = "verified_faster"
    else:
        status = "negative_speedup"

    return {
        "tested": True,
        "status": status,
        "workload": workload_id,
        "num_runs_off": len(off_rows),
        "num_runs_on": len(on_rows),
        "speedup": speedup,
        "gen_rate_off": round(off_rate, 2),
        "gen_rate_on": round(on_rate, 2),
        "gen_rate_delta_pct": round(rate_delta_pct, 2),
        "intra_group_spread_pct": {
            "off": round(off_spread, 2),
            "on": round(on_spread, 2),
        },
        "mean_step_off_ms": round(off_step, 2),
        "mean_step_on_ms": round(on_step, 2),
        "mean_step_delta_pct": round(step_delta_pct, 2),
    }


def update_operator_outer_evidence(
    op_dir: str,
    serving_l3: dict[str, Any] | None = None,
    l4_attribution: dict[str, Any] | None = None,
    serving_e2e: dict[str, Any] | None = None,
) -> str:
    """Write outer-loop L3/L4/E2E results into `<op_dir>/evidence.json` and re-hash."""
    evidence_path = os.path.join(op_dir, "evidence.json")
    if not os.path.isfile(evidence_path):
        raise FileNotFoundError(f"Missing evidence.json in {op_dir}")

    with open(evidence_path, encoding="utf-8") as f:
        data = json.load(f)

    if serving_l3 is not None:
        l3_sec = dict(data.get("l3_numerical_guard") or {})
        l3_sec["serving_greedy_guard"] = serving_l3
        if not serving_l3.get("passed", False):
            l3_sec["passed"] = False
        data["l3_numerical_guard"] = l3_sec

    if l4_attribution is not None:
        data["l4_attribution"] = l4_attribution

    if serving_e2e is not None:
        data["serving_end_to_end"] = serving_e2e

    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return write_dir_checksum(op_dir)


def build_outer_loop_plan(
    op_name: str,
    gate_env: str,
    phase: str = "decode",
    max_active_rows: int | None = None,
    port: int = 8350,
    model: str = "qwen35-35b",
    out_dir: str = "/tmp/outer_loop_results",
) -> dict[str, Any]:
    """Generate the outer-loop plan: primary, stress and cross-check commands."""
    pairing = select_workloads_for_operator(
        phase=phase, max_active_rows=max_active_rows
    )
    primary_id = pairing["primary_workload"]
    stress_id = pairing["stress_workload"]
    cross_check_id = pairing["cross_check_workload"]

    primary_cmd = build_bench_serve_cmd(
        workload_id=primary_id,
        port=port,
        model=model,
        result_dir=out_dir,
        result_filename=f"{op_name}_{primary_id}.json",
        concurrency=pairing["primary_concurrency"],
    )
    stress_profile = WORKLOAD_CATALOG[stress_id]
    stress_cmd = build_bench_serve_cmd(
        workload_id=stress_id,
        port=port,
        model=model,
        result_dir=out_dir,
        result_filename=f"{op_name}_{stress_id}.json",
        concurrency=stress_profile.default_concurrency,
    )
    cross_check_profile = WORKLOAD_CATALOG[cross_check_id]
    cross_check_cmd = build_bench_serve_cmd(
        workload_id=cross_check_id,
        port=port,
        model=model,
        result_dir=out_dir,
        result_filename=f"{op_name}_{cross_check_id}.json",
        concurrency=cross_check_profile.default_concurrency,
    )
    return {
        "operator": op_name,
        "gate_env": gate_env,
        "workload_pairing": pairing,
        "primary_bench_cmd": primary_cmd,
        "stress_bench_cmd": stress_cmd,
        "cross_check_bench_cmd": cross_check_cmd,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Outer-loop whole-network verification and workload planner."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan_p = sub.add_parser("plan", help="Print outer-loop workload plan for an op.")
    plan_p.add_argument("--op", required=True, help="Operator directory name.")
    plan_p.add_argument("--gate-env", required=True, help="e.g. VLLM_HUST_FOO=1")
    plan_p.add_argument("--phase", default="decode", help="decode / prefill / mixed")
    plan_p.add_argument("--max-rows", type=int, default=None)
    plan_p.add_argument("--port", type=int, default=8350)
    plan_p.add_argument("--model", default="qwen35-35b")

    l4_p = sub.add_parser("parse-l4", help="Parse op_statistic.csv for kernel hits.")
    l4_p.add_argument("--csv", required=True, help="Path to op_statistic.csv.")
    l4_p.add_argument("--kernel", required=True, help="Substring of target kernel.")
    l4_p.add_argument("--expected-calls", type=int, default=None)

    ab_p = sub.add_parser("summarize-ab", help="Summarize off/on bench serve JSONs.")
    ab_p.add_argument("--workload", default="random")
    ab_p.add_argument("--off", nargs="+", required=True)
    ab_p.add_argument("--on", nargs="+", required=True)
    ab_p.add_argument("--update-op-dir", default=None)

    greedy_p = sub.add_parser(
        "compare-greedy", help="Compare baseline vs candidate greedy text files."
    )
    greedy_p.add_argument("--baseline", required=True, help="Baseline JSON or TXT.")
    greedy_p.add_argument("--candidate", required=True, help="Candidate JSON or TXT.")
    greedy_p.add_argument("--update-op-dir", default=None)

    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build_outer_loop_plan(
            op_name=args.op,
            gate_env=args.gate_env,
            phase=args.phase,
            max_active_rows=args.max_rows,
            port=args.port,
            model=args.model,
        )
        print(json.dumps(plan, indent=2, ensure_ascii=False))
    elif args.command == "parse-l4":
        res = parse_op_statistic_csv(
            csv_path=args.csv,
            target_kernel_substr=args.kernel,
            expected_calls=args.expected_calls,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.command == "summarize-ab":
        summary = summarize_serving_ab_runs(
            off_result_paths=args.off,
            on_result_paths=args.on,
            workload_id=args.workload,
        )
        if args.update_op_dir:
            update_operator_outer_evidence(args.update_op_dir, serving_e2e=summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "compare-greedy":

        def _load_lines(path: str) -> list[str]:
            with open(path, encoding="utf-8") as f:
                if path.endswith(".json"):
                    content = json.load(f)
                    return content if isinstance(content, list) else [str(content)]
                return [line.rstrip("\r\n") for line in f if line.strip()]

        base_texts = _load_lines(args.baseline)
        cand_texts = _load_lines(args.candidate)
        res = compare_greedy_outputs(base_texts, cand_texts)
        if args.update_op_dir:
            update_operator_outer_evidence(args.update_op_dir, serving_l3=res)
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
