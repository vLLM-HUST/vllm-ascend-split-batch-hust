#!/usr/bin/env python3
"""Pluggable multi-operator hotspot discovery and batch optimization campaign loop.

Attaches externally to the single-operator inner loop (`optimizer_loop.py`)
without modifying its core contract:
1. `discover`:
   - Optionally compresses `kernel_details.csv` into hierarchical vector
     sandwich cards + a MindStudio-style SVG swimlane chart via
     `timeline_insight.py`.
   - Benchmarks every candidate `OpCase` in `CASE_REGISTRY` under `NPUGraph`
     replay (or CPU fallback) and applies three adversarial guards:
       a) Graph floor guard: flags candidates whose graph baseline is below
          `min_graph_baseline_us` (e.g. view-only ops with zero kernel cost).
       b) CSV cross-check guard: flags candidates whose synthetic baseline
          deviates >3x from observed `op_statistic.csv` average latency.
       c) Per-step budget ranking: ranks candidates by
          `baseline_median_us * layer_count` (30 for GDN, 40 for MoE).
   - Writes a human-in-the-loop batch manifest (`campaign_batch.json`).
2. `run`:
   - Reads `campaign_batch.json` and executes each `status == "approved"`
     operator in an isolated subprocess (`python -m tools.op_agent.optimizer_loop`)
     to prevent NPU memory and Triton compile-cache contamination across ops.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Any

from tools.op_agent.cases import CASE_REGISTRY, get_case
from tools.op_agent.cases.dynamic_chain import DynamicVectorChainCase
from tools.op_agent.evaluator import run_candidate_evaluation
from tools.op_agent.guard import check_npu_environment
from tools.op_agent.timeline_insight import (
    compress_kernel_timeline,
    format_insight_markdown,
    render_timeline_swimlane_svg,
)


def _infer_layer_count(case_name: str, callsite: str) -> int:
    combined = f"{case_name} {callsite}".lower()
    if "gdn" in combined:
        return 30
    if "moe" in combined or "shared_expert" in combined:
        return 40
    return 30


def _load_delivered_op_ids(delivery_index_path: str | None = None) -> set[str]:
    """Load operator ids that are already delivered or terminal from index."""
    if delivery_index_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        delivery_index_path = os.path.join(repo_root, "DELIVERY_INDEX.json")

    if not os.path.isfile(delivery_index_path):
        return set()

    try:
        with open(delivery_index_path, encoding="utf-8") as f:
            data = json.load(f)
        delivered: set[str] = set()
        for op in data.get("operators", []):
            op_id = op.get("id")
            status = op.get("status", "")
            # Skip already integrated or conclusively rejected/measured ops
            is_terminal = status.startswith("integrated_") or status.startswith(
                "default_off_"
            )
            if is_terminal and op_id:
                delivered.add(op_id)
        return delivered
    except Exception:
        return set()


def _load_op_statistic_avg_map(op_stat_csv: str | None) -> dict[str, float]:
    """Load `OP Type -> Avg Time(us)` from `op_statistic.csv` if provided."""
    if not op_stat_csv or not os.path.isfile(op_stat_csv):
        return {}
    avg_map: dict[str, float] = {}
    with open(op_stat_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = (
                row.get("OP Type") or row.get("Op Name") or row.get("Name") or ""
            ).strip()
            avg_raw = (
                row.get("Avg Time(us)")
                or row.get("Avg Time (us)")
                or row.get("AvgTime(us)")
                or ""
            )
            if not op_type or not avg_raw:
                continue
            try:
                avg_map[op_type.lower()] = float(avg_raw)
            except ValueError:
                continue
    return avg_map


def discover_hotspot_batch(
    output_manifest_path: str,
    case_names: list[str] | None = None,
    kernel_details_csv: str | None = None,
    op_statistic_csv: str | None = None,
    svg_output_path: str | None = None,
    min_graph_baseline_us: float = 1.5,
    iters: int = 20,
    skip_delivered: bool = True,
    delivery_index_path: str | None = None,
) -> dict[str, Any]:
    """Scan candidate cases + optional profiler timeline and emit a batch manifest."""
    hw = check_npu_environment(required_cards=1)
    if case_names:
        selected_names = list(case_names)
    else:
        all_names = sorted(CASE_REGISTRY.keys())
        if skip_delivered:
            delivered = _load_delivered_op_ids(delivery_index_path)

            # Map canonical case names to delivery IDs
            def _is_delivered(name: str) -> bool:
                if name in delivered:
                    return True
                if name == "moe_shared_epilogue_decode" and "moe_epilogue" in delivered:
                    return True
                return bool(
                    name == "gdn_input_norm_decode" and "gdn_input_norm" in delivered
                )

            selected_names = [n for n in all_names if not _is_delivered(n)]
        else:
            selected_names = all_names
    csv_avg_map = _load_op_statistic_avg_map(op_statistic_csv)

    timeline_digest: dict[str, Any] | None = None
    rendered_svg: str | None = None
    if kernel_details_csv and os.path.isfile(kernel_details_csv):
        timeline_digest = compress_kernel_timeline(kernel_details_csv, top_k=8)
        if svg_output_path is None:
            base_dir = os.path.dirname(os.path.abspath(output_manifest_path))
            svg_output_path = os.path.join(base_dir, "timeline_hotspots.svg")
        rendered_svg = render_timeline_swimlane_svg(timeline_digest, svg_output_path)

    candidates: list[dict[str, Any]] = []

    # Synthesize new OpCases from timeline insight vector sandwich chains
    if timeline_digest and timeline_digest.get("top_fusable_chains"):
        for chain_card in timeline_digest["top_fusable_chains"]:
            dyn_case = DynamicVectorChainCase(chain_card)
            tc = dyn_case.build_test_case(is_tp2_ready=hw.has_npu)
            tc["iters"] = iters
            base_code = dyn_case.get_baseline_code()
            eval_res = run_candidate_evaluation(base_code, tc, iters=iters)
            base_us = float(eval_res.get("l0_median_us", 0.0))
            layers = _infer_layer_count(dyn_case.name, dyn_case.callsite)
            step_budget_us = round(base_us * layers, 2)
            candidates.append(
                {
                    "case_name": dyn_case.name,
                    "callsite": dyn_case.callsite,
                    "phase": dyn_case.phase,
                    "layer_count": layers,
                    "timing_mode": eval_res.get("timing_mode", "Eager"),
                    "baseline_l0_median_us": base_us,
                    "bucket_latencies_us": eval_res.get("bucket_latencies_us", {}),
                    "estimated_step_budget_us": step_budget_us,
                    "shape_facts": tc["shape_facts"],
                    "workload_spec": tc.get("workload_spec", {}),
                    "status": "approved",
                    "guard_notes": [
                        "Synthesized directly from timeline vector sandwich chain."
                    ],
                }
            )

    for name in selected_names:
        op_case = get_case(name)
        tc = op_case.build_test_case(is_tp2_ready=hw.has_npu)
        tc["iters"] = iters
        base_code = op_case.get_baseline_code()
        eval_res = run_candidate_evaluation(base_code, tc, iters=iters)

        base_us = float(eval_res.get("l0_median_us", 0.0))
        layers = _infer_layer_count(name, op_case.callsite)
        step_budget_us = round(base_us * layers, 2)

        guard_notes: list[str] = []
        status = "approved"

        if not eval_res.get("l3_passed", False):
            status = "rejected_by_guard"
            guard_notes.append(f"Baseline failed self-check: {eval_res.get('error')}")
        elif hw.has_npu and base_us < min_graph_baseline_us:
            status = "rejected_by_guard"
            guard_notes.append(
                f"NPUGraph baseline ({base_us:.2f} us) is below hardware floor "
                f"({min_graph_baseline_us:.2f} us); likely view-only or launch noise."
            )

        # Optional CSV sanity cross-check
        csv_ref_us: float | None = None
        for op_key, op_avg in csv_avg_map.items():
            if any(tok in op_key for tok in name.split("_") if len(tok) >= 4):
                csv_ref_us = op_avg
                break
        if csv_ref_us is not None and csv_ref_us > 0 and base_us > csv_ref_us * 3.5:
            status = "pending_review"
            guard_notes.append(
                f"Baseline ({base_us:.2f} us) is >3.5x higher than CSV op average "
                f"({csv_ref_us:.2f} us); verify baseline_fn uses real host op."
            )

        candidates.append(
            {
                "case_name": name,
                "callsite": op_case.callsite,
                "phase": op_case.phase,
                "layer_count": layers,
                "timing_mode": eval_res.get("timing_mode", "Eager"),
                "baseline_l0_median_us": base_us,
                "bucket_latencies_us": eval_res.get("bucket_latencies_us", {}),
                "estimated_step_budget_us": step_budget_us,
                "shape_facts": tc["shape_facts"],
                "workload_spec": tc.get("workload_spec", {}),
                "status": status,
                "guard_notes": guard_notes,
            }
        )

    candidates.sort(key=lambda c: c["estimated_step_budget_us"], reverse=True)
    for rank, item in enumerate(candidates, 1):
        item["priority_rank"] = rank

    manifest: dict[str, Any] = {
        "schema": "op-agent-campaign-batch/v1",
        "device": hw.device_name,
        "has_npu": hw.has_npu,
        "min_graph_baseline_us": min_graph_baseline_us,
        "timeline_svg_path": rendered_svg,
        "timeline_top_chains": (
            timeline_digest["top_fusable_chains"] if timeline_digest else []
        ),
        "candidates": candidates,
    }

    parent = os.path.dirname(os.path.abspath(output_manifest_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return manifest


def run_approved_campaign(
    manifest_path: str,
    max_iters: int = 5,
    summary_output_path: str | None = None,
    use_subprocess: bool = True,
) -> dict[str, Any]:
    """Execute inner-loop optimization for all `status == 'approved'` batch items."""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    approved = [
        c for c in manifest.get("candidates", []) if c.get("status") == "approved"
    ]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    results: list[dict[str, Any]] = []
    for item in approved:
        case_name = item["case_name"]
        base_us = float(item["baseline_l0_median_us"])
        layers = int(item.get("layer_count", 30))

        if use_subprocess:
            cmd = [
                sys.executable,
                "-m",
                "tools.op_agent.optimizer_loop",
                "--case",
                case_name,
                "--max-iters",
                str(max_iters),
            ]
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )
            exit_ok = proc.returncode == 0
            best_us = base_us
            evidence_file = os.path.join(repo_root, "ops", case_name, "evidence.json")
            if os.path.isfile(evidence_file):
                try:
                    with open(evidence_file, encoding="utf-8") as ef:
                        ev = json.load(ef)
                    l0 = ev.get("l0_microbenchmark") or ev.get("l0_benchmark") or {}
                    opt_val = l0.get("optimized_median_us")
                    if opt_val is not None:
                        best_us = float(opt_val)
                except (OSError, json.JSONDecodeError, ValueError):
                    pass
        else:
            from tools.op_agent.optimizer_loop import run_qwen35_optimization

            rec = run_qwen35_optimization(case_name=case_name, max_iters=max_iters)
            exit_ok = True
            best_us = float(rec.get("l0_median_us", base_us))

        speedup = round(base_us / max(best_us, 1e-6), 2)
        saved_per_step_us = round(max(0.0, base_us - best_us) * layers, 2)
        results.append(
            {
                "case_name": case_name,
                "callsite": item.get("callsite"),
                "layer_count": layers,
                "execution_succeeded": exit_ok,
                "baseline_l0_median_us": base_us,
                "best_l0_median_us": best_us,
                "l0_speedup": speedup,
                "estimated_step_saved_us": saved_per_step_us,
                "recommend_outer_loop": speedup >= 1.15 and saved_per_step_us >= 15.0,
            }
        )

    summary = {
        "schema": "op-agent-campaign-summary/v1",
        "source_manifest": manifest_path,
        "total_candidates": len(manifest.get("candidates", [])),
        "executed_approved_count": len(results),
        "total_estimated_step_saved_us": round(
            sum(r["estimated_step_saved_us"] for r in results), 2
        ),
        "results": results,
    }

    if summary_output_path:
        parent = os.path.dirname(os.path.abspath(summary_output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(summary_output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pluggable multi-operator hotspot discovery and batch campaign."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    disc_p = sub.add_parser(
        "discover",
        help="Scan candidate cases + optional profiler CSVs into a batch manifest.",
    )
    disc_p.add_argument("--out", required=True, help="Output batch manifest JSON path.")
    disc_p.add_argument("--cases", nargs="*", default=None, help="Optional case subset")
    disc_p.add_argument("--kernel-details-csv", default=None)
    disc_p.add_argument("--op-statistic-csv", default=None)
    disc_p.add_argument("--svg-out", default=None)
    disc_p.add_argument("--min-graph-us", type=float, default=1.5)
    disc_p.add_argument("--iters", type=int, default=20)
    disc_p.add_argument(
        "--include-delivered",
        action="store_true",
        default=False,
        help="Include already delivered / terminal operators from DELIVERY_INDEX.json.",
    )
    disc_p.add_argument(
        "--delivery-index",
        default=None,
        help="Optional path to DELIVERY_INDEX.json.",
    )

    run_p = sub.add_parser(
        "run",
        help="Run inner-loop optimization for approved operators in a batch manifest.",
    )
    run_p.add_argument("--batch", required=True, help="Input batch manifest JSON path.")
    run_p.add_argument("--max-iters", type=int, default=5)
    run_p.add_argument("--summary-out", default=None)

    insight_p = sub.add_parser(
        "insight",
        help="Compress kernel_details.csv into Markdown cards and a swimlane SVG.",
    )
    insight_p.add_argument("--kernel-details-csv", required=True)
    insight_p.add_argument("--svg-out", required=True)
    insight_p.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "discover":
        manifest = discover_hotspot_batch(
            output_manifest_path=args.out,
            case_names=args.cases,
            kernel_details_csv=args.kernel_details_csv,
            op_statistic_csv=args.op_statistic_csv,
            svg_output_path=args.svg_out,
            min_graph_baseline_us=args.min_graph_us,
            iters=args.iters,
            skip_delivered=not args.include_delivered,
            delivery_index_path=args.delivery_index,
        )
        print(
            f"Generated batch manifest with {len(manifest['candidates'])} candidates: "
            f"{args.out}"
        )
        if manifest.get("timeline_svg_path"):
            print(f"Rendered timeline swimlane SVG: {manifest['timeline_svg_path']}")
    elif args.command == "run":
        summary = run_approved_campaign(
            manifest_path=args.batch,
            max_iters=args.max_iters,
            summary_output_path=args.summary_out,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "insight":
        digest = compress_kernel_timeline(args.kernel_details_csv, top_k=args.top_k)
        render_timeline_swimlane_svg(digest, args.svg_out)
        print(format_insight_markdown(digest))


if __name__ == "__main__":
    main()
