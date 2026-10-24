# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import json
import sys
from pathlib import Path

import bench_concatd_views_adapter as adapter
import bench_gdn_qkv_unpack as bench
import torch


def shape_source(path, dims):
    doc = json.loads(Path(path).read_text())
    tags = doc.get("cohort", {})
    if tags.get("model") != bench.MODEL or tags.get("precision") != "bf16":
        raise ValueError("shape source model/precision mismatch")
    kernel = Path(doc["asc_dir"]) / "ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    counts = {}
    with kernel.open() as f:
        for row in csv.DictReader(f):
            if row.get("Type") != "RecurrentGatedDeltaRule":
                continue
            parts = row.get("Input Shapes", "").strip('"').split(";")
            if len(parts) < 3:
                continue
            tokens = int(parts[0].split(",")[0])
            expected = [
                f"{tokens},{dim // head},{head}"
                for dim, head in (
                    (dims.q_dim, dims.head_k_dim),
                    (dims.k_dim, dims.head_k_dim),
                    (dims.v_dim, dims.head_v_dim),
                )
            ]
            if parts[:3] == expected and row.get("Input Data Types", "").startswith(
                "DT_BF16;DT_BF16;DT_BF16;"
            ):
                counts[tokens] = counts.get(tokens, 0) + 1
    if not counts:
        raise ValueError("no matching recurrent decode shape")
    return {
        "opstat": {"path": str(path), "sha256": bench.sha256_file(path)},
        "kernel_details": {"path": str(kernel), "sha256": bench.sha256_file(kernel)},
        "weight_tags": tags,
        "decode_calls_by_tokens": counts,
        "layout_boundary": "contiguous conv output; no cross-layout claim",
    }


def check_outputs(mixed, dtype, dims):
    before = bench.int_view_cpu(mixed, dtype).clone()
    baseline = bench.baseline_unpack(mixed, dims)
    candidate = adapter.unpack(mixed)
    comparison = bench.compare_arm(baseline, candidate, dtype)
    contiguous = all(t.is_contiguous() for t in candidate)
    unchanged = bool(torch.equal(before, bench.int_view_cpu(mixed, dtype)))
    return {
        "comparison": comparison,
        "contiguous": contiguous,
        "input_unmodified": unchanged,
        "ok": comparison["ok"] and contiguous and unchanged,
    }


def run_cell(tokens, args, dims, source, device, normalize=None):
    numerics = {}
    for dtype in ("bfloat16", "float16", "float32"):
        x = bench.build_input(tokens, dims.width, dtype, args.seed, True, device)
        numerics[dtype] = check_outputs(x, dtype, dims)
        if not numerics[dtype]["ok"]:
            return {"tokens": tokens, "numeric": numerics, "ok": False}
    x = bench.build_input(tokens, dims.width, "bfloat16", args.seed + 1, False, device)
    consumer = None
    if normalize is not None:
        base = bench.baseline_unpack(x, dims)
        cand = adapter.unpack(x)
        consumer = [
            bool(
                torch.equal(
                    bench.int_view_cpu(normalize(a), "bfloat16"),
                    bench.int_view_cpu(normalize(b), "bfloat16"),
                )
            )
            for a, b in zip(base[:2], cand[:2], strict=True)
        ]
        if not all(consumer):
            return {
                "tokens": tokens,
                "numeric": numerics,
                "l2norm_bitwise": consumer,
                "ok": False,
            }
    functions = {
        "baseline": lambda: bench.baseline_unpack(x, dims),
        "candidate": lambda: adapter.unpack(x),
    }
    for fn in functions.values():
        for _ in range(args.warmup):
            fn()
    rows = {name: [] for name in functions}
    for order in bench.ab_schedule(tuple(functions), args.repeats):
        for name in order:
            rows[name].append(bench.timed(functions[name], args.iters, device))
    arms = {}
    for name in rows:
        median, key = bench.median_metric(rows[name], device)
        arms[name] = {"us_median": median, "metric": key, "runs": rows[name]}
    key = arms["baseline"]["metric"]
    paired = bench.paired_changes(
        [r[key] for r in rows["baseline"]], [r[key] for r in rows["candidate"]]
    )
    return {
        "tokens": tokens,
        "input_shape": [tokens, dims.width],
        "input_stride": list(x.stride()),
        "phase": "decode",
        "shape_role": (
            "measured-shape"
            if tokens in source["decode_calls_by_tokens"]
            else "synthetic-coverage"
        ),
        "numeric": numerics,
        "l2norm_bitwise": consumer,
        "performance": {
            "arms": arms,
            "paired_delta_pct": paired,
            "timer_scope": bench.TIMER_SCOPE_NPU
            if device == "npu"
            else bench.TIMER_SCOPE_CPU,
        },
        "ok": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-source-opstat", required=True)
    parser.add_argument("--tokens", default="1,2,8,16,32")
    parser.add_argument("--device", choices=["npu", "cpu"], default="npu")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.iters <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("new output path and positive timing parameters required")
    tokens = [int(x) for x in args.tokens.split(",")]
    if not tokens or any(t <= 0 for t in tokens):
        parser.error("tokens must be positive")
    product = {
        "schema": "concatd-contiguous-l0.v1",
        "status": "failed",
        "cells": [],
        "command": list(sys.argv if argv is None else argv),
        "not_claimed": [
            "No FULL graph or throughput conversion of eager timings",
            "Separate QKV storage is permitted by this helper; each tensor must "
            "match stock shape/stride/dtype/values exactly",
            "Supplementary token counts without CSV matches remain synthetic",
        ],
    }
    try:
        dims = bench.Dims()
        source = shape_source(args.shape_source_opstat, dims)
        product["shape_source"] = source
        product["cohort"] = {
            "model": source["weight_tags"]["model"],
            "precision": source["weight_tags"]["precision"],
            "weight_format": "not-applicable",
        }
        device, hw = bench.resolve_device(args.device)
        product["hardware"] = hw
        runtime_source = (
            Path(adapter.__file__).resolve().parents[1]
            / "src/vllm_ascend_split_batch/runtime_source_patches.py"
        )
        product["source_hashes"] = [
            {"path": str(p), "sha256": bench.sha256_file(str(p))}
            for p in (
                Path(__file__),
                Path(adapter.__file__),
                Path(bench.__file__),
                runtime_source,
            )
        ]
        normalize = None
        if device == "npu":
            from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

            normalize = l2norm_fwd
        for t in tokens:
            cell = run_cell(t, args, dims, source, device, normalize)
            product["cells"].append(cell)
            print(json.dumps(cell), flush=True)
            if not cell["ok"]:
                raise ValueError(f"first failed cell T={t}")
        product["status"] = "ok"
    except Exception as exc:
        product["error"] = repr(exc)
        print(repr(exc), file=sys.stderr, flush=True)
    with args.out.open("x") as f:
        json.dump(product, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0 if product["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
