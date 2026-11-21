"""Replay the archived Recurrent experiment. No accelerator or live host imports."""

import argparse
import csv
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import median

from tools.analysis.gdn_recurrent_trace import summarize
from tools.operator_tools.next import check_hash, verify

RECORD = (
    Path(__file__).resolve().parents[2]
    / "docs/optimization-history/gdn-recurrent-vrows.json"
)


def manifest(root, spec):
    path = check_hash(root, "inputs.sha256", spec["sha256"])
    names = set()
    for line in path.read_text().splitlines():
        sha, name = line.split("  ", 1)
        if name in names:
            raise ValueError(f"duplicate manifest entry: {name}")
        check_hash(root, name, sha)
        names.add(name)
    return names


def timing(samples):
    off, on = samples["off"], samples["on"]
    if not (
        len(off) == len(on) == 100 and all(math.isfinite(x) and x > 0 for x in off + on)
    ):
        raise ValueError("expected 100 finite positive paired samples")
    deltas = [b - a for a, b in zip(off, on, strict=True)]
    return {
        "pairs": len(deltas),
        "on_faster": sum(x < 0 for x in deltas),
        "off_median_us": round(median(off), 3),
        "on_median_us": round(median(on), 3),
        "paired_median_on_minus_off_us": round(median(deltas), 3),
    }


def rank_budget(path, rank, expected):
    start = end = None
    seen = Counter()
    duration = Counter()
    graph_us = Counter()
    by_bucket = {}
    with path.open(newline="") as source:
        for r in csv.DictReader(source):
            if r["Device_id"] != rank:
                raise ValueError("rank mismatch")
            t = Decimal(r["Start Time(us)"].strip())
            dt = Decimal(r["Duration(us)"])
            if not dt.is_finite() or dt < 0:
                raise ValueError("bad duration")
            start = t if start is None else min(t, start)
            end = t + dt if end is None else max(end, t + dt)
            graph = r["Model ID"]
            if graph == "4294967295":
                continue
            graph_us[graph] += dt
            if r["Type"] != "RecurrentGatedDeltaRule":
                continue
            n = r["Input Shapes"].strip('"').split(",")[0]
            if by_bucket.setdefault(n, graph) != graph:
                raise ValueError("ambiguous graph bucket")
            seen[n] += 1
            duration[n] += dt
    if seen != expected or len(set(by_bucket.values())) != len(by_bucket):
        raise ValueError("unaccounted ordinary calls")
    eligible = duration["1"] + duration["2"]
    span = end - start
    return {
        "ordinary_calls": dict(seen),
        "eligible_main_us": float(eligible),
        "observed_kernel_span_us": float(span),
        "eligible_main_over_span_pct": round(float(100 * eligible / span), 4),
        "eligible_main_over_same_graph_kernel_sum_pct": {
            k: round(float(100 * duration[k] / graph_us[by_bucket[k]]), 3)
            for k in ("1", "2")
        },
    }


def replay(record, parent):
    specs = record["archives"]
    native, probe, trace = (
        parent / specs[k]["directory"] for k in ("native", "probe", "trace")
    )
    receipt = verify(
        {
            "receipt_sha256": specs["native"]["sha256"],
            "host_commits": record["host_commits"],
        },
        native,
    )
    for key, root, required in [
        ("probe", probe, {"samples-1.json", "samples-2.json", "l3-cases.json"}),
        ("trace", trace, {"attribution.json"}),
    ]:
        names = manifest(root, specs[key])
        required = required | {"exit-code", "identity.json"}
        if key == "trace":
            required.update(
                str(p.relative_to(trace))
                for pattern in ("kernel_details.csv", "trace_view.json")
                for p in trace.rglob(pattern)
            )
        if not required <= names or (root / "exit-code").read_text().strip() != "0":
            raise ValueError(f"incomplete {key}")
    identity = json.loads((probe / "identity.json").read_bytes())
    actual = {k: v["revision"] for k, v in identity["hosts"].items()}
    if actual != record["host_commits"] or identity != json.loads(
        (trace / "identity.json").read_bytes()
    ):
        raise ValueError("host identity mismatch")
    cases = json.loads((probe / "l3-cases.json").read_bytes())
    if [(c["rows"], c["eager_and_graph_bitwise"], c["replays"]) for c in cases] != [
        (1, True, 3),
        (2, True, 3),
    ]:
        raise ValueError("missing bitwise guard")
    capture = summarize(trace)
    if capture != json.loads((trace / "attribution.json").read_bytes()):
        raise ValueError("trace summary mismatch")
    rows = {}
    for n, c in capture.items():
        off, on = c["off"], c["on"]
        rows[n] = {
            **timing(json.loads((probe / f"samples-{n}.json").read_bytes())),
            "main_off_us": off["median_us"]["main"],
            "main_on_us": on["median_us"]["main"],
            "main_off_range_us": off["main_range_us"],
            "main_on_range_us": on["main_range_us"],
            "gate_expansion_on_us": c["extra_gate_kernels_us"],
            "blocks_off": off["main_blocks"],
            "blocks_on": on["main_blocks"],
        }
    per_rank = {}
    result = json.loads((native / "result.json").read_bytes())
    for rank, counts in receipt["per_rank"].items():
        (name,) = [
            p
            for p in result["source_sha256"]
            if f"_rank{rank}_" in p and p.endswith("/kernel_details.csv")
        ]
        per_rank[rank] = rank_budget(
            native / name, rank, counts["ordinary_device_rows"]
        )
    return {"local_ab": rows, "native_window_by_rank": per_rank}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_parent", type=Path)
    parser.add_argument(
        "--print",
        action="store_true",
        help="print recomputed facts instead of checking the record",
    )
    args = parser.parse_args()
    record = json.loads(RECORD.read_bytes())
    facts = replay(record, args.archive_parent)
    if args.print:
        print(json.dumps(facts, indent=2))
    elif facts != record["measurements"]:
        raise SystemExit("archived measurements disagree with the committed record")
    else:
        print("Recurrent: L3, paired timing, trace and native ranks verified")
