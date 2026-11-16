"""Summarize Ascend profiler output as per-op pipe composition.

Reads the `ASCEND_PROFILER_OUTPUT` directory produced by either
`torch_npu.profiler.analyse` or `msprof --export=on`, and reports where each
kernel spends time inside its own core: Cube MAC, MTE1/MTE2 fixpipe, Vector,
and MTE2/MTE3. Pipe shares describe core-internal busy time; they are not a SOL
ratio, which additionally requires a matching peak capacity.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from contextlib import suppress
from pathlib import Path

AIC_PIPES = (
    ("mac", "aic_mac_time(us)"),
    ("mte1", "aic_mte1_time(us)"),
    ("mte2", "aic_mte2_time(us)"),
    ("fixpipe", "aic_fixpipe_time(us)"),
)
AIV_PIPES = (
    ("vec", "aiv_vec_time(us)"),
    ("mte2", "aiv_mte2_time(us)"),
    ("mte3", "aiv_mte3_time(us)"),
)

# Below this share the core is not the constraint and the kernel is dominated by
# launch, tail, or wait time rather than by any single pipe.
BUSY_FLOOR = 0.15


def _number(row: dict, column: str) -> float:
    try:
        return float(row.get(column) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _classify(core: str, shares: dict[str, float]) -> str:
    busy = max(shares.values(), default=0.0)
    if busy < BUSY_FLOOR:
        return "launch_or_wait_bound"
    pipe = max(shares, key=shares.get)
    return f"{core}_{pipe}_bound"


def summarize(directory: Path) -> dict:
    kernel_csv = None
    for candidate in sorted(directory.rglob("kernel_details.csv")):
        kernel_csv = candidate
        break
    if kernel_csv is None:
        raise FileNotFoundError(f"kernel_details.csv not found under {directory}")

    ops: dict[str, dict] = {}
    with kernel_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row.get("Name")
            if not name or name.startswith("hcom"):
                # Collectives carry wait time from the peer rank; their device
                # duration is not comparable to a compute kernel's pipe shares.
                continue
            slot = ops.setdefault(
                name,
                {
                    "name": name,
                    "core": row.get("Accelerator Core", ""),
                    "calls": 0,
                    "total_us": 0.0,
                    "blocks": [],
                    "shapes": {},
                    "pipe_us": {},
                    "core_us": {"aic": 0.0, "aiv": 0.0},
                    "duration_us": [],
                },
            )
            slot["calls"] += 1
            total = _number(row, "Duration(us)")
            slot["total_us"] += total
            slot["duration_us"].append(total)
            with suppress(ValueError):
                slot["blocks"].append(int(row.get("Block Num") or 0))
            shape = row.get("Input Shapes") or ""
            slot["shapes"][shape] = slot["shapes"].get(shape, 0) + 1
            slot["core_us"]["aic"] += _number(row, "aicore_time(us)")
            slot["core_us"]["aiv"] += _number(row, "aiv_time(us)")
            for pipe, column in AIC_PIPES + AIV_PIPES:
                slot["pipe_us"][pipe] = slot["pipe_us"].get(pipe, 0.0) + _number(
                    row, column
                )

    grand = sum(slot["total_us"] for slot in ops.values()) or 1.0
    rows = []
    for slot in ops.values():
        aic, aiv = slot["core_us"]["aic"], slot["core_us"]["aiv"]
        core = "aic" if aic >= aiv else "aiv"
        core_total = max(aic, aiv)
        pipes = AIC_PIPES if core == "aic" else AIV_PIPES
        shares = {
            pipe: (slot["pipe_us"].get(pipe, 0.0) / core_total if core_total else 0.0)
            for pipe, _ in pipes
        }
        rows.append(
            {
                "op": slot["name"],
                "core": slot["core"],
                "calls": slot["calls"],
                "total_us": round(slot["total_us"], 2),
                "share_pct": round(slot["total_us"] * 100 / grand, 3),
                "avg_us": round(slot["total_us"] / slot["calls"], 3),
                "median_us": round(statistics.median(slot["duration_us"]), 3),
                "median_blocks": statistics.median(slot["blocks"])
                if slot["blocks"]
                else None,
                "core_time_us": round(core_total, 2),
                "pipe_share": {k: round(v, 4) for k, v in shares.items()},
                "busy_pipe_pct": round(max(shares.values(), default=0.0) * 100, 1),
                "shape": max(slot["shapes"], key=slot["shapes"].get)
                if slot["shapes"]
                else "",
                "bottleneck_form": _classify(core, shares),
            }
        )
    rows.sort(key=lambda item: item["total_us"], reverse=True)
    compute_us = sum(item["total_us"] for item in rows)
    return {
        "kernel_details": str(kernel_csv),
        "compute_total_us": round(compute_us, 2),
        "ops": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = summarize(args.directory)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"compute_total_us={report['compute_total_us']:.0f}")
    print(
        f"{'op':<34}{'calls':>7}{'us':>10}{'share%':>8}"
        f"{'blk':>6}{'busy%':>7}  {'core':<4}{'form':<22}shape"
    )
    for item in report["ops"][: args.top]:
        blocks = item["median_blocks"]
        blocks = "-" if blocks is None else f"{blocks:.0f}"
        print(
            f"{item['op']:<34}{item['calls']:>7}{item['total_us']:>10.1f}"
            f"{item['share_pct']:>8.2f}{blocks:>6}{item['busy_pipe_pct']:>7.1f}"
            f"  {item['core']:<4}{item['bottleneck_form']:<22}{item['shape'][:40]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
