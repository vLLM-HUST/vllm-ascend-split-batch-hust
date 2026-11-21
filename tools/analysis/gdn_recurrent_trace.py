"""Read the four --trace-only replay windows; do not run or time a kernel."""

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path
from statistics import median

NORM = "l2norm_fwd_kernel2"
BROADCAST = "aclnnRepeatInterleaveIntWithDim_BroadcastToAiCore_BroadcastTo"
REPEAT = "aclnnRepeatInterleaveIntWithDim_RepeatInterleaveV2_RepeatInterleave"
MAIN = "aclnnRecurrentGatedDeltaRule_RecurrentGatedDeltaRule_RecurrentGatedDeltaRule"


def window(path, split):
    (detail,) = path.rglob("kernel_details.csv")
    with detail.open(newline="") as source:
        rows = sorted(
            csv.DictReader(source), key=lambda r: Decimal(r["Start Time(us)"])
        )
    timeline = json.loads(detail.with_name("trace_view.json").read_bytes())
    events = [e for e in timeline if e.get("ph") == "X"]
    starts = sorted(Decimal(e["ts"]) for e in events if e["name"] == "MODEL_EXECUTE")
    kernels = sorted(
        (
            e
            for e in events
            if e.get("args", {}).get("Task Type", "").startswith("KERNEL")
        ),
        key=lambda e: Decimal(e["ts"]),
    )
    seq = [NORM, NORM] + ([BROADCAST, REPEAT] * 2 if split else []) + [MAIN]
    if len(starts) != 16 or len(kernels) != len(rows) or len(rows) != 16 * len(seq):
        raise ValueError(f"{path}: missing or extra kernels/replays")
    graphs = {(e["args"]["Model Id"], e["args"]["Physic Stream Id"]) for e in kernels}
    if len(graphs) != 1 or next(iter(graphs))[0] == 4294967295:
        raise ValueError(f"{path}: kernels not on one captured graph/stream")
    times = {"qk_norm": [], "gate_expansion": [], "main": [], "all_kernels": []}
    samples = []
    blocks = set()
    for i, start in enumerate(starts):
        ds = []
        end = starts[i + 1] if i + 1 < len(starts) else Decimal("Infinity")
        for task, name in enumerate(seq):
            j = i * len(seq) + task
            r, e = rows[j], kernels[j]
            ts, dt = Decimal(r["Start Time(us)"]), Decimal(r["Duration(us)"])
            if not (
                r["Device_id"] == "0"
                and e["name"] == r["Name"] == name
                and e["args"]["Task Id"] == task
                and ts == Decimal(e["ts"])
                and dt == Decimal(str(e["dur"]))
                and dt > 0
                and start <= ts < ts + dt <= end
            ):
                raise ValueError(
                    f"{path}: replay {i} task {task} disagrees with graph or CSV"
                )
            ds.append(dt)
            if name == MAIN:
                blocks.add(int(r["Block Num"]))
        for k, val in zip(
            times, (sum(ds[:2]), sum(ds[2:-1]), ds[-1], sum(ds)), strict=True
        ):
            times[k].append(val)
        samples.append([float(x) for x in ds])
    if len(blocks) != 1:
        raise ValueError(f"{path}: main block count changed")
    return {
        "replays": len(starts),
        "kernel_order": seq,
        "main_blocks": blocks.pop(),
        "per_replay_us": samples,
        "median_us": {k: float(median(v)) for k, v in times.items()},
        "main_range_us": [float(min(times["main"])), float(max(times["main"]))],
    }


def summarize(root):
    if (root / "exit-code").read_text().strip() != "0":
        raise ValueError("incomplete collection")
    result = {}
    for n in (1, 2):
        arms = {
            arm: window(root / f"rows-{n}-{arm}", arm == "on") for arm in ("off", "on")
        }
        off, on = (arms[a]["median_us"] for a in ("off", "on"))
        result[str(n)] = {
            **arms,
            "main_delta_on_minus_off_us": round(on["main"] - off["main"], 3),
            "extra_gate_kernels_us": on["gate_expansion"],
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(summarize(parser.parse_args().archive), indent=2))
