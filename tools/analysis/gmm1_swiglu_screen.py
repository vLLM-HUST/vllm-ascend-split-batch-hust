"""Read the fixed BF16 GMM1/SwiGLU graph bucket, separately for each round and rank."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean, median

GMM = "GroupedMatmul"
GATE_UP = "128,2048;128,2048,1024;;;;;;128"
DOWN = "128,512;128,512,2048;;;;;;128"
MID = "128,1024"
ACT = "128,512"
BF16 = "DT_BF16"
RECORD = (
    Path(__file__).resolve().parents[2]
    / "docs/optimization-history/gmm1-swiglu-screen.json"
)


def load(path):
    return json.loads(path.read_bytes())


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def pairs(rows, rank, stage):
    streams = defaultdict(list)
    for r in rows:
        if r["Device_id"] != rank:
            raise ValueError("rank mismatch")
        streams[r["Stream ID"]].append(r)
    up = []
    act = []
    gaps = []
    sites = Counter()
    for stream in streams.values():
        stream.sort(key=lambda r: Decimal(r["Start Time(us)"].strip()))
        for i, x in enumerate(stream):
            if x["Type"] != GMM or x["Input Shapes"].strip('"') != GATE_UP:
                continue
            if x["Model ID"] == "4294967295":
                continue  # eager is not this cohort
            if i == 0 or i + 2 >= len(stream):
                raise ValueError("partial chain")
            route = stream[i - 1]
            y, z = stream[i + 1 : i + 3]
            task = int(x["Task ID"])
            model = x["Model ID"]
            if not (
                route["Type"] == "MoeInitRoutingV3"
                and all(r["Model ID"] == model for r in (route, y, z))
                and x["Input Data Types"].split(";")[:2] == [BF16, BF16]
                and x["Input Data Types"].split(";")[-1] == "INT64"
                and route["Output Shapes"].strip('"').split(";")[0] == "128,2048"
                and x["Output Shapes"].strip('"') == y["Input Shapes"].strip('"') == MID
                and x["Output Data Types"]
                == y["Input Data Types"]
                == y["Output Data Types"]
                == BF16
                and y["Type"] == "SwiGlu"
                and y["Output Shapes"].strip('"') == ACT
                and z["Type"] == GMM
                and z["Input Shapes"].strip('"') == DOWN
                and z["Output Shapes"].strip('"') == "128,2048"
                and z["Input Data Types"].split(";")[:2] == [BF16, BF16]
                and int(y["Task ID"]) == task + 1
                and int(z["Task ID"]) > task + 1
            ):
                raise ValueError("GMM route, task order, dtype or shape mismatch")
            u = Decimal(x["Duration(us)"])
            v = Decimal(y["Duration(us)"])
            gap = (
                Decimal(y["Start Time(us)"].strip())
                - Decimal(x["Start Time(us)"].strip())
                - u
            )
            if min(u, v) <= 0 or gap < 0:
                raise ValueError("invalid serial device times")
            up.append(u)
            act.append(v)
            gaps.append(gap)
            sites[model, x["Stream ID"], task] += 1
    if not sites or stage <= 0:
        raise ValueError("no matched graph calls or stage")
    group = sum(Decimal(x["Duration(us)"]) for x in rows if x["Type"] == GMM)

    def rounded(x):
        return round(float(x), 4)

    return {
        "matched": len(up),
        "graph_sites": len(sites),
        "repeats_per_site": sorted(set(sites.values())),
        "gmm1_mean_us": rounded(mean(up)),
        "swiglu_mean_us": rounded(mean(act)),
        "gap_median_us_not_attributed": rounded(median(gaps)),
        "gmm1_sum_us": rounded(sum(up)),
        "swiglu_sum_us": rounded(sum(act)),
        "stage_us": rounded(stage),
        "gmm1_stage_pct": rounded(100 * sum(up) / stage),
        "swiglu_stage_pct": rounded(100 * sum(act) / stage),
        "gmm1_share_all_groupedmatmul_us_pct": rounded(100 * sum(up) / group),
        "nominal_intermediate_write_plus_read_bytes_per_call": 2 * 128 * 1024 * 2,
    }


def screen(root, expected_hosts):
    files = {}

    def source(path):
        files[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return path

    manifest = load(source(root / "collection_manifest.json"))
    if manifest["rounds"] != 3 or manifest["host_commits"] != expected_hosts:
        raise ValueError("unexpected archive identity")
    result = {}
    for i in range(3):
        round_dir = root / f"round_{i:02d}_off"
        result[round_dir.name] = per_rank = {}
        exit_ = load(source(round_dir / "exit.json"))
        inv = load(source(round_dir / "invocation.json"))
        if (
            exit_["collection_completed"] is not True
            or exit_["returncode"] != 0
            or inv["gate"] != "0"
        ):
            raise ValueError("incomplete or wrong arm")
        log = source(round_dir / "server.log").read_text()
        responses = load(source(round_dir / "profile_responses.json"))
        if (
            "quantization=None" not in log
            or "enable_return_routed_experts=False" not in log
            or len(responses) != 16
        ):
            raise ValueError("unexpected routing/quantization context")
        if any(
            c["routed_experts"] is not None for r in responses for c in r["choices"]
        ):
            raise ValueError("routing data present: inspect before declaring absent")
        for detail in sorted((round_dir / "profile").rglob("kernel_details.csv")):
            worker = detail.parent.parent
            (info,) = worker.glob("profiler_info_*.json")
            meta = load(source(info))
            rank = str(meta["rank_id"])
            if rank in per_rank or f"_rank{rank}_" not in worker.name:
                raise ValueError("ambiguous worker")
            if meta["config"]["experimental_config"]["record_op_args"]:
                raise ValueError("reinspect recorded op arguments")
            (step,) = csv_rows(source(detail.with_name("step_trace_time.csv")))
            if step["Device_id"] != rank:
                raise ValueError("stage belongs to another rank")
            facts = pairs(csv_rows(source(detail)), rank, Decimal(step["Stage"]))
            if (facts["graph_sites"], facts["repeats_per_site"]) != (40, [15]):
                raise ValueError("unexpected site or replay coverage")
            per_rank[rank] = facts
        if set(per_rank) != {"0", "1"}:
            raise ValueError("incomplete rank coverage")
    return {"per_round_rank": result, "sha256": files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()
    record = load(RECORD)
    data = screen(args.archive, record["host_commits"])
    if args.print:
        print(json.dumps(data, indent=2))
    elif data != record["observations"]:
        raise SystemExit("recorded observations differ from the archive")
    else:
        print(
            "GMM1/SwiGLU: three rounds and both ranks verified (no expert-count claim)"
        )
