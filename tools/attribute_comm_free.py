# SPDX-License-Identifier: Apache-2.0

import argparse
import bisect
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_kernels(path):
    kernels, communication = [], {}
    with open(path, newline="") as source:
        for row in csv.DictReader(source):
            start = int(Decimal(row["Start Time(us)"].strip()) * 1000)
            duration = int(Decimal(row["Duration(us)"]) * 1000)
            name = row["Name"]
            role = (
                "COMM"
                if row["Accelerator Core"] == "COMMUNICATION"
                else row["Stream ID"]
            )
            kernel = (start, start + duration, role, name)
            kernels.append(kernel)
            if role == "COMM":
                if name in communication:
                    raise ValueError(f"duplicate communication name: {name}")
                communication[name] = kernel
    if not kernels or not communication:
        raise ValueError("missing kernel rows or communication rows")
    return kernels, communication


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def intersection_ns(left, right):
    left, right = merged(left), merged(right)
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def stream_pairs(free, kernels, scope):
    by_start = sorted(kernels)
    by_end = sorted(kernels, key=lambda k: (k[1], k[0]))
    starts = [k[0] for k in by_start]
    ends = [k[1] for k in by_end]
    pairs = defaultdict(int)
    unpaired = total = 0
    for a, b in free:
        a, b = max(a, scope[0]), min(b, scope[1])
        if b <= a:
            continue
        total += b - a
        prev = bisect.bisect_right(ends, a) - 1
        next_ = bisect.bisect_left(starts, b)
        if prev < 0 or next_ == len(starts):
            unpaired += b - a
        else:
            pairs[f"{by_end[prev][2]}_to_{by_start[next_][2]}"] += b - a
    assert sum(pairs.values()) + unpaired == total
    return (
        {key: round(value / 1e6, 6) for key, value in sorted(pairs.items())},
        round(unpaired / 1e6, 6),
        round(total / 1e6, 6),
    )


def analyze(db, rank0, rank1, cohort, opstat0, opstat1, config):
    inputs = [db, rank0, rank1, cohort, opstat0, opstat1, config]
    sources = [{"path": str(path), "sha256": digest(path)} for path in inputs]
    metadata = json.loads(Path(cohort).read_text())
    if (metadata.get("model"), metadata.get("precision")) != (
        "Qwen3.5-35B-A3B",
        "bf16",
    ):
        raise ValueError("unexpected cohort")
    if metadata.get("serve", {}).get("tp") != 2:
        raise ValueError("expected two ranks")
    width = json.loads(Path(config).read_text())["text_config"]["hidden_size"]
    stats = [json.loads(Path(p).read_text()) for p in (opstat0, opstat1)]
    kernels0, comm0 = read_kernels(rank0)
    kernels1, comm1 = read_kernels(rank1)
    if list(comm0) != list(comm1):
        raise ValueError("rank communication name/order mismatch")
    summaries = []
    for i, kernels in enumerate((kernels0, kernels1)):
        device_ms = sum(b - a for a, b, _, _ in kernels) / 1e6
        observed_ms = stats[i]["device_ms"]["all_kernel_ms"]
        if abs(device_ms - observed_ms) > 0.003:
            raise ValueError(
                f"rank {i} all-kernel mismatch {device_ms} != {observed_ms}"
            )
        summaries.append({"rank": i, "all_kernel_ms": round(device_ms, 6)})
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        types = dict(con.execute("SELECT id,name FROM ENUM_HCCL_DATA_TYPE"))
        ar = list(
            con.execute(
                "SELECT o.startNs,o.endNs,o.count,o.dataType,o.rankSize,s.value "
                "FROM COMMUNICATION_OP o JOIN STRING_IDS s ON s.id=o.opName "
                "WHERE s.value LIKE 'hcom_allReduce%' ORDER BY o.startNs"
            )
        )
        csv_ar_names = [n for n in comm0 if n.startswith("hcom_allReduce")]
        if not ar or len(ar) != len(csv_ar_names):
            raise ValueError("rank0 all-reduce DB/CSV cardinality mismatch")
        if [row[5] for row in ar] != csv_ar_names:
            raise ValueError("rank0 all-reduce DB/CSV order mismatch")
        dtype_set = {types[row[3]] for row in ar}
        rank_sizes = {row[4] for row in ar}
        if dtype_set != {"BFP16"} or rank_sizes != {2}:
            raise ValueError("unexpected communication dtype/rank size")
        overlap = {}
        for kind in (0, 1, 2, 3):
            overlap[kind] = list(
                con.execute(
                    "SELECT startNs,endNs FROM OVERLAP_ANALYSIS "
                    "WHERE type=? ORDER BY startNs",
                    (kind,),
                )
            )
        comp, all_comm, not_overlapped, free = (overlap[kind] for kind in (0, 1, 2, 3))
        starts = [
            r[0]
            for r in con.execute(
                "SELECT c.startNs FROM CANN_API c JOIN STRING_IDS s ON c.name=s.id "
                "WHERE s.value='aclmdlRIExecuteAsync' ORDER BY c.startNs"
            )
        ]
        if len(starts) < 2 or all_comm != not_overlapped:
            raise ValueError("missing execute markers or communication overlap changed")
        if len(all_comm) != len(comm0) or not free:
            raise ValueError("rank0 overlap rows do not match communication/Free")
        allreduce_iv = [(r[0], r[1]) for r in ar]
        overlap_ns = intersection_ns(allreduce_iv, comp)
        span = (starts[0], starts[-1])
        pairs, unpaired, span_free = stream_pairs(free, kernels0, span)
        full_scope = (min(a for a, _ in free), max(b for _, b in free))
        full_pairs, full_unpaired, full_free = stream_pairs(free, kernels0, full_scope)
        if abs(full_free - stats[0]["step_budget"]["free_ms"]) > 0.003:
            raise ValueError("rank0 FREE does not match step budget")
        host_api = [
            (int(a), int(b))
            for a, b in con.execute("SELECT startNs,endNs FROM CANN_API")
        ]
        host_api.extend(
            (int(a), int(b))
            for a, b in con.execute("SELECT startNs,endNs FROM PYTORCH_API")
        )
        temporal_ms = intersection_ns(free, host_api) / 1e6
        api = list(
            con.execute(
                "SELECT p.inputShapes,p.callchainId FROM PYTORCH_API p "
                "JOIN STRING_IDS s ON p.name=s.id "
                "WHERE s.value LIKE '%HcclAllreduce%'"
            )
        )
    finally:
        con.close()
    payload = defaultdict(lambda: {"calls": 0, "rank0_ms": 0.0, "rank1_ms": 0.0})
    for _, _, count, _, _, name in ar:
        row = payload[str(count)]
        row["calls"] += 1
        row["rank0_ms"] += (comm0[name][1] - comm0[name][0]) / 1e6
        row["rank1_ms"] += (comm1[name][1] - comm1[name][0]) / 1e6
    for count, row in payload.items():
        row.update(
            token_equivalent=int(count) // width if int(count) % width == 0 else None,
            rank0_ms=round(row["rank0_ms"], 6),
            rank1_ms=round(row["rank1_ms"], 6),
        )
    for i, comm in enumerate((comm0, comm1)):
        comm_ms = sum(k[1] - k[0] for k in comm.values()) / 1e6
        if abs(comm_ms - stats[i]["device_ms"]["comm_ms"]) > 0.003:
            raise ValueError(f"rank {i} communication total mismatch")
        reduce_ms = (
            sum(
                k[1] - k[0]
                for name, k in comm.items()
                if name.startswith("hcom_allReduce")
            )
            / 1e6
        )
        summaries[i].update(comm_ms=round(comm_ms, 6), allreduce_ms=round(reduce_ms, 6))
    coverage = stats[0]["load_caliber"]["window_coverage"]
    in_span = sum(span[0] <= start < span[1] for start, *_ in ar)
    return {
        "schema": "comm-free-paired-forensics.v2",
        "cohort": {
            "model": metadata["model"],
            "precision": metadata["precision"],
            "tp": 2,
            "graph": metadata.get("graph"),
            "carriers": metadata.get("carriers"),
            "hidden_size": width,
        },
        "coverage": {
            key: coverage.get(key)
            for key in (
                "prefill_token_volume",
                "window_iterations",
                "captured_request_count",
                "truncated",
                "prefill_iterations",
                "decoded_iterations",
            )
        },
        "rank_pair": {
            "communication_name_sequence_equal": True,
            "all_comm_calls": len(comm0),
            "allreduce_calls": len(ar),
            "totals": summaries,
        },
        "allreduce": {
            "dtype": "BFP16",
            "payload_by_element_count": dict(
                sorted(payload.items(), key=lambda item: int(item[0]))
            ),
            "nonoverlap_rows_equal": len(all_comm),
            "compute_overlap_ms_rank0": round(overlap_ns / 1e6, 6),
            "execute_start_span_calls": in_span,
            "outside_execute_start_span_calls": len(ar) - in_span,
            "host_allreduce_api_rows_rank0": len(api),
            "host_api_rows_with_shapes": sum(sh is not None for sh, _ in api),
            "host_api_rows_with_callchain": sum(cc is not None for _, cc in api),
            "phase_note": (
                "16-token calls repeat 19*81 times; other token-equivalent "
                "groups repeat 13*81 times. This matches recorded 19 decode/13 "
                "prefill iterations, but per-call phase is only inferred."
            ),
            "callsite_note": (
                "81 collectives per iteration; without a recorded layer index "
                "or stack, graph nodes cannot be mapped to exact callsites."
            ),
        },
        "free_rank0": {
            "total_ms": full_free,
            "host_api_temporal_intersection_ms": round(temporal_ms, 6),
            "all_window_adjacent_stream_pair_ms": full_pairs,
            "all_window_unpaired_ms": full_unpaired,
            "execute_start_span_ms": round((span[1] - span[0]) / 1e6, 6),
            "execute_start_span_free_ms": span_free,
            "execute_start_span_adjacent_stream_pair_ms": pairs,
            "execute_start_span_unpaired_ms": unpaired,
            "scope_note": (
                "The execute-start span is not a proven complete graph replay "
                "or a phase boundary. Stream adjacency and host API "
                "intersection describe time, not causal host or communication "
                "attribution. Rank1 lacks a parsed DB for Free classification."
            ),
        },
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser()
    for key in (
        "db",
        "rank0",
        "rank1",
        "cohort",
        "opstat0",
        "opstat1",
        "config",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    doc = analyze(
        args.db,
        args.rank0,
        args.rank1,
        args.cohort,
        args.opstat0,
        args.opstat1,
        args.config,
    )
    with args.output.open("x") as target:
        json.dump(doc, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rank_pair": doc["rank_pair"],
                "free_rank0": {
                    key: value
                    for key, value in doc["free_rank0"].items()
                    if key.endswith("_ms") and not isinstance(value, dict)
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
