#!/usr/bin/env python3
"""按 gmm-route-reconciliation.json 的字段表做纯 CPU 合成对账（fail closed）。

期望键完整枚举每个 in_cohort 步 × 每层 × 每个已登记物理 rank：零路由 rank
也必须有全零 counts 的实测记录；真实负载含 padding/warmup/capture 贡献时逐
槽位必然失配并被拒绝，没有抹除后重试的选项。只用标准库，不写仓内文件。
"""

import argparse
import json
import re
from pathlib import Path

_HEX = re.compile(r"[0-9a-f]{40}")


class Rejected(ValueError):
    """对账不唯一即抛出，不降级为警告。"""

    def __init__(self, code, **detail):
        super().__init__(code, detail)
        self.code, self.detail = code, detail


def _expect(ok, code, **detail):
    if not ok:
        raise Rejected(code, **detail)


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def reconcile(sample):
    """校验并连接合成样例；全部唯一时返回逐调用逐专家对账结果。"""
    window = sample["window"]
    window_id = window["window_id"]
    _expect(isinstance(window_id, str) and window_id, "bad_window", window_id=window_id)
    host = window.get("host_commits")
    ok = isinstance(host, dict) and set(host) == {"vllm", "vllm-ascend"}
    ok = ok and all(isinstance(v, str) and _HEX.fullmatch(v) for v in host.values())
    _expect(ok, "host_mismatch", host_commits=host)
    values = [window.get(k) for k in ("moe_layers", "num_experts", "top_k", "ep_size")]
    _expect(all(_int(v) and v >= 1 for v in values), "bad_window")
    moe_layers, num_experts, top_k, ep_size = values
    phases = window["in_cohort_phases"]
    _expect(isinstance(phases, list) and phases, "bad_window")
    _expect(not set(phases) & {"warmup", "capture"}, "phase_forbidden", phases=phases)
    _expect(sample["requests"] and sample["steps"], "empty_window")
    requests = {}
    for req in sample["requests"]:
        rid = req["request_id"]
        _expect(req.get("window_id") == window_id, "window_mismatch", request_id=rid)
        _expect(rid not in requests, "bad_request", duplicate_request_id=rid)
        tokens = (req["prompt_tokens"], req["completion_tokens"])
        _expect(all(_int(t) and t >= 1 for t in tokens), "bad_request", request_id=rid)
        routed = req["routed_experts"]
        ok = isinstance(routed, list) and len(routed) == tokens[0] + tokens[1] - 1
        ok = ok and all(isinstance(pl, list) and len(pl) == moe_layers for pl in routed)
        cells = [ids for pl in routed for ids in pl] if ok else []
        ok = ok and all(isinstance(ids, list) and len(ids) == top_k for ids in cells)
        ok = ok and all(_int(e) and 0 <= e < num_experts for ids in cells for e in ids)
        _expect(ok and all(len(set(ids)) == len(ids) for ids in cells), "bad_request")
        requests[rid] = req
    ep_map = sample["ep_map"]
    _expect(ep_map.get("window_id") == window_id, "window_mismatch")
    _expect(ep_map.get("host_commits") == host, "host_mismatch")
    ranks, owner = {}, {}
    for rank in ep_map["ranks"]:
        physical, local = rank["physical_rank"], rank["local_experts"]
        ok = _int(physical) and physical >= 0 and _int(local) and local >= 1
        ok = ok and physical not in ranks
        _expect(ok, "bad_ep_map", physical_rank=physical)
        slots = set()
        for key, slot in rank["log2phy"].items():
            ok = isinstance(key, str) and key.isdigit() and _int(slot)
            ok = ok and 0 <= slot < local
            _expect(ok and slot not in slots and int(key) not in owner, "bad_ep_map")
            owner[int(key)] = (physical, slot)
            slots.add(slot)
        ranks[physical] = local
    _expect(len(ranks) == ep_size, "bad_window", ep_size=ep_size)
    devices = window.get("rank_device")
    ok = isinstance(devices, dict)
    ok = ok and all(_int(v) and v >= 0 for v in devices.values())
    ok = ok and len(set(devices.values())) == len(devices)
    _expect(ok and set(devices) == {str(rank) for rank in ranks}, "bad_window")
    ok = set(owner) == set(range(num_experts))
    _expect(ok, "expert_unmapped", diff=sorted(set(owner) ^ set(range(num_experts))))
    steps = {}
    for step in sample["steps"]:
        sid = step["step_id"]
        _expect(step.get("window_id") == window_id, "window_mismatch", step_id=sid)
        _expect(_int(sid) and sid not in steps, "bad_window", step_id=sid)
        if step["phase"] not in phases:
            for row in step["rows"]:
                _expect(row.get("request_id") is None, "row_outside_cohort")
        steps[sid] = step
    cohort = {sid for sid, step in steps.items() if step["phase"] in phases}
    _expect(cohort, "empty_window")
    measured, excluded = {}, 0
    for record in sample["group_lists"]:
        _expect(record.get("window_id") == window_id, "window_mismatch")
        sid = record["step_id"]
        _expect(_int(sid), "bad_group_list", step_id=sid)
        _expect(sid in steps, "orphan_group_list", step_id=sid)
        if steps[sid]["phase"] not in phases:
            excluded += 1
            continue
        layer, rank, counts = record["layer"], record["physical_rank"], record["counts"]
        ok = _int(record["group_list_type"]) and record["group_list_type"] == 1
        ok = ok and _int(layer) and 0 <= layer < moe_layers
        ok = ok and _int(rank) and rank in ranks and isinstance(counts, list)
        ok = ok and len(counts) == ranks[rank]
        _expect(ok, "bad_group_list", step_id=sid)
        _expect(all(_int(c) and c >= 0 for c in counts), "bad_counts_value")
        key = (sid, layer, rank)
        _expect(key not in measured, "duplicate_call_record", call=str(key))
        measured[key] = list(counts)
    expected = {
        (sid, layer, physical): [0] * ranks[physical]
        for sid in sorted(cohort)
        for layer in range(moe_layers)
        for physical in ranks
    }
    covered, padded = {}, 0
    for sid in sorted(cohort):
        seen = set()
        for row in steps[sid]["rows"]:
            kind, rid = row.get("kind"), row.get("request_id")
            pos = row.get("token_position")
            if kind == "padded":
                _expect(rid is None and pos is None, "bad_row", step_id=sid)
                padded += 1
                continue
            _expect(kind == "forwarded" and rid is not None, "bad_row", step_id=sid)
            _expect(rid in requests, "unknown_request", step_id=sid, request_id=rid)
            req = requests[rid]
            last = req["prompt_tokens"] + req["completion_tokens"] - 2
            _expect(_int(pos) and 0 <= pos <= last, "row_out_of_range", pos=pos)
            token = (rid, pos)
            _expect(token not in seen and token not in covered, "duplicate_token")
            seen.add(token)
            covered[token] = sid
            for layer, ids in enumerate(req["routed_experts"][pos]):
                for expert in ids:
                    physical, slot = owner[expert]
                    expected[(sid, layer, physical)][slot] += 1
    _expect(covered, "empty_window", valid_tokens=0)
    trailing = {}
    for rid, req in requests.items():
        tail = req["prompt_tokens"] + req["completion_tokens"] - 1
        trailing[rid] = tail
        have = {pos for other, pos in covered if other == rid}
        _expect(have == set(range(tail)), "tokens_not_covered", request_id=rid)
    calls = {}
    for key in sorted(expected):
        _expect(key in measured, "missing_group_list", call=str(key))
        ok = expected[key] == measured[key]
        _expect(ok, "count_mismatch", call=str(key), expected=expected[key])
        calls[str(key)] = measured[key]
    return {
        "status": "reconciled",
        "calls": calls,
        "valid_tokens": len(covered),
        "padded_rows": padded,
        "out_of_cohort_steps": sorted(set(steps) - cohort),
        "out_of_cohort_group_lists": excluded,
        "trailing_not_forwarded": trailing,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="gmm-route-reconciliation.json 路径")
    args = parser.parse_args()
    record = json.loads(args.record.read_bytes())
    reject = record["samples"]["must_reject"]
    code = reject["expected_reject_code"]
    try:
        reconcile(record["samples"]["pass"])
    except (Rejected, ValueError, KeyError) as exc:
        parser.exit(1, f"gmm-route-reconciliation: pass sample rejected: {exc}\n")
    try:
        reconcile(reject["sample"])
        parser.exit(1, "gmm-route-reconciliation: must_reject sample was accepted\n")
    except Rejected as exc:
        if exc.code != code:
            parser.exit(
                1, f"gmm-route-reconciliation: ambiguity {exc.code} != {code}\n"
            )
    print(f"gmm-route-reconciliation: pass sample reconciled; rejected ({code})")


if __name__ == "__main__":
    main()
