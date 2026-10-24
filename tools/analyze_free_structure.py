#!/usr/bin/env python3
"""把窗口的 free 时间拆成「谁在等」：段结构 + 段内间隙 + 批量形状。纯离线，不占卡。

动机：`step_trace_time.csv` 只给整窗一行聚合（computing / comm / free），
无法回答"free 是 host 下发、是等待、还是空泡"。本工具从 `kernel_details.csv`
的 kernel 时间线重建：

  1. 段结构   按间隙阈值切段，逐段给出 span / busy / free / kernel 数 / 段内间隙中位数
  2. 间隙分布 按 <2us / 2-10us / 10-100us / 0.1-1ms / >1ms 分桶给出时长占比
  3. 批量事实 按 M 维统计 MatMul 输入形状，区分 prefill（M>1）与 decode（M=1）

判读约定（2026-10-24 定）：
  - 段内间隙中位数 ~1 us  → 该段在图中重放，host 不是瓶颈
  - 段内间隙中位数 ≫10 us → 该段未进图，host 下发/依赖等待占主导
  - 不同段的 free 不可混算："整窗 free 占比"会被 prefill/warmup 段拉高

用法:
  tools/analyze_free_structure.py <window_dir|raw_window_dir> [--gap-us N] [--json OUT]
  未给路径时取 /root/prof 下最新的 ladder-* 窗口。
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys

RAW_ROOT = "/root/包体恢复/records/ladder/raw"
BUCKETS = (
    ("<2us", 0, 2),
    ("2-10us", 2, 10),
    ("10-100us", 10, 100),
    ("0.1-1ms", 100, 1000),
    (">1ms", 1000, float("inf")),
)


def find_kernel_csv(path: str) -> str:
    cands = glob.glob(os.path.join(path, "**", "kernel_details.csv"), recursive=True)
    if not cands:
        raise SystemExit(f"没找到 kernel_details.csv: {path}")
    return cands[0]


def load_rows(csv_path: str) -> list[tuple[float, float, str, str]]:
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                start = float(r["Start Time(us)"])
                dur = float(r["Duration(us)"])
            except (KeyError, ValueError):
                continue
            rows.append((start, dur, r.get("Name", ""), r.get("Input Shapes", "")))
    rows.sort()
    if not rows:
        raise SystemExit(f"kernel 行为空: {csv_path}")
    return rows


def bucket_of(gap_us: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= gap_us < hi:
            return name
    return ">1ms"


def segment(rows, gap_us: float):
    segs, cur = [], [rows[0]]
    for i in range(len(rows) - 1):
        gap = rows[i + 1][0] - (rows[i][0] + rows[i][1])
        if gap > gap_us:
            segs.append((cur, gap))
            cur = [rows[i + 1]]
        else:
            cur.append(rows[i + 1])
    segs.append((cur, 0.0))
    return segs


def seg_stats(kern, gap):
    span = (kern[-1][0] + kern[-1][1] - kern[0][0]) / 1000.0
    busy = sum(d for _, d, _, _ in kern) / 1000.0
    inner = [kern[i + 1][0] - (kern[i][0] + kern[i][1]) for i in range(len(kern) - 1)]
    inner = [x for x in inner if x > 0]
    return {
        "t_start_ms": round((kern[0][0] - GLOBAL_T0) / 1000.0, 1),
        "span_ms": round(span, 2),
        "busy_ms": round(busy, 2),
        "free_ms": round(span - busy, 2),
        "kernels": len(kern),
        "inner_gap_median_us": round(statistics.median(inner), 2) if inner else 0.0,
        "inner_gap_gt100us": sum(1 for x in inner if x > 100),
        "gap_after_ms": round(gap / 1000.0, 2),
    }


def batch_profile(kern) -> dict:
    """以 MatMul 的 M 维区分 prefill / decode；并统计每段最大 M。"""
    ms = collections.Counter()
    for _, _, name, shapes in kern:
        if "Matmul" not in name and "MatMul" not in name:
            continue
        head = shapes.strip().strip('"').split(";")[0]
        try:
            m = int(head.split(",")[0])
        except (ValueError, IndexError):
            continue
        ms[m] += 1
    if not ms:
        return {}
    top = ms.most_common(3)
    return {"m_mode": top[0][0], "m_top": [[m, c] for m, c in top]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="")
    ap.add_argument(
        "--gap-us", type=float, default=3000.0, help="切段间隙阈值（默认 3000us）"
    )
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    path = args.path
    if not path:
        cands = sorted(glob.glob("/root/prof/ladder-*"), key=os.path.getmtime)
        if not cands:
            raise SystemExit("未给路径且 /root/prof 下无窗口")
        path = cands[-1]

    global GLOBAL_T0
    csv_path = find_kernel_csv(path)
    rows = load_rows(csv_path)
    GLOBAL_T0 = rows[0][0]

    span = (rows[-1][0] + rows[-1][1] - rows[0][0]) / 1000.0
    busy = sum(d for _, d, _, _ in rows) / 1000.0

    gaps = []
    for i in range(len(rows) - 1):
        g = rows[i + 1][0] - (rows[i][0] + rows[i][1])
        if g > 0:
            gaps.append(g)
    bsum = collections.defaultdict(lambda: [0, 0.0])
    for g in gaps:
        b = bucket_of(g)
        bsum[b][0] += 1
        bsum[b][1] += g / 1000.0
    free_total = sum(v[1] for v in bsum.values())

    segs = segment(rows, args.gap_us)
    out = {
        "schema": "opt-ops-free-structure.v1",
        "window": os.path.basename(os.path.normpath(path)),
        "kernel_csv": csv_path,
        "totals": {
            "span_ms": round(span, 1),
            "busy_ms": round(busy, 1),
            "free_ms": round(free_total, 1),
            "free_pct": round(free_total / span * 100, 1) if span else 0,
        },
        "gap_buckets": {
            k: {
                "count": bsum[k][0],
                "ms": round(bsum[k][1], 1),
                "share_pct": round(bsum[k][1] / free_total * 100, 1)
                if free_total
                else 0,
            }
            for k, _, _ in BUCKETS
        },
        "segments": [],
    }
    for kern, gap in segs:
        st = seg_stats(kern, gap)
        st.update(batch_profile(kern))
        out["segments"].append(st)

    in_seg = sum(s["free_ms"] for s in out["segments"])
    out["totals"]["free_in_segments_ms"] = round(in_seg, 1)
    out["totals"]["free_between_segments_ms"] = round(free_total - in_seg, 1)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)

    t = out["totals"]
    print(
        f"窗口 {out['window']}: span={t['span_ms']}ms busy={t['busy_ms']}ms "
        f"free={t['free_ms']}ms({t['free_pct']}%)"
    )
    print(
        f"  段内 free={t['free_in_segments_ms']}ms  "
        f"段间空档={t['free_between_segments_ms']}ms"
    )
    print(f"\n{'间隙桶':<10}{'次数':>7}{'时长ms':>9}{'占free':>8}")
    for k, _, _ in BUCKETS:
        b = out["gap_buckets"][k]
        print(f"{k:<10}{b['count']:>7}{b['ms']:>9.1f}{b['share_pct']:>7.1f}%")
    print(
        f"\n{'#':>3}{'t_ms':>8}{'span':>8}{'busy':>8}{'free':>8}{'kern':>6}{'内隙中位us':>10}{'M':>5}{'段后空档':>9}"
    )
    for i, s in enumerate(out["segments"]):
        print(
            f"{i:>3}{s['t_start_ms']:>8.1f}{s['span_ms']:>8.1f}{s['busy_ms']:>8.1f}"
            f"{s['free_ms']:>8.1f}{s['kernels']:>6}{s['inner_gap_median_us']:>10.1f}"
            f"{str(s.get('m_mode', '-')):>5}{s['gap_after_ms']:>9.1f}"
        )
    return 0


GLOBAL_T0 = 0.0

if __name__ == "__main__":
    sys.exit(main())
