#!/usr/bin/env python3
"""Hierarchical timeline subgraph compressor and SVG swimlane renderer.

Solves the high token cost and flat-table blindness of raw profiler CSVs by:
1. Segmenting `kernel_details.csv` around heavy anchor ops (`AI_CORE`,
   `MIX_AIC`, `HCCL`) to extract contiguous `AI_VECTOR_CORE` sandwich chains.
2. Folding repeated cross-layer patterns (e.g. 30 GDN layers or 40 MoE layers)
   into concise Top-K fusable subgraph cards with shapes, wait bubbles, and
   HBM memory-bound ratios (`aiv_mte2_ratio`).
3. Rendering a self-contained MindStudio-style SVG swimlane figure for human or
   multimodal-LLM batch review.
"""

from __future__ import annotations

import csv
import html
import os
from collections import defaultdict
from typing import Any

ANCHOR_CORE_TYPES = {"AI_CORE", "MIX_AIC", "HCCL"}
ANCHOR_TYPE_KEYWORDS = (
    "MatMul",
    "GroupedMatmul",
    "RecurrentGatedDeltaRule",
    "CausalConv1d",
    "FusedInferAttentionScore",
    "AllReduce",
    "ReduceScatter",
    "AllGather",
    "FlashAttention",
    "PagedAttention",
    "hcom_",
)


def _is_anchor_row(core_type: str, op_type: str, op_name: str) -> bool:
    if core_type in ANCHOR_CORE_TYPES:
        return True
    combined = f"{op_type} {op_name}"
    return any(kw in combined for kw in ANCHOR_TYPE_KEYWORDS)


def _safe_float(val: str | None, default: float = 0.0) -> float:
    if not val:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _clean_shape(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.strip().strip('"').replace('""', "")


def compress_kernel_timeline(
    csv_path: str,
    top_k: int = 8,
    min_chain_len: int = 2,
    max_chain_len: int = 12,
    max_rows_to_scan: int = 40000,
) -> dict[str, Any]:
    """Fold raw `kernel_details.csv` into Top-K repeated vector sandwich chains."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"kernel_details.csv not found: {csv_path}")

    events: list[dict[str, Any]] = []
    total_device_us = 0.0
    total_vector_us = 0.0

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= max_rows_to_scan:
                break
            op_name = (row.get("Name") or "").strip()
            op_type = (row.get("Type") or op_name).strip()
            core = (row.get("Accelerator Core") or "").strip()
            dur_us = _safe_float(row.get("Duration(us)"))
            wait_us = _safe_float(row.get("Wait Time(us)"))
            mte2_ratio = _safe_float(row.get("aiv_mte2_ratio"))
            mte3_ratio = _safe_float(row.get("aiv_mte3_ratio"))
            vec_ratio = _safe_float(row.get("aiv_vec_ratio"))
            in_shape = _clean_shape(row.get("Input Shapes"))
            in_dtype = _clean_shape(row.get("Input Data Types"))
            out_shape = _clean_shape(row.get("Output Shapes"))

            total_device_us += dur_us
            is_anchor = _is_anchor_row(core, op_type, op_name)
            if not is_anchor:
                total_vector_us += dur_us

            events.append(
                {
                    "name": op_name,
                    "type": op_type,
                    "core": core,
                    "is_anchor": is_anchor,
                    "dur_us": dur_us,
                    "wait_us": wait_us,
                    "mte_ratio": round(max(mte2_ratio, mte3_ratio), 3),
                    "vec_ratio": round(vec_ratio, 3),
                    "in_shape": in_shape,
                    "in_dtype": in_dtype,
                    "out_shape": out_shape,
                }
            )

    # Segment contiguous non-anchor chains bounded by anchors
    pattern_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    n_events = len(events)
    i = 0
    while i < n_events:
        if events[i]["is_anchor"]:
            i += 1
            continue
        start = i
        while i < n_events and not events[i]["is_anchor"]:
            i += 1
        end = i
        chain = events[start:end]
        if min_chain_len <= len(chain) <= max_chain_len:
            anchor_before = events[start - 1] if start > 0 else None
            anchor_after = events[end] if end < n_events else None
            sig = (
                anchor_before["type"] if anchor_before else "START",
                *[op["type"] for op in chain],
                anchor_after["type"] if anchor_after else "END",
            )
            pattern_groups[sig].append(
                {
                    "anchor_before": anchor_before,
                    "chain": chain,
                    "anchor_after": anchor_after,
                }
            )

    cards: list[dict[str, Any]] = []
    for sig, occurrences in pattern_groups.items():
        count = len(occurrences)
        chain_durs = [sum(op["dur_us"] for op in occ["chain"]) for occ in occurrences]
        chain_waits = [
            sum(min(op["wait_us"], 100.0) for op in occ["chain"]) for occ in occurrences
        ]
        total_chain_us = sum(chain_durs)
        avg_chain_us = total_chain_us / count
        avg_wait_us = sum(chain_waits) / count

        rep = occurrences[len(occurrences) // 2]
        ops_summary = []
        for pos, op_template in enumerate(rep["chain"]):
            avg_op_dur = sum(occ["chain"][pos]["dur_us"] for occ in occurrences) / count
            avg_mte = sum(occ["chain"][pos]["mte_ratio"] for occ in occurrences) / count
            ops_summary.append(
                {
                    "type": op_template["type"],
                    "name": op_template["name"],
                    "avg_dur_us": round(avg_op_dur, 2),
                    "in_shape": op_template["in_shape"],
                    "in_dtype": op_template["in_dtype"],
                    "out_shape": op_template["out_shape"],
                    "mte_ratio": round(avg_mte, 2),
                    "bottleneck": (
                        "memory_bound" if avg_mte >= 0.4 else "compute_or_launch"
                    ),
                }
            )

        a_before = rep["anchor_before"]
        a_after = rep["anchor_after"]
        cards.append(
            {
                "pattern_signature": " -> ".join(sig[1:-1]),
                "anchor_before": {
                    "type": a_before["type"] if a_before else "START",
                    "in_shape": a_before["in_shape"] if a_before else "",
                    "out_shape": a_before["out_shape"] if a_before else "",
                    "dur_us": round(a_before["dur_us"], 2) if a_before else 0.0,
                },
                "anchor_after": {
                    "type": a_after["type"] if a_after else "END",
                    "in_shape": a_after["in_shape"] if a_after else "",
                    "out_shape": a_after["out_shape"] if a_after else "",
                    "dur_us": round(a_after["dur_us"], 2) if a_after else 0.0,
                },
                "occurrence_count": count,
                "num_ops_in_chain": len(ops_summary),
                "avg_chain_dur_us": round(avg_chain_us, 2),
                "avg_wait_bubble_us": round(avg_wait_us, 2),
                "total_window_dur_us": round(total_chain_us, 2),
                "share_of_vector_pct": (
                    round((total_chain_us / total_vector_us) * 100.0, 2)
                    if total_vector_us > 0
                    else 0.0
                ),
                "share_of_all_device_pct": (
                    round((total_chain_us / total_device_us) * 100.0, 2)
                    if total_device_us > 0
                    else 0.0
                ),
                "operators": ops_summary,
            }
        )

    cards.sort(key=lambda c: c["total_window_dur_us"], reverse=True)
    top_cards = cards[:top_k]
    for rank, card in enumerate(top_cards, 1):
        card["rank"] = rank

    return {
        "source_csv": csv_path,
        "scanned_events": n_events,
        "total_device_ms": round(total_device_us / 1000.0, 3),
        "total_vector_ms": round(total_vector_us / 1000.0, 3),
        "top_fusable_chains": top_cards,
    }


def format_insight_markdown(summary: dict[str, Any]) -> str:
    """Format compressed timeline cards into a compact Markdown digest for LLMs."""
    lines = [
        "# Hierarchical Timeline Subgraph Digest",
        "",
        f"- Scanned Kernel Events: `{summary['scanned_events']}`",
        f"- Total Device Time: `{summary['total_device_ms']} ms` "
        f"(Vector Core Subset: `{summary['total_vector_ms']} ms`)",
        "",
        "## Top Fusable Vector Sandwich Chains",
        "",
    ]
    for card in summary.get("top_fusable_chains", []):
        ab = card["anchor_before"]
        aa = card["anchor_after"]
        lines.append(
            f"### Rank #{card['rank']}: `{card['pattern_signature']}` "
            f"(repeats={card['occurrence_count']}x, "
            f"avg={card['avg_chain_dur_us']} us, "
            f"total={card['total_window_dur_us']} us, "
            f"vector_share={card['share_of_vector_pct']}%)"
        )
        lines.append(
            f"- **Anchor Before**: `{ab['type']}` in=`{ab['in_shape']}` "
            f"out=`{ab['out_shape']}` ({ab['dur_us']} us)"
        )
        for idx, op in enumerate(card["operators"], 1):
            lines.append(
                f"  {idx}. `{op['type']}` ({op['avg_dur_us']} us, "
                f"in=`{op['in_shape']}` {op['in_dtype']}, "
                f"mte={op['mte_ratio']}, {op['bottleneck']})"
            )
        lines.append(
            f"- **Anchor After**: `{aa['type']}` in=`{aa['in_shape']}` "
            f"out=`{aa['out_shape']}` ({aa['dur_us']} us)"
        )
        lines.append("")
    return "\n".join(lines)


def render_timeline_swimlane_svg(
    summary: dict[str, Any],
    output_svg_path: str,
) -> str:
    """Render a MindStudio-style swimlane SVG for Top-K fusable vector chains."""
    cards = summary.get("top_fusable_chains", [])
    row_height = 95
    header_height = 65
    width = 1040
    height = header_height + max(1, len(cards)) * row_height + 30

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="monospace, sans-serif">',
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        '<text x="20" y="28" fill="#f8fafc" font-size="16" font-weight="bold">'
        "Ascend NPU Timeline Subgraph Insight (Top Fusable Vector Sandwich Chains)"
        "</text>",
        f'<text x="20" y="48" fill="#94a3b8" font-size="12">'
        f"Scanned Events: {summary.get('scanned_events', 0)} | "
        f"Total Device: {summary.get('total_device_ms', 0)} ms | "
        f"Vector Subset: {summary.get('total_vector_ms', 0)} ms | "
        "Slate=Anchor(Cube/Comm), Orange=Memory-Bound Vector, Blue=Vector Compute"
        "</text>",
    ]

    track_x = 300
    track_w = 710

    for idx, card in enumerate(cards):
        y0 = header_height + idx * row_height
        svg_lines.append(
            f'<rect x="14" y="{y0}" width="{width - 28}" height="{row_height - 10}" '
            'rx="6" fill="#1e293b" stroke="#334155"/>'
        )
        title = html.escape(card["pattern_signature"][:36])
        meta = (
            f"#{card['rank']} ({card['occurrence_count']}x) "
            f"avg={card['avg_chain_dur_us']}us | "
            f"vec_share={card['share_of_vector_pct']}%"
        )
        svg_lines.append(
            f'<text x="24" y="{y0 + 26}" fill="#f1f5f9" font-size="12" '
            f'font-weight="bold">{html.escape(meta)}</text>'
        )
        svg_lines.append(
            f'<text x="24" y="{y0 + 46}" fill="#cbd5e1" font-size="11">{title}</text>'
        )
        ab_type = html.escape(card["anchor_before"]["type"][:18])
        aa_type = html.escape(card["anchor_after"]["type"][:18])
        svg_lines.append(
            f'<text x="24" y="{y0 + 66}" fill="#64748b" font-size="10">'
            f"[{ab_type}] -> VectorChain -> [{aa_type}]</text>"
        )

        # Draw swimlane bars proportional to duration
        ab_dur = max(card["anchor_before"]["dur_us"], 1.0)
        aa_dur = max(card["anchor_after"]["dur_us"], 1.0)
        ops = card["operators"]
        op_durs = [max(op["avg_dur_us"], 0.5) for op in ops]
        total_span = ab_dur + sum(op_durs) + aa_dur
        scale = (track_w - (len(ops) + 1) * 4) / max(total_span, 1e-3)

        cur_x = float(track_x)
        bar_y = y0 + 18
        bar_h = 48

        # Anchor Before
        w_ab = max(36.0, ab_dur * scale)
        svg_lines.append(
            f'<rect x="{cur_x:.1f}" y="{bar_y}" width="{w_ab:.1f}" height="{bar_h}" '
            'rx="4" fill="#475569"/>'
        )
        svg_lines.append(
            f'<text x="{cur_x + 4:.1f}" y="{bar_y + 22}" fill="#e2e8f0" '
            f'font-size="10">{ab_type[:10]}</text>'
        )
        svg_lines.append(
            f'<text x="{cur_x + 4:.1f}" y="{bar_y + 38}" fill="#94a3b8" '
            f'font-size="9">{card["anchor_before"]["dur_us"]}us</text>'
        )
        cur_x += w_ab + 4

        # Vector Chain Ops
        for op, d_val in zip(ops, op_durs, strict=True):
            w_op = max(42.0, d_val * scale)
            color = "#ea580c" if op["mte_ratio"] >= 0.4 else "#0284c7"
            op_label = html.escape(op["type"][:11])
            shape_label = html.escape(op["in_shape"][:12])
            svg_lines.append(
                f'<rect x="{cur_x:.1f}" y="{bar_y}" width="{w_op:.1f}" '
                f'height="{bar_h}" rx="4" fill="{color}"/>'
            )
            svg_lines.append(
                f'<text x="{cur_x + 4:.1f}" y="{bar_y + 18}" fill="#ffffff" '
                f'font-size="10" font-weight="bold">{op_label}</text>'
            )
            svg_lines.append(
                f'<text x="{cur_x + 4:.1f}" y="{bar_y + 32}" fill="#ffedd5" '
                f'font-size="9">{op["avg_dur_us"]}us</text>'
            )
            svg_lines.append(
                f'<text x="{cur_x + 4:.1f}" y="{bar_y + 44}" fill="#fed7aa" '
                f'font-size="8">{shape_label}</text>'
            )
            cur_x += w_op + 4

        # Anchor After
        w_aa = max(36.0, max(track_x + track_w - cur_x, 36.0))
        svg_lines.append(
            f'<rect x="{cur_x:.1f}" y="{bar_y}" width="{w_aa:.1f}" height="{bar_h}" '
            'rx="4" fill="#475569"/>'
        )
        svg_lines.append(
            f'<text x="{cur_x + 4:.1f}" y="{bar_y + 22}" fill="#e2e8f0" '
            f'font-size="10">{aa_type[:10]}</text>'
        )
        svg_lines.append(
            f'<text x="{cur_x + 4:.1f}" y="{bar_y + 38}" fill="#94a3b8" '
            f'font-size="9">{card["anchor_after"]["dur_us"]}us</text>'
        )

    svg_lines.append("</svg>")
    parent = os.path.dirname(os.path.abspath(output_svg_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines) + "\n")
    return output_svg_path
