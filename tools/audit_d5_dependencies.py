# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CONTRACTS = {
    "decoder": (
        "Qwen3NextDecoderLayer",
        (
            "hidden_states = self.linear_attn(hidden_states=hidden_states)",
            "hidden_states, residual = "
            "self.post_attention_layernorm(hidden_states, residual)",
            "hidden_states = self.mlp(hidden_states)",
        ),
    ),
    "moe": (
        "Qwen3NextSparseMoeBlock",
        (
            "hidden_states = hidden_states.view(-1, hidden_dim)",
            "final_hidden_states = self.experts("
            "hidden_states=hidden_states, router_logits=hidden_states)",
        ),
    ),
    "gdn": (
        "AscendGatedDeltaNetAttention",
        (
            "torch.ops.vllm.qwen_gdn_attention_core("
            "mixed_qkv, b, a, core_attn_out, self.prefix, False)",
            "core_attn_out = self.norm(core_attn_out, z)",
            "out, _ = self.out_proj(core_attn_out)",
            "return out",
        ),
    ),
}
TARGETS = {"GroupedMatmul", "RecurrentGatedDeltaRule", "fused_gdn_gating_kernel"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method(tree, class_name):
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )


def inspect_contract(source, class_name, expected):
    tree = ast.parse(source)
    forward = method(tree, class_name)
    statements = sorted(
        (n for n in ast.walk(forward) if isinstance(n, ast.stmt)),
        key=lambda n: n.lineno,
    )
    cursor = 0
    hits = []
    for text in expected:
        canonical = ast.unparse(ast.parse(text).body[0])
        found = next(
            (
                n
                for n in statements
                if n.lineno > cursor and ast.unparse(n) == canonical
            ),
            None,
        )
        if found is None:
            raise ValueError(f"unrecognized {class_name}.forward: {text}")
        cursor = found.end_lineno
        hits.append({"line": found.lineno, "end_line": cursor, "statement": canonical})
    return {
        "class": class_name,
        "method": "forward",
        "line": forward.lineno,
        "end_line": forward.end_lineno,
        "statements": hits,
        "source": ast.get_source_segment(source, forward),
    }


def inspect_inheritance(source):
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3_5DecoderLayer"
    )
    if [ast.unparse(b) for b in cls.bases] != ["Qwen3NextDecoderLayer"]:
        raise ValueError("decoder inheritance drift")
    if any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in cls.body):
        raise ValueError("decoder forward override requires manual review")
    return {"line": cls.lineno, "inherits_forward_from": "Qwen3NextDecoderLayer"}


def shape_summary(path):
    counts = Counter()
    with path.open() as f:
        reader = csv.DictReader(f)
        required = {"Type", "Input Shapes", "Input Data Types", "Input Formats"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("missing shape metadata")
        for row in reader:
            if row["Type"] in TARGETS:
                counts[tuple(row[k] for k in sorted(required))] += 1
    if {key[-1] for key in counts} != TARGETS:
        raise ValueError("missing target kernels")
    return [
        dict(zip(sorted(required), key, strict=True), count=n)
        for key, n in sorted(counts.items())
    ]


def audit(core, ascend, kernels):
    paths = {
        "next": core / "vllm/model_executor/models/qwen3_next.py",
        "qwen35": core / "vllm/model_executor/models/qwen3_5.py",
        "gdn": ascend / "vllm_ascend/ops/gdn.py",
    }
    sources = {key: path.read_text() for key, path in paths.items()}
    observed = {
        key: inspect_contract(sources["gdn" if key == "gdn" else "next"], cls, texts)
        for key, (cls, texts) in CONTRACTS.items()
    }
    return {
        "schema": "d5-dependency-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "static-source-observation-not-device-validation",
        "status": "SERIAL_CHAIN_OBSERVED",
        "scope": "same decode batch, same layer GDN -> routed-expert GMM",
        "limits": [
            "Ordered AST statement witnesses are not a general dataflow proof.",
            "Review full captured methods and runtime carriers before a decision.",
            "Does not rule out independent microbatches, MoE shared branches, "
            "or a redesigned algorithm; these are outside this work order.",
            "Retained baseline has one captured request, not saturated concurrency.",
            "group_list values/per-expert counts and packed weight layout "
            "are not reconstructed from tensor shapes or Input Formats.",
            "No runtime overlap, core quota support, L0, L3 or L4 A/B claim.",
        ],
        "sources": {
            key: {"path": str(path), "sha256": digest(path)}
            for key, path in paths.items()
        },
        "inheritance": inspect_inheritance(sources["qwen35"]),
        "observed": observed,
        "shape_source": {"path": str(kernels), "sha256": digest(kernels)},
        "shapes": shape_summary(kernels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--ascend", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite evidence")
    report = audit(args.core, args.ascend, args.kernels)
    with args.out.open("x") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"{report['status']}: {args.out}; static only, manual review required")


if __name__ == "__main__":
    main()
