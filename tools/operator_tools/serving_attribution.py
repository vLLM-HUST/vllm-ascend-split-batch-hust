"""Shape-specific, per-device attribution for the supported host optimizations."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

GATES = {
    "gdn_conv_pack": "VLLM_HUST_D4_CONV_PACK",
    "gate_fp32": "VLLM_HUST_GATE_FP32",
    "gdn_merge_inproj": "VLLM_HUST_GDN_MERGE_INPROJ",
    "gdn_split_fusion": "VLLM_HUST_GDN_UNPACK_FUSED",
    "qwen_text_split_qkv_fuse": "VLLM_HUST_SPLIT_QKV_ROPE_FUSED",
}


def shapes(value):
    return tuple(
        tuple(int(d) for d in tensor.split(",") if d)
        for tensor in value.replace('"', "").replace(" ", "").split(";")
    )


def count(rows, kind, inputs=(), output=None, dtype=None, output_dtype=None):
    return sum(
        kind in row["Type"]
        and (not inputs or shapes(row["Input Shapes"])[: len(inputs)] == inputs)
        and (output is None or shapes(row["Output Shapes"])[0] == output)
        and (dtype is None or row["Input Data Types"].split(";")[0] == dtype)
        and (output_dtype is None or row.get("Output Data Types") == output_dtype)
        for row in rows
    )


def evaluate(operator, off, on, steps=16, gdn_layers=30, moe_layers=40):
    """Return explicit checks; missing observations always fail closed."""
    gdn_calls = steps * gdn_layers
    moe_calls = steps * moe_layers
    checks = {}

    def exact(name, actual, expected):
        checks[name] = {
            "actual": actual,
            "expected": expected,
            "passed": actual == expected,
        }

    if operator == "gdn_conv_pack":
        for arm, rows in (("off", off), ("on", on)):
            exact(f"{arm}_conv", count(rows, "CausalConv1d"), gdn_calls)
            exact(
                f"{arm}_weight_transpose",
                count(rows, "Transpose", ((4096, 4),), (4, 4096)),
                gdn_calls if arm == "off" else 0,
            )
    elif operator == "gate_fp32":
        for arm, rows in (("off", off), ("on", on)):
            exact(
                f"{arm}_router_weight_cast",
                count(rows, "Cast", ((256, 2048),), (256, 2048), "DT_BF16", "FLOAT"),
                moe_calls if arm == "off" else 0,
            )
            router = sum(
                row["Type"] == "MatMulV2"
                and len(shapes(row["Input Shapes"])) >= 2
                and shapes(row["Input Shapes"])[1] == (256, 2048)
                and row["Input Data Types"] == "FLOAT;FLOAT"
                for row in rows
            )
            exact(f"{arm}_router_matmul", router, moe_calls)
    elif operator == "gdn_merge_inproj":
        for arm, rows in (("off", off), ("on", on)):
            for width in (6144, 32, 6176):
                matched = sum(
                    row["Type"] == "MatMulV2"
                    and len(shapes(row["Input Shapes"])) >= 2
                    and shapes(row["Input Shapes"])[1] == (width, 2048)
                    and row["Input Data Types"] == "DT_BF16;DT_BF16"
                    for row in rows
                )
                expected = gdn_calls if (width == 6176) == (arm == "on") else 0
                exact(f"{arm}_projection_{width}", matched, expected)
    elif operator == "gdn_split_fusion":
        # Eager, one uncached multi-token prefill, then single-row decode fallback.
        for kernel in ("_unpack_kernel", "_rearrange_kernel"):
            exact(f"off_{kernel}", count(off, kernel), 0)
            exact(f"on_{kernel}", count(on, kernel), gdn_layers)
        for arm, rows in (("off", off), ("on", on)):
            exact(f"{arm}_conv", count(rows, "CausalConv1d"), gdn_calls)
    elif operator == "qwen_text_split_qkv_fuse":
        for arm, rows in (("off", off), ("on", on)):
            exact(
                f"{arm}_rope_kernel",
                count(rows, "split_qkv_rmsnorm_mrope_kernel"),
                0 if arm == "off" else steps * 10,
            )
    else:
        raise ValueError(f"Unsupported attribution: {operator}")
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def verify(config, output):
    operator = config["attribution"]
    if GATES.get(operator) != config["gate"]:
        raise ValueError("Attribution does not match the enabled gate")
    model = json.loads((Path(config["model_path"]) / "config.json").read_text())
    text = model.get("text_config", model)
    if (
        text.get("hidden_size") != 2048
        or text.get("num_experts") != 256
        or text.get("layer_types", []).count("linear_attention") != 30
        or text.get("layer_types", []).count("full_attention") != 10
        or config["world_size"] != 2
        or not {"--enforce-eager", "--no-enable-prefix-caching"}.issubset(
            config.get("guard_server_args", [])
        )
    ):
        raise ValueError(
            "Dedicated attribution requires the audited model and guard mode"
        )
    arms = {}
    sources = []
    for arm in ("off", "on"):
        devices = {}
        for path in sorted(
            (output / f"guard_{arm}" / "profile").rglob("kernel_details.csv")
        ):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            ids = {row["Device_id"] for row in rows}
            if len(ids) != 1:
                raise ValueError("Expected one device per kernel details file")
            device = ids.pop()
            if device in devices:
                raise ValueError("Duplicate device profile")
            devices[device] = rows
            sources.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        if len(devices) != config["world_size"]:
            raise ValueError("Incomplete per-device kernel details")
        arms[arm] = devices
    if arms["off"].keys() != arms["on"].keys():
        raise ValueError("OFF/ON device sets differ")
    reports = {
        device: evaluate(operator, arms["off"][device], arms["on"][device])
        for device in arms["off"]
    }
    return {
        "passed": all(r["passed"] for r in reports.values()),
        "operator": operator,
        "protocol": "single-request-16-tokens",
        "devices": reports,
        "sources": sources,
    }
