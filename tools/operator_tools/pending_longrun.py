"""Build a resumable serving plan for operators that still lack service evidence.

The plan never promotes results into repository evidence: every runnable entry
uses ``sync_mode: none``. Entries without a verified callsite or attribution
rule stay skipped or guard-only instead of producing a false A/B conclusion.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools.serving_e2e_auto import validate_plan  # noqa: E402

DEFAULT_PENDING_OPS = (
    "moe_epilogue",
    "gdn_qkv_norm",
    "gdn_split_ba",
    "moe_down_residual_norm",
    "ffn_activation",
)

SPECS: dict[str, dict[str, Any]] = {
    "moe_epilogue": {
        "mode": "full",
        "gate": "VLLM_HUST_MOE_EPILOGUE",
        "kernel": "_gate_mul_kernel",
        "attention_layers": 40,
    },
    "gdn_qkv_norm": {
        "mode": "skip",
        "reason": (
            "索引 gate_env=unexposed，历史服务门控已撤回；缺公开调用点和归因规则，"
            "不能安排有效服务 A/B"
        ),
    },
    "gdn_split_ba": {
        "mode": "full",
        "gate": "VLLM_HUST_GDN_SPLIT_BA",
        "kernel": "gdn_split_ba_2d_kernel",
        "attention_layers": 30,
    },
    "moe_down_residual_norm": {
        "mode": "guard_only",
        "gate": "VLLM_HUST_MOE_DOWN_RESIDUAL_NORM",
        "reason": "缺可核对的逐卡归因规则，只做服务内数值护栏，不形成服务 A/B 结论",
    },
    "ffn_activation": {
        "mode": "guard_only",
        "gate": "VLLM_HUST_FFN_ACTIVATION",
        "reason": "缺可核对的逐卡归因规则，只做服务内数值护栏，不形成服务 A/B 结论",
    },
}

RESET_KEYS = (
    "attribution",
    "attributions",
    "guard_only",
    "guard_and_l4_only",
    "sync_mode",
    "promote_combination_primary",
)


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def base_config(template: dict[str, Any]) -> dict[str, Any]:
    entries = template.get("items") or template.get("operators") or []
    for entry in entries:
        config = entry.get("config") if isinstance(entry, dict) else None
        if isinstance(config, dict) and config.get("host_commits"):
            return copy.deepcopy(config)
    raise ValueError("template has no config with host_commits")


def find_template(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    env = __import__("os").environ.get("SERVING_BASE_PLAN")
    if env:
        return Path(env)
    candidates = sorted(
        (ROOT / ".openbitfun").glob("npu-retest-*/attributed-serving-plan.json"),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "no base serving plan found; pass --template or set SERVING_BASE_PLAN"
        )
    return candidates[0]


def parse_modes(values: list[str]) -> dict[str, str]:
    modes: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"mode expects <operator>=<mode>, got {value!r}")
        op, mode = value.split("=", 1)
        if mode not in {"full", "guard_only", "skip"}:
            raise ValueError(f"unsupported mode {mode!r} for {op!r}")
        modes[op] = mode
    return modes


def build_plan(
    template: dict[str, Any],
    operators: tuple[str, ...] = DEFAULT_PENDING_OPS,
    modes: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    modes = dict(modes or {})
    config_template = base_config(template)
    for key in RESET_KEYS:
        config_template.pop(key, None)
    config_template["sync_mode"] = "none"
    config_template["phase"] = "decode"
    config_template["world_size"] = int(config_template.get("world_size", 2))
    config_template["startup_timeout"] = int(
        config_template.get("startup_timeout", 1200)
    )

    items: list[dict[str, Any]] = []
    report: list[dict[str, str]] = []
    seen: set[str] = set()
    for op in operators:
        if op in seen:
            raise ValueError(f"duplicate operator: {op}")
        seen.add(op)
        spec = SPECS.get(op)
        if spec is None:
            raise ValueError(f"no pending spec for {op!r}")
        mode = modes.get(op, spec["mode"])
        if mode == "skip":
            reason = spec.get("reason") or f"{op} 已被显式跳过，未配置有效服务门控"
            items.append({"id": op, "reason": reason})
            report.append({"operator": op, "mode": "skip", "detail": reason})
            continue
        config = copy.deepcopy(config_template)
        config["gate"] = spec["gate"]
        detail = spec.get("reason", "")
        if mode == "guard_only":
            config["guard_only"] = True
        else:
            config["guard_only"] = False
            if not spec.get("kernel") or not spec.get("attention_layers"):
                raise ValueError(f"{op!r} has no auditable kernel rule")
            config["kernel"] = spec["kernel"]
            config["attention_layers"] = spec["attention_layers"]
        items.append({"id": op, "config": config})
        report.append(
            {
                "operator": op,
                "mode": mode,
                "gate": config["gate"],
                "kernel": config.get("kernel", ""),
                "detail": detail,
            }
        )

    plan = {
        "timeout_seconds": int(template.get("timeout_seconds", 14400)),
        "operators": items,
    }
    validate_plan(plan)
    return plan, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, help="base serving plan JSON")
    parser.add_argument("--plan", type=Path, required=True, help="output plan path")
    parser.add_argument("--report", type=Path, help="output readiness report path")
    parser.add_argument(
        "--operator",
        action="append",
        dest="operators",
        help="override the default pending operator list",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="override one mode: <operator>=full|guard_only|skip",
    )
    args = parser.parse_args()

    template_path = find_template(args.template)
    operators = tuple(args.operators) if args.operators else DEFAULT_PENDING_OPS
    plan, report = build_plan(
        load_plan(template_path), operators, parse_modes(args.mode)
    )
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "template": str(template_path),
                "plan": str(args.plan),
                "operators": len(plan["operators"]),
                "report": report,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
