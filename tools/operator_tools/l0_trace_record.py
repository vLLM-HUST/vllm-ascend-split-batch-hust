#!/usr/bin/env python3
"""把测量产物机械记录进 ops/<operator>/evidence.json 的 `l0_traces`。

数字只来自测量 harness 产出的结构化产物（计时 receipt、验证结果、
`bench_l0.py` 的执行输出），本脚本不改写、不推导任何数字；结论标签
（`serving_status`、`l3_numerical_guard`、`status` 等）一律不动。写入后
用 `custody.py` 重新生成目录指纹。失败用例照原样记录，不得丢弃。

用法：
    python tools/operator_tools/l0_trace_record.py <operator> \
        --receipt <计时 receipt 的 JSON 路径> [--verify-result <verify_result.json>]
    python tools/operator_tools/l0_trace_record.py <operator> --exec-bench

`--receipt` 记录一次 L0 计时 trace；`--exec-bench` 只执行 `bench_l0.py`
并把退出码机械入账（退出码 3 = 占位脚本、无测量）。`--dry-run` 只打印
将写入的记录，不落盘、不刷指纹。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPERATORS_DIR = ROOT / "ops"
CONTRACT_FILES = ("README.md", "REPORT.md", "bench_l0.py", "evidence.json")
STAT_KEYS = ("median_us", "p90_us", "p99_us")
STDOUT_KEEP_BYTES = 4096


def load_custody():
    """按文件路径加载 custody 工具（不依赖 PYTHONPATH）。"""
    path = Path(__file__).resolve().parent / "custody.py"
    spec = importlib.util.spec_from_file_location("operator_custody", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slim_stats(stats: dict) -> dict:
    missing = [key for key in STAT_KEYS if key not in stats]
    if missing:
        raise ValueError(f"计时统计缺少字段 {missing}: {sorted(stats)}")
    return {key: stats[key] for key in STAT_KEYS}


def _slim_case(case: dict) -> dict:
    for side in ("baseline", "candidate"):
        if side not in case:
            raise ValueError(f"用例缺少 {side} 计时: {sorted(case)}")
    slim = {
        "shape": case.get("shape"),
        "stride": case.get("stride"),
        "atol": case.get("atol"),
        "rtol": case.get("rtol"),
        "passed": case.get("passed"),
        "speedup": case.get("speedup"),
        "baseline": _slim_stats(case["baseline"]),
        "candidate": _slim_stats(case["candidate"]),
    }
    return {key: value for key, value in slim.items() if value is not None}


def build_receipt_trace(
    operator: str, receipt_path: Path, verify_path: Path | None, note: str | None
) -> dict:
    """从计时 receipt 构造一条 `l0_traces` 记录（不改写任何数字）。"""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("operator") != operator:
        raise ValueError(
            f"receipt 的 operator 是 {receipt.get('operator')!r}，"
            f"与目标算子 {operator!r} 不符"
        )
    cases = receipt.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("receipt 里没有 cases 计时记录")

    record: dict = {
        "kind": "l0_receipt_trace",
        "operator": operator,
        "scope": receipt.get("scope"),
        "source": {
            "receipt_path": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
        },
        "environment": {
            key: receipt.get(key)
            for key in (
                "torch",
                "torch_npu",
                "device",
                "seed",
                "warmup",
                "repetitions_per_graph",
                "samples",
                "timing",
            )
            if key in receipt
        },
        "receipt_argv": receipt.get("argv"),
        "receipt_script_sha256": receipt.get("script_sha256"),
        "cases": [_slim_case(case) for case in cases],
        "case_summary": {
            "total": len(cases),
            "passed": sum(1 for case in cases if case.get("passed") is True),
            "failed": sum(1 for case in cases if case.get("passed") is not True),
        },
    }
    if note:
        record["note"] = note
    if verify_path is not None:
        verify = json.loads(verify_path.read_text(encoding="utf-8"))
        record["verify_result"] = {
            "path": str(verify_path),
            "sha256": _sha256(verify_path),
            "op_name": verify.get("op_name"),
            "total_cases": verify.get("total_cases"),
            "passed_cases": verify.get("passed_cases"),
            "failed_cases": verify.get("failed_cases"),
        }
        if verify.get("op_name") != operator:
            raise ValueError(
                f"verify_result 的 op_name 是 {verify.get('op_name')!r}，"
                f"与目标算子 {operator!r} 不符"
            )
    return record


def build_bench_execution_trace(
    operator: str, bench_stdout: bytes, exit_code: int, note: str | None
) -> dict:
    """把 `bench_l0.py` 的执行结果机械入账（退出码 3 = 无测量）。"""
    record: dict = {
        "kind": "bench_l0_execution",
        "operator": operator,
        "exit_code": exit_code,
        "no_measurement": exit_code == 3,
        "stdout": bench_stdout.decode("utf-8", errors="replace")[:STDOUT_KEEP_BYTES],
        "bench_sha256": _sha256(OPERATORS_DIR / operator / "bench_l0.py"),
    }
    if note:
        record["note"] = note
    return record


def append_trace(operator: str, record: dict, dry_run: bool = False) -> str:
    """把一条记录追加进 evidence.json 的 `l0_traces`，并刷新目录指纹。"""
    op_dir = OPERATORS_DIR / operator
    missing = [name for name in CONTRACT_FILES if not (op_dir / name).is_file()]
    if missing:
        raise ValueError(f"ops/{operator} 缺少契约文件 {missing}，拒绝写入")
    evidence_path = op_dir / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("operator_name") not in (None, operator):
        raise ValueError(
            f"evidence.json 的 operator_name 是 {evidence.get('operator_name')!r}，"
            f"与目标算子 {operator!r} 不符"
        )

    traces = evidence.setdefault("l0_traces", [])
    traces.append(record)

    if dry_run:
        return json.dumps(record, ensure_ascii=False, indent=2)

    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    custody = load_custody()
    custody.write_dir_checksum(str(op_dir))
    ok, message = custody.verify_dir_checksum(str(op_dir))
    if not ok:
        raise ValueError(f"指纹刷新后校验失败: {message}")
    return message


def _run_bench(operator: str) -> tuple[bytes, int]:
    bench = OPERATORS_DIR / operator / "bench_l0.py"
    proc = subprocess.run(
        [sys.executable, str(bench)], capture_output=True, check=False
    )
    return proc.stdout, proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("operator", help="ops/ 下的算子目录名")
    parser.add_argument("--receipt", type=Path, help="计时 receipt 的 JSON 路径")
    parser.add_argument("--verify-result", type=Path, help="verify_result.json 路径")
    parser.add_argument(
        "--exec-bench", action="store_true", help="执行 bench_l0.py 并把退出码机械入账"
    )
    parser.add_argument("--note", help="随记录保存的备注（不承载度量数字）")
    parser.add_argument("--dry-run", action="store_true", help="只打印记录，不落盘")
    args = parser.parse_args(argv)

    try:
        if args.exec_bench:
            stdout, exit_code = _run_bench(args.operator)
            record = build_bench_execution_trace(
                args.operator, stdout, exit_code, args.note
            )
        elif args.receipt:
            record = build_receipt_trace(
                args.operator, args.receipt, args.verify_result, args.note
            )
        else:
            parser.error("需要 --receipt <路径> 或 --exec-bench 二者之一")
            return 2
        result = append_trace(args.operator, record, dry_run=args.dry_run)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"拒绝写入: {error}", file=sys.stderr)
        return 2

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
