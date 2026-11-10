#!/usr/bin/env python3
"""D0 基线窗口：一次运行，把硬件常数与搬运粒度标定下来。

对应 `EXPERIMENT_PLAN.md` 的 P0 / D0。一次运行做五件事，每件事落一个独立
JSON，最后汇总成 `window.json` 与人读的 `SUMMARY.md`：

| 阶段 | 测什么 | 产物 |
|---|---|---|
| A | 环境快照：设备、版本、显存、门控、宿主提交 | `00_env.json`、`00_npu-smi.txt` |
| B | 片外带宽：多尺寸连续拷贝，取稳定段 | `10_bandwidth.json` |
| C | Cube 峰值算力：固定 K/N 扫 M，取饱和段 | `11_cube_tflops.json` |
| D | HBM 可分配上限：逐步加块直到失败 | `12_hbm_ceiling.json` |
| E | 搬运粒度：行字节数从 2 到 128 扫一遍找台阶 | `13_transfer_granularity.json` |
| F | 汇总与未测项登记 | `window.json`、`SUMMARY.md` |

产物位置
--------
默认写到 `$SPLIT_BATCH_RESULTS_DIR/<stamp>/`，**必须在仓库外**：校验复用
`bench/evidence_paths.py`（它同时禁止相对路径和落在 `MOD` 仓库内的路径）。
显式给 `--out-dir` 时按同一规则再校验一次。本脚本自身不向仓库内写任何文件。

未测项（明确登记，不猜）
------------------------
- **片上 UB / L1 容量**：Python 侧无法测量，只能由 Ascend C kernel 申请到失败为止，
  结果里记 `measured: false` 并给出原因，不填估值。
- **跨核事件 / PDL 类机制**：属 D5，本窗口不做。

用法::

    export SPLIT_BATCH_RESULTS_DIR=<absolute dir outside the repo>
    python3 experiments/ascend910b-2card/run_baseline_window.py --stamp 1109-1400
    python3 experiments/ascend910b-2card/run_baseline_window.py --dry-run   # 只打印计划
"""

from __future__ import annotations

import argparse
import functools
import json
import operator
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALIGN_BYTES = 32
BANDWIDTH_SIZES_MIB = (16, 64, 128, 256, 512)
CUBE_SWEEP_M = (16, 64, 256, 1024, 2048, 4096)
CUBE_K = 4096
CUBE_N = 4096
GRANULARITY_ROW_ELEMS = (1, 2, 4, 8, 16, 32, 64)
GRANULARITY_ROWS = 1 << 20
GRANULARITY_REPS = 100
ARTIFACTS = (
    "00_env.json",
    "00_npu-smi.txt",
    "10_bandwidth.json",
    "11_cube_tflops.json",
    "12_hbm_ceiling.json",
    "13_transfer_granularity.json",
    "window.json",
    "SUMMARY.md",
)


class WindowError(RuntimeError):
    """窗口前置条件不满足时抛出（fail-closed，不做降级重试）。"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_out_dir(explicit: str | None) -> Path:
    """确定产物目录：必须在仓库外，且必须绝对路径。"""
    os.environ.setdefault("SPLIT_BATCH_RESULTS_DIR", "")
    from bench.evidence_paths import output_directory

    if explicit:
        os.environ["SPLIT_BATCH_RESULTS_DIR"] = explicit
    try:
        base = output_directory()
    except ValueError as exc:
        raise WindowError(f"产物目录不合规：{exc}") from exc
    return base


def plan_manifest(stamp: str, out_root: Path) -> dict:
    """列出这一次运行会做什么、产物落在哪。"""
    target = out_root / stamp
    return {
        "stamp": stamp,
        "output_dir": str(target),
        "artifacts": [str(target / name) for name in ARTIFACTS],
        "phases": [
            {
                "id": "A",
                "name": "environment_snapshot",
                "content": "设备数/名称/HBM、torch 与 CANN 版本、门控变量、宿主提交",
                "artifact": "00_env.json + 00_npu-smi.txt",
            },
            {
                "id": "B",
                "name": "hbm_bandwidth",
                "content": f"连续拷贝 {list(BANDWIDTH_SIZES_MIB)} MiB，取稳定段中位数",
                "artifact": "10_bandwidth.json",
            },
            {
                "id": "C",
                "name": "cube_peak_tflops",
                "content": f"K=N={CUBE_K}，M 扫 {list(CUBE_SWEEP_M)}，fp16 与 bf16",
                "artifact": "11_cube_tflops.json",
            },
            {
                "id": "D",
                "name": "hbm_alloc_ceiling",
                "content": "按 1 GiB 步进申请直到失败，记录可分配上限",
                "artifact": "12_hbm_ceiling.json",
            },
            {
                "id": "E",
                "name": "transfer_granularity",
                "content": (
                    f"行元素 {list(GRANULARITY_ROW_ELEMS)}（bf16 即 2~128 字节），"
                    f"{GRANULARITY_ROWS} 行 × {GRANULARITY_REPS} 次，找退化台阶"
                ),
                "artifact": "13_transfer_granularity.json",
            },
            {
                "id": "F",
                "name": "summary",
                "content": "汇总常数、登记未测项、打印产物清单",
                "artifact": "window.json + SUMMARY.md",
            },
        ],
        "not_measured": [
            {
                "item": "片上 UB / L1 容量",
                "reason": "Python 侧无接口，只能由 Ascend C kernel 申请到失败为止",
            },
            {"item": "跨核事件提前放行", "reason": "属计划 D5，本窗口不做"},
        ],
    }


def _timed(fn, reps: int, warmup: int, torch) -> list:
    """用 NPU Event 计时，返回每次的毫秒数。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def _copy_into(dst, src) -> None:
    """拷贝体：独立函数便于用 partial 绑定，避免闭包捕获循环变量。"""
    dst.copy_(src)


def _repeat_copy(dst, src, times: int = GRANULARITY_REPS) -> None:
    """重复拷贝，用来摊掉单次启动开销。"""
    for _ in range(times):
        dst.copy_(src)


def measure_bandwidth(torch, device: str, reps: int = 10) -> dict:
    """多尺寸连续拷贝，返回每个尺寸的 GB/s 与最稳的一段。"""
    rows = []
    for mib in BANDWIDTH_SIZES_MIB:
        elems = mib * 1024 * 1024 // 2  # bf16
        src = torch.randn(elems, dtype=torch.bfloat16, device=device)
        dst = torch.empty_like(src)
        op = functools.partial(_copy_into, dst, src)
        samples = _timed(op, reps=reps, warmup=3, torch=torch)
        ms = statistics.median(samples)
        moved = 2 * elems * 2 / 1024**3  # 读 + 写
        rows.append(
            {
                "size_mib": mib,
                "median_ms": round(ms, 4),
                "min_ms": round(min(samples), 4),
                "max_ms": round(max(samples), 4),
                "spread_pct": round(100 * (max(samples) - min(samples)) / ms, 2)
                if ms
                else None,
                "gb_per_s": round(moved / (ms / 1000), 1),
            }
        )
        del src, dst, op
    best = max(rows, key=lambda r: r["gb_per_s"])
    stable = [r for r in rows if r["size_mib"] >= 128] or rows
    return {
        "phase": "B",
        "measured_at": _now(),
        "device": device,
        "caliber": "连续拷贝，读写各计一次；单卡、单进程、eager",
        "cases": rows,
        "peak_case": best,
        "steady_gb_per_s": round(statistics.median(r["gb_per_s"] for r in stable), 1),
        "note": "steady 取不小于 128 MiB 的尺寸中位数；小尺寸受启动开销影响，不参与",
    }


def measure_cube(torch, device: str, reps: int = 10) -> dict:
    """固定 K/N 扫 M，估算 Cube 峰值算力。"""
    cases = []
    for dtype_name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        for m in CUBE_SWEEP_M:
            a = torch.randn(m, CUBE_K, dtype=dtype, device=device)
            b = torch.randn(CUBE_K, CUBE_N, dtype=dtype, device=device)
            samples = _timed(
                functools.partial(operator.matmul, a, b),
                reps=reps,
                warmup=5,
                torch=torch,
            )
            ms = statistics.median(samples)
            tflops = 2 * m * CUBE_K * CUBE_N / (ms / 1000) / 1e12
            cases.append(
                {
                    "dtype": dtype_name,
                    "m": m,
                    "k": CUBE_K,
                    "n": CUBE_N,
                    "median_ms": round(ms, 4),
                    "spread_pct": round(100 * (max(samples) - min(samples)) / ms, 2)
                    if ms
                    else None,
                    "tflops": round(tflops, 1),
                }
            )
            del a, b
    peak = max(cases, key=lambda c: c["tflops"])
    return {
        "phase": "C",
        "measured_at": _now(),
        "device": device,
        "caliber": f"eager 矩阵乘，K=N={CUBE_K}，取中位数；含分块与搬运",
        "cases": cases,
        "peak": peak,
        "note": "峰值取全部用例最大；小 M 受启动开销与 K/N 分块影响，只看趋势",
    }


def measure_hbm_ceiling(torch, device: str, max_gib: int = 40) -> dict:
    """按 1 GiB 步进申请，直到失败或触到上限。"""
    chunks = []
    step = 1024**3 // 2  # bf16 元素数，1 GiB
    total_gib = 0
    failed_at = None
    while total_gib < max_gib:
        try:
            chunks.append(torch.empty(step, dtype=torch.bfloat16, device=device))
            total_gib += 1
        except Exception as exc:  # noqa: BLE001 - 设备侧异常类型不稳定
            failed_at = f"{type(exc).__name__}: {exc}"
            break
    free, total = torch.npu.mem_get_info(device)
    result = {
        "phase": "D",
        "measured_at": _now(),
        "device": device,
        "caliber": "连续申请 1 GiB 块直到报错；记录申请成功量",
        "allocated_gib": total_gib,
        "cap_gib": max_gib,
        "hit_cap": failed_at is None and total_gib >= max_gib,
        "failure": failed_at,
        "mem_get_info": {
            "free_gib": round(free / 1024**3, 2),
            "total_gib": round(total / 1024**3, 2),
        },
        "note": "这是 HBM 可分配上限，不是片上 UB/L1 容量；片上容量需 Ascend C 探针",
    }
    del chunks
    return result


def detect_granularity_step(cases: list, factor: float = 10.0) -> dict:
    """在行字节数 vs 有效带宽的曲线上找第一个跨过 ``factor`` 倍的台阶。"""
    ordered = sorted(cases, key=lambda c: c["row_bytes"])
    if not ordered:
        raise WindowError("粒度用例为空")
    low = ordered[0]["gb_per_s"] or 0.0
    step = None
    for row in ordered:
        if low and row["gb_per_s"] >= low * factor:
            step = {
                "row_bytes": row["row_bytes"],
                "previous_gb_per_s": low,
                "gb_per_s": row["gb_per_s"],
                "ratio": round(row["gb_per_s"] / low, 1),
            }
            break
        low = max(low, row["gb_per_s"] or 0.0)
    return {
        "step": step,
        "min_bytes_observed": ordered[0]["row_bytes"],
        "max_bytes_observed": ordered[-1]["row_bytes"],
    }


def measure_granularity(torch, device: str) -> dict:
    """扫行字节数，找 MTE 退化台阶。"""
    cases = []
    for width in GRANULARITY_ROW_ELEMS:
        src = torch.randn(GRANULARITY_ROWS, GRANULARITY_ROW_ELEMS[-1])
        src = src.to(torch.bfloat16).to(device)
        dst = torch.empty_like(src)
        part_src = src[:, :width].contiguous()
        part_dst = dst[:, :width]
        samples = _timed(
            functools.partial(_repeat_copy, part_dst, part_src),
            reps=3,
            warmup=1,
            torch=torch,
        )
        ms = statistics.median(samples)
        moved = 2 * GRANULARITY_REPS * GRANULARITY_ROWS * width * 2
        cases.append(
            {
                "row_elems": width,
                "row_bytes": width * 2,
                "median_ms": round(ms, 3),
                "gb_per_s": round(moved / 1024**3 / (ms / 1000), 1),
            }
        )
        del src, dst
    verdict = detect_granularity_step(cases)
    return {
        "phase": "E",
        "measured_at": _now(),
        "device": device,
        "caliber": (
            f"每档 {GRANULARITY_ROWS} 行 × {GRANULARITY_REPS} 次列切片拷贝；"
            "切片来自连续大张量，行距固定 128 字节"
        ),
        "cases": cases,
        "verdict": verdict,
        "expected_boundary_bytes": ALIGN_BYTES,
        "agrees_with_expectation": (
            verdict["step"] is not None and verdict["step"]["row_bytes"] == ALIGN_BYTES
        ),
        "note": "有效带宽为读写合计；台阶位置对应最小搬运粒度",
    }


def snapshot_env(torch) -> dict:
    """设备与版本快照。"""
    import torch_npu  # noqa: F401 - 注册设备后端

    devices = []
    for i in range(torch.npu.device_count()):
        free, total = torch.npu.mem_get_info(i)
        devices.append(
            {
                "index": i,
                "name": torch.npu.get_device_name(i),
                "free_gib": round(free / 1024**3, 2),
                "total_gib": round(total / 1024**3, 2),
            }
        )
    return {
        "phase": "A",
        "measured_at": _now(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "ascend_home": os.environ.get("ASCEND_HOME_PATH", ""),
        "device_count": torch.npu.device_count(),
        "devices": devices,
        "gate_env": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("VLLM_HUST_")
        },
        "host_repos": _host_commits(),
    }


def _host_commits() -> dict:
    """Record the two host repos' HEADs; unset or missing repos record null.

    Repo locations come from the environment so this file carries no absolute
    path, and so the recorded fingerprint always points at the tree that was
    actually imported rather than an image-side copy that may differ.
    """
    result = {}
    for name, env_key in (
        ("vllm_ascend", "VLLM_ASCEND_REPO"),
        ("vllm_hust", "VLLM_HUST_REPO"),
    ):
        path = os.environ.get(env_key)
        if not path:
            result[name] = None
            continue
        try:
            out = subprocess.run(
                ["git", "-C", path, "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            result[name] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result[name] = None
    return result


def _npu_smi_text() -> str:
    try:
        out = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=30
        )
        return out.stdout or out.stderr
    except (OSError, subprocess.SubprocessError) as exc:
        return f"npu-smi 不可用: {type(exc).__name__}: {exc}"


def build_summary(window: dict) -> str:
    """人读摘要：只写口径、结论与未测项，不复述原始数字明细。"""
    env = window["env"]
    bw = window.get("bandwidth") or {}
    cube = window.get("cube") or {}
    ceiling = window.get("hbm_ceiling") or {}
    grain = window.get("granularity") or {}
    device_name = env["devices"][0]["name"] if env["devices"] else "unknown"
    peak_case = bw.get("peak_case") or {}
    cube_peak = cube.get("peak") or {}
    step = (grain.get("verdict") or {}).get("step")
    if step is None:
        step_text = "未发现台阶"
    else:
        step_text = (
            f"{step.get('row_bytes', '-')} 字节（相对前档 {step.get('ratio', '-')} 倍）"
        )
    if grain.get("agrees_with_expectation"):
        verdict_text = f"粒度台阶与计划的 {ALIGN_BYTES} 字节预期一致。"
    else:
        verdict_text = "粒度台阶与计划预期不一致，需按本次结果更新计划附录 A。"
    lines = [
        f"# 基线窗口 {window['stamp']}",
        "",
        f"- 生成时间（UTC）：{window['generated_at']}",
        f"- 产物目录：`{window['output_dir']}`",
        f"- 设备：{env['device_count']} × {device_name}",
        f"- python：{env['python']}",
        f"- torch：{env['torch']} / torch_npu：{env['torch_npu']}",
        f"- 宿主提交：{env['host_repos']}",
        "",
        "## 标定结果",
        "",
        f"- 片外带宽（稳定段）：{bw.get('steady_gb_per_s', '未测')} GB/s；"
        f"峰值用例 {peak_case.get('size_mib', '-')} MiB",
        f"- Cube 峰值算力：{cube_peak.get('tflops', '未测')} TFLOPS"
        f"（{cube_peak.get('dtype', '-')}, M={cube_peak.get('m', '-')}）",
        f"- HBM 可分配上限：{ceiling.get('allocated_gib', '未测')} GiB"
        f"（触顶={ceiling.get('hit_cap')}）",
        f"- 最小搬运粒度：{step_text}",
        "",
        "## 判读",
        "",
        f"- {verdict_text}",
        "- 带宽与算力两个常数可直接代入 "
        "`tools/analysis/ascend910b_static_audit.py` 的 "
        "`--peak-gb-s` / `--peak-tflops`。",
        "- 片上 UB/L1 容量本窗口未测：Python 无接口，需 Ascend C kernel 探针。",
        "",
        "## 未测项与原因",
        "",
    ]
    for item in window["not_measured"]:
        lines.append(f"- {item['item']}：{item['reason']}")
    lines += [
        "",
        "## 产物清单",
        "",
    ]
    for name in ARTIFACTS:
        lines.append(f"- `{window['output_dir']}/{name}`")
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_window(stamp: str, out_root: Path, skip_ceiling: bool, max_gib: int) -> dict:
    """执行一次完整窗口。"""
    from tools.op_agent.guard import check_npu_environment

    status = check_npu_environment(required_cards=2)
    if not status.has_npu or not status.is_tp2_ready:
        raise WindowError(f"双卡前置不满足：{status.message}")

    import torch
    import torch_npu  # noqa: F401 - 注册设备后端

    target = out_root / stamp
    target.mkdir(parents=True, exist_ok=True)
    device = "npu:0"
    started = time.time()

    env = snapshot_env(torch)
    _write_json(target / "00_env.json", env)
    (target / "00_npu-smi.txt").write_text(_npu_smi_text(), encoding="utf-8")

    bandwidth = measure_bandwidth(torch, device)
    _write_json(target / "10_bandwidth.json", bandwidth)

    cube = measure_cube(torch, device)
    _write_json(target / "11_cube_tflops.json", cube)

    if skip_ceiling:
        ceiling = {
            "phase": "D",
            "measured_at": _now(),
            "device": device,
            "skipped": True,
            "note": "按 --skip-hbm-ceiling 跳过",
        }
    else:
        ceiling = measure_hbm_ceiling(torch, device, max_gib=max_gib)
    _write_json(target / "12_hbm_ceiling.json", ceiling)

    granularity = measure_granularity(torch, device)
    _write_json(target / "13_transfer_granularity.json", granularity)

    window = {
        "schema": "ascend910b-baseline-window/v1",
        "stamp": stamp,
        "generated_at": _now(),
        "output_dir": str(target),
        "elapsed_s": round(time.time() - started, 1),
        "env": env,
        "bandwidth": bandwidth,
        "cube": cube,
        "hbm_ceiling": ceiling,
        "granularity": granularity,
        "not_measured": plan_manifest(stamp, out_root)["not_measured"],
        "caliber": "单卡、单进程、eager、无并发；一次运行一个窗口",
    }
    _write_json(target / "window.json", window)
    (target / "SUMMARY.md").write_text(build_summary(window), encoding="utf-8")
    return window


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="D0 基线窗口：标定带宽、Cube 峰值、HBM 上限与搬运粒度"
    )
    parser.add_argument(
        "--stamp",
        default=datetime.now().strftime("%m%d-%H%M"),
        help="窗口标记，默认按本地时间生成",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="产物根目录（必须在仓库外）；默认读 SPLIT_BATCH_RESULTS_DIR",
    )
    parser.add_argument("--max-alloc-gib", type=int, default=40)
    parser.add_argument("--skip-hbm-ceiling", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划与产物路径，不碰设备（CPU 环境可用）",
    )
    args = parser.parse_args(argv)

    try:
        out_root = resolve_out_dir(args.out_dir)
    except WindowError as exc:
        print(f"[fail-closed] {exc}", file=sys.stderr)
        return 2
    manifest = plan_manifest(args.stamp, out_root)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    try:
        window = run_window(
            stamp=args.stamp,
            out_root=out_root,
            skip_ceiling=args.skip_hbm_ceiling,
            max_gib=args.max_alloc_gib,
        )
    except WindowError as exc:
        print(f"[fail-closed] {exc}", file=sys.stderr)
        return 3

    print(f"窗口 {window['stamp']} 完成，用时 {window['elapsed_s']}s")
    for name in ARTIFACTS:
        print(f"  {window['output_dir']}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
