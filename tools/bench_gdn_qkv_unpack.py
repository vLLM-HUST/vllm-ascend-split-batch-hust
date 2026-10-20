#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GDN 混合 QKV unpack（``rearrange_mixed_qkv``）L0 配对 A/B 微基准。

单变量：基线 = split + 3×reshape(-1) + cat（4 launch，逐字照抄
``vllm-hust/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:890-924``）；
候选 = 同契约单次 copy。

契约（不满足即无数字）：bitwise（int 视图，含 ±0.0/NaN 位图）｜shape+stride 同（并与
直接切片真值比）｜输入无修改｜输出同一 backing 且三段不重叠｜bf16 主性能，fp16/fp32 只作
正确性。

计时：NPU 用 ``torch.npu.Event`` 包夹 iters 次 eager 调用 → /iters，scope =
``npu_event_elapsed_eager_chain_including_host_submission_gaps``：链内 host 供给空洞
计入，只作**同 shape 独立 eager 链的流程局部延迟**，不是 FULL/图模式时长，也不是
L4/all-kernel 占比。

预算：内部只在 cell 之间软停（不打断编译/校验/测量）；硬上限交给外层 ``timeout``，
单进程、不重试。

用法::

    timeout -k 30 1500 opt bench l0 GdnQkvUnpack \\
        --candidate vllm_ascend_split_batch.gdn_qkv_unpack:rearrange_mixed_qkv

退出码：0 成功（含 --baseline-only，产物标记无 A/B）｜1 失败（不换配置重试）｜
2 用法错｜5 软预算停（只留完整 cell）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import torch

SCHEMA = "opt-ops-l0-gdn-qkv-unpack.v1"
DEFAULT_CANDIDATE = "vllm_ascend_split_batch.gdn_qkv_unpack:rearrange_mixed_qkv"
DEFAULT_CANDIDATE_ARG_STYLE = "dims"
MODEL = "Qwen3.5-35B-A3B"
CANONICAL = "vllm-hust/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
CANONICAL_LINES = "890-924"
# 行号读自运行时安装树（真实被 serve 的那份），不是仓内 clone
CALLSITE_TREE = "/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py"
CALLSITES = (
    f"{CALLSITE_TREE}:479 rearrange_mixed_qkv(mixed_qkv_spec)",
    f"{CALLSITE_TREE}:480 rearrange_mixed_qkv(mixed_qkv_non_spec)",
    f"{CALLSITE_TREE}:535 rearrange_mixed_qkv(mixed_qkv_non_spec[:num_decode_tokens])",
)
# 仓内 clone 的同名文件里 split 两条在 532/533，且无 decode 那条：两树行号不可互抄
CALLSITE_NOTE = (
    "仓内 clone vllm-ascend-hust 的同文件 split 两条为 532/533、"
    "无 decode 第 3 条；行号只对安装树成立"
)
# 相位只对已声明点给值；1/2/17 是正确性点，源未声明相位
DECLARED_PHASES = {
    16: "decode",
    825: "prefill",
    1489: "prefill",
    1587: "prefill",
    2048: "prefill",
}
DEFAULT_TOKENS = "1,2,16,17,825,1489,1587,2048"
DEFAULT_ROW_STRIDES = "4096,6176"
MEASURED_ROW_STRIDE = 4096
SHAPE_MEASURED = "measured-shape"
SHAPE_SYNTHETIC = "synthetic-coverage"
TIMER_SCOPE_NPU = "npu_event_elapsed_eager_chain_including_host_submission_gaps"
TIMER_SCOPE_CPU = "host_wall_perf_counter_elapsed_including_host_submission"
METRIC_NPU = "us_per_iter_event_elapsed"
METRIC_CPU = "us_host_total"
SCOPE_MEASURED = (
    "same-shape independent eager-chain local latency per call "
    "(baseline 4 launches vs candidate 1)"
)
SCOPE_SYNTHETIC = "synthetic coverage only; not a production-shape gain"
PERF_DTYPE = "bfloat16"
SUPPLEMENT_DTYPES = ("float16", "float32")
INT_VIEW = {"bfloat16": torch.int16, "float16": torch.int16, "float32": torch.int32}
NAN_MASK = {
    "bfloat16": (0x7F80, 0x007F),
    "float16": (0x7C00, 0x03FF),
    "float32": (0x7F800000, 0x007FFFFF),
}
SPECIAL_BITS = {
    "bfloat16": {
        "pos_zero": 0x0000,
        "neg_zero": 0x8000,
        "qnan": 0x7FC0,
        "neg_qnan": 0xFFC0,
        "pos_inf": 0x7F80,
        "neg_inf": 0xFF80,
    },
    "float16": {
        "pos_zero": 0x0000,
        "neg_zero": 0x8000,
        "qnan": 0x7E00,
        "neg_qnan": 0xFE00,
        "pos_inf": 0x7C00,
        "neg_inf": 0xFC00,
    },
    "float32": {
        "pos_zero": 0x00000000,
        "neg_zero": 0x80000000,
        "qnan": 0x7FC00000,
        "neg_qnan": 0xFFC00000,
        "pos_inf": 0x7F800000,
        "neg_inf": 0xFF800000,
    },
}
RC_OK, RC_FAIL, RC_USAGE, RC_BUDGET = 0, 1, 2, 5


def die_usage(msg: str) -> int:
    print(msg, file=sys.stderr)
    return RC_USAGE


@dataclass(frozen=True)
class Dims:
    q_dim: int = 1024
    k_dim: int = 1024
    v_dim: int = 2048
    head_k_dim: int = 128
    head_v_dim: int = 128

    @property
    def width(self) -> int:
        return self.q_dim + self.k_dim + self.v_dim

    @property
    def out_shapes(self) -> tuple[tuple[int, ...], ...]:
        return (
            (1, 0, self.q_dim // self.head_k_dim, self.head_k_dim),
            (1, 0, self.k_dim // self.head_k_dim, self.head_k_dim),
            (1, 0, self.v_dim // self.head_v_dim, self.head_v_dim),
        )

    def shapes_for(self, tokens: int) -> tuple[tuple[int, ...], ...]:
        return tuple((1, tokens, s[2], s[3]) for s in self.out_shapes)

    def as_dict(self) -> dict:
        return {
            "q_dim": self.q_dim,
            "k_dim": self.k_dim,
            "v_dim": self.v_dim,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "packed_width": self.width,
        }


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, minutes: float) -> None:
        self.minutes = minutes
        self.deadline = time.monotonic() + minutes * 60.0
        self.exhausted = False
        self.stop_reason = ""

    def check(self, where: str) -> None:
        if time.monotonic() >= self.deadline:
            self.exhausted = True
            self.stop_reason = where
            raise BudgetExhausted(where)

    def as_dict(self) -> dict:
        return {
            "wall_budget_min": self.minutes,
            "exhausted": self.exhausted,
            "stop_reason": self.stop_reason,
        }


# ---------------------------------------------------------------- 基线（逐字）
def baseline_unpack(mixed_qkv: torch.Tensor, d: Dims) -> tuple[torch.Tensor, ...]:
    seq_len = mixed_qkv.shape[0]
    q_dim, k_dim, v_dim = d.q_dim, d.k_dim, d.v_dim
    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)
    q_size = seq_len * q_dim
    k_size = seq_len * k_dim
    q_contig = fused[0:q_size]
    k_contig = fused[q_size : q_size + k_size]
    v_contig = fused[q_size + k_size :]
    query = q_contig.view(1, seq_len, -1, d.head_k_dim)
    key = k_contig.view(1, seq_len, -1, d.head_k_dim)
    value = v_contig.view(1, seq_len, -1, d.head_v_dim)
    return query, key, value


def truth_unpack(mixed_qkv: torch.Tensor, d: Dims) -> tuple[torch.Tensor, ...]:
    """直接切片真值（与基线实现无关），用于发现基线自身漂移。"""
    seq_len = mixed_qkv.shape[0]
    q = mixed_qkv[:, 0 : d.q_dim].reshape(1, seq_len, -1, d.head_k_dim)
    k = mixed_qkv[:, d.q_dim : d.q_dim + d.k_dim].reshape(1, seq_len, -1, d.head_k_dim)
    v = mixed_qkv[:, d.q_dim + d.k_dim :].reshape(1, seq_len, -1, d.head_v_dim)
    return q, k, v


# ---------------------------------------------------------------- 输入构造
def _signed(bits: int, int_dtype: torch.dtype) -> int:
    width = 16 if int_dtype is torch.int16 else 32
    return bits if bits < (1 << (width - 1)) else bits - (1 << width)


def build_input(
    tokens: int, row_stride: int, dtype: str, seed: int, specials: bool, device: str
) -> torch.Tensor:
    """构造 packed 输入视图：[tokens, width] 行跨距 = row_stride 的 strided view。"""
    d = _DIMS
    gen = torch.Generator(device="cpu").manual_seed(seed)
    inner = torch.randn(tokens, d.width, generator=gen, dtype=torch.float32)
    inner = inner.to(getattr(torch, dtype))
    if specials:
        view = inner.view(INT_VIEW[dtype]).reshape(-1)
        spots = [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (5, 6)]
        for (row, col), name in zip(spots, SPECIAL_BITS[dtype], strict=True):
            if row < tokens and col < d.width:
                view[row * d.width + col] = _signed(
                    SPECIAL_BITS[dtype][name], INT_VIEW[dtype]
                )
    buf = torch.zeros(tokens, row_stride, dtype=getattr(torch, dtype), device=device)
    buf[:, : d.width] = inner.to(device)
    return buf[:, : d.width]


_DIMS = Dims()


# ---------------------------------------------------------------- 数值/契约
def int_view_cpu(t: torch.Tensor, dtype: str) -> torch.Tensor:
    return t.detach().to("cpu").contiguous().view(INT_VIEW[dtype]).reshape(-1)


def bit_stats(t: torch.Tensor, dtype: str) -> dict:
    if t.dtype != getattr(torch, dtype):
        return {"dtype": str(t.dtype), "skipped": "dtype mismatch"}
    iv = int_view_cpu(t, dtype)
    exp_mask, mant_mask = NAN_MASK[dtype]
    is_nan = ((iv & exp_mask) == exp_mask) & ((iv & mant_mask) != 0)
    return {
        "dtype": dtype,
        "numel": int(iv.numel()),
        "nan_count": int(is_nan.sum()),
        "neg_zero_count": int(
            (iv == _signed(SPECIAL_BITS[dtype]["neg_zero"], INT_VIEW[dtype])).sum()
        ),
    }


def compare_tensors(
    ref: torch.Tensor, cand: torch.Tensor, dtype: str, check_stride: bool = True
) -> dict:
    shape_ok = tuple(ref.shape) == tuple(cand.shape)
    stride_ok = tuple(ref.stride()) == tuple(cand.stride())
    dtype_ok = ref.dtype == cand.dtype
    out = {
        "dtype_ok": dtype_ok,
        "shape_ok": shape_ok,
        "stride_ok": stride_ok if check_stride else None,
        "stride_checked": check_stride,
        "ref_shape": list(ref.shape),
        "cand_shape": list(cand.shape),
        "ref_stride": list(ref.stride()),
        "cand_stride": list(cand.stride()),
    }
    if not (dtype_ok and shape_ok):
        out["bitwise_ok"] = False
        out["ref_bits"] = bit_stats(ref, dtype)
        out["cand_bits"] = bit_stats(cand, dtype)
        return out
    a = int_view_cpu(ref, dtype)
    b = int_view_cpu(cand, dtype)
    bitwise = bool(torch.equal(a, b))
    out["bitwise_ok"] = bitwise
    out["ref_bits"] = bit_stats(ref, dtype)
    out["cand_bits"] = bit_stats(cand, dtype)
    if not bitwise:
        diff = a != b
        out["mismatch_count"] = int(diff.sum())
        out["first_mismatch_index"] = int(diff.nonzero()[0].item())
    return out


def compare_arm(
    ref: tuple[torch.Tensor, ...],
    cand: tuple[torch.Tensor, ...],
    dtype: str,
    check_stride: bool = True,
) -> dict:
    parts = [
        compare_tensors(r, c, dtype, check_stride)
        for r, c in zip(ref, cand, strict=True)
    ]
    return {
        "parts": parts,
        "ok": all(
            p["bitwise_ok"]
            and p["shape_ok"]
            and p["stride_ok"] in (True, None)
            and p["dtype_ok"]
            for p in parts
        ),
    }


def output_contract(outs: tuple[torch.Tensor, ...], tokens: int, d: Dims) -> dict:
    exp = d.shapes_for(tokens)
    shapes_ok = len(outs) == 3 and all(
        tuple(t.shape) == e for t, e in zip(outs, exp, strict=True)
    )
    segs = [(t.data_ptr(), t.numel() * t.element_size()) for t in outs]
    distinct = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (s0, n0), (s1, n1) = segs[i], segs[j]
            distinct.append(s0 + n0 <= s1 or s1 + n1 <= s0)
    same_backing = len({t.untyped_storage().data_ptr() for t in outs}) == 1
    doc = {
        "n_tensors": len(outs),
        "shape_ok": bool(shapes_ok),
        "expected_shapes": [list(e) for e in exp],
        "actual_shapes": [list(t.shape) for t in outs],
        "same_backing": bool(same_backing),
        "segments_disjoint": bool(all(distinct)),
    }
    doc["ok"] = (
        doc["n_tensors"] == 3
        and doc["shape_ok"]
        and doc["same_backing"]
        and doc["segments_disjoint"]
    )
    return doc


# ---------------------------------------------------------------- 臂与调度
def ab_schedule(arms: tuple[str, ...], repeats: int) -> list[list[str]]:
    return [list(arms) if i % 2 == 0 else list(reversed(arms)) for i in range(repeats)]


def resolve_candidate(spec: str, style: str, d: Dims):
    """解析候选 (module:attr 或 path/to/file.py:attr)；失败抛异常，不回落基线。"""
    mod_part, _, attr = spec.rpartition(":")
    if not mod_part or not attr:
        raise ValueError(f"--candidate 需 <module|file.py>:<attr>（收到 {spec!r}）")
    if mod_part.endswith(".py"):
        if not os.path.isfile(mod_part):
            raise FileNotFoundError(f"候选实现不存在: {mod_part}")
        name = "gdn_qkv_unpack_candidate"
        mspec = importlib.util.spec_from_file_location(name, mod_part)
        if mspec is None or mspec.loader is None:
            raise ImportError(f"无法从 {mod_part} 构造模块 spec")
        mod = importlib.util.module_from_spec(mspec)
        sys.modules[name] = mod
        mspec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_part)
    fn = getattr(mod, attr, None)
    if not callable(fn):
        raise AttributeError(f"{mod_part} 没有可调用的 {attr}")
    info = {
        "spec": spec,
        "module": mod_part,
        "attr": attr,
        "module_file": os.path.abspath(getattr(mod, "__file__", "") or ""),
        "callable_module_file": _callable_file(fn),
    }
    if style == "packed":
        return (lambda x: tuple(fn(x))), info
    if style == "dims":
        kwargs = {
            "q_dim": d.q_dim,
            "k_dim": d.k_dim,
            "v_dim": d.v_dim,
            "head_k_dim": d.head_k_dim,
            "head_v_dim": d.head_v_dim,
        }
        return (lambda x: tuple(fn(x, **kwargs))), info
    raise ValueError(f"--candidate-arg-style 只能是 packed|dims（收到 {style!r}）")


def _callable_file(fn) -> str:
    """候选可调用对象**定义处**的文件（惰性包属性下与 spec 模块不是同一份）。"""
    name = getattr(fn, "__module__", None)
    if not name:
        return ""
    mod = sys.modules.get(name)
    if mod is None:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - 拿不到来源就如实留空
            return ""
    return os.path.abspath(getattr(mod, "__file__", "") or "")


def shape_role(tokens: int, row_stride: int) -> str:
    """两条轴都取实测事实才算 measured；否则只作合成覆盖。"""
    if tokens in DECLARED_PHASES and row_stride == MEASURED_ROW_STRIDE:
        return SHAPE_MEASURED
    return SHAPE_SYNTHETIC


def timed(fn, iters: int, device: str) -> dict:
    if device == "npu":
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        t_enq = time.perf_counter()
        torch.npu.synchronize()
        t_tot = time.perf_counter()
        total_us = start.elapsed_time(end) * 1000.0
        return {
            "timer_scope": TIMER_SCOPE_NPU,
            "iters": iters,
            "event_elapsed_us_total": round(total_us, 3),
            "us_per_iter_event_elapsed": round(total_us / iters, 3),
            "us_host_enqueue": round((t_enq - t0) * 1e6 / iters, 3),
            "us_host_total": round((t_tot - t0) * 1e6 / iters, 3),
        }
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    wall = (time.perf_counter() - t0) * 1e6 / iters
    return {
        "timer_scope": TIMER_SCOPE_CPU,
        "iters": iters,
        "event_elapsed_us_total": None,
        "us_per_iter_event_elapsed": None,
        "us_host_enqueue": round(wall, 3),
        "us_host_total": round(wall, 3),
    }


def median_metric(rows: list[dict], device: str) -> tuple[float, str]:
    key = METRIC_NPU if device == "npu" else METRIC_CPU
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.median(vals), 3), key


# ---------------------------------------------------------------- 代码/硬件
def sha256_file(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _inside_repo(path: str, repo: str) -> bool:
    if not path:
        return False
    real_path, real_repo = os.path.realpath(path), os.path.realpath(repo)
    try:
        return os.path.commonpath([real_repo, real_path]) == real_repo
    except ValueError:
        return False


def package_py_files(module_file: str, repo: str) -> dict:
    """候选包内**全部** .py（含 kernel）→ 仓内相对路径 + sha256。

    仓外路径只标记 ``inside_repo=False``，不 hash（避免把别处代码当本仓候选）。
    """
    doc: dict = {"dir": None, "inside_repo": False, "files": {}}
    if not module_file or not os.path.isfile(module_file):
        return doc
    pkg_dir = os.path.dirname(os.path.abspath(module_file))
    doc["dir"] = pkg_dir
    doc["inside_repo"] = _inside_repo(pkg_dir, repo)
    if not doc["inside_repo"]:
        return doc
    real_repo, real_dir = os.path.realpath(repo), os.path.realpath(pkg_dir)
    for root, dirs, names in os.walk(real_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for name in sorted(names):
            if name.endswith(".py"):
                full = os.path.join(root, name)
                doc["files"][os.path.relpath(full, real_repo)] = sha256_file(full)
    return doc


def code_provenance(cand: dict) -> dict:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    head = dirty = None
    try:
        head = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        dirty = len([ln for ln in porcelain.splitlines() if ln.strip()])
    except (OSError, subprocess.SubprocessError):
        pass
    doc = {
        "repo": repo,
        "repo_head": head or None,
        "repo_dirty_paths": dirty,
        "tool_sha256": sha256_file(os.path.abspath(__file__)),
    }
    if cand:
        module_file = cand.get("module_file") or ""
        callable_file = cand.get("callable_module_file") or ""
        doc["candidate"] = {
            "spec": cand.get("spec"),
            "module": cand.get("module"),
            "attr": cand.get("attr"),
            "module_file": module_file or None,
            "module_sha256": sha256_file(module_file),
            "callable_module_file": callable_file or None,
            "callable_module_sha256": sha256_file(callable_file),
            "inside_repo": _inside_repo(module_file, repo),
            "callable_inside_repo": _inside_repo(callable_file, repo),
            "package": package_py_files(module_file, repo),
        }
    return doc


def resolve_device(spec: str) -> tuple[str, dict]:
    if spec == "cpu":
        return "cpu", {
            "device": "cpu",
            "device_name": platform.processor() or "cpu",
            "device_count": 0,
            "host": platform.node(),
            "machine": platform.machine(),
            "torch": torch.__version__,
        }
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - 缺 torch_npu 时如实回落
        if spec == "npu":
            raise RuntimeError(
                f"要求 --device npu 但无法 import torch_npu: {exc!r}"
            ) from exc
        return "cpu", {
            "device": "cpu",
            "device_name": platform.processor() or "cpu",
            "device_count": 0,
            "host": platform.node(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "npu_unavailable": repr(exc),
        }
    count = torch.npu.device_count()
    if count == 0:
        if spec == "npu":
            raise RuntimeError("要求 --device npu 但 torch.npu.device_count()==0")
        return "cpu", {
            "device": "cpu",
            "device_name": platform.processor() or "cpu",
            "device_count": 0,
            "host": platform.node(),
            "machine": platform.machine(),
            "torch": torch.__version__,
        }
    return "npu", {
        "device": "npu",
        "device_name": torch.npu.get_device_name(0),
        "device_count": count,
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", None),
    }


# ---------------------------------------------------------------- 单元
def run_cell(cell: dict, ctx: dict) -> dict:
    d, device = ctx["dims"], ctx["device"]
    tokens, row_stride = cell["T"], cell["row_stride"]
    role = shape_role(tokens, row_stride)
    seed_base = ctx["seed"] + tokens * 1000 + row_stride
    doc = {
        "id": cell["id"],
        "T": tokens,
        "row_stride": row_stride,
        "input_layout": "contiguous" if row_stride == d.width else "strided-view",
        "phase": cell["phase"],
        "phase_source": cell["phase_source"],
        "shape_role": role,
        "claim_scope": SCOPE_MEASURED if role == SHAPE_MEASURED else SCOPE_SYNTHETIC,
        "in_shape": [tokens, d.width],
        "in_stride": [row_stride, 1],
        "out_shapes": [list(s) for s in d.shapes_for(tokens)],
        "numeric": {},
        "contract": {},
    }

    for dtype in (PERF_DTYPE, *ctx["supplement"]):
        x = build_input(tokens, row_stride, dtype, seed_base, True, device)
        snap = int_view_cpu(x, dtype).clone()
        base = tuple(baseline_unpack(x, d))
        truth = truth_unpack(x, d)
        entry: dict = {
            "baseline_vs_truth": compare_arm(truth, base, dtype, False),
            "contract": {"baseline": output_contract(base, tokens, d)},
        }
        checks = [entry["baseline_vs_truth"]["ok"], entry["contract"]["baseline"]["ok"]]
        if ctx["candidate"] is not None:
            cand = tuple(ctx["candidate"](x))
            entry["candidate_vs_baseline"] = compare_arm(base, cand, dtype)
            entry["candidate_vs_truth"] = compare_arm(truth, cand, dtype, False)
            entry["contract"]["candidate"] = output_contract(cand, tokens, d)
            checks += [
                entry["candidate_vs_baseline"]["ok"],
                entry["candidate_vs_truth"]["ok"],
                entry["contract"]["candidate"]["ok"],
            ]
        entry["input_unmodified"] = bool(torch.equal(snap, int_view_cpu(x, dtype)))
        checks.append(entry["input_unmodified"])
        entry["ok"] = bool(all(checks))
        doc["numeric"][dtype] = entry
        if not entry["ok"]:
            doc["failed_dtype"] = dtype
            return doc
        del base, truth, x, snap
        if ctx["candidate"] is not None:
            del cand

    x = build_input(tokens, row_stride, PERF_DTYPE, seed_base + 1, False, device)
    snap = int_view_cpu(x, PERF_DTYPE).clone()
    arms = ctx["arms"]
    fns = {"baseline": lambda: baseline_unpack(x, d)}
    if ctx["candidate"] is not None:
        fns["candidate"] = lambda: ctx["candidate"](x)
    for f in fns.values():
        for _ in range(ctx["warmup"]):
            f()
    if device == "npu":
        torch.npu.synchronize()
    rows: dict[str, list[dict]] = {a: [] for a in arms}
    for order in ctx["schedule"]:
        for arm in order:
            rows[arm].append(timed(fns[arm], ctx["iters"], device))
    post_intact = bool(torch.equal(snap, int_view_cpu(x, PERF_DTYPE)))
    perf: dict[str, dict] = {}
    for arm in arms:
        med, key = median_metric(rows[arm], device)
        perf[arm] = {"us_median": med, "metric": key, "runs": rows[arm]}
    doc["perf"] = {
        "dtype": PERF_DTYPE,
        "iters": ctx["iters"],
        "repeats": ctx["repeats"],
        "warmup": ctx["warmup"],
        "schedule": [list(o) for o in ctx["schedule"]],
        "timer_scope": (TIMER_SCOPE_NPU if device == "npu" else TIMER_SCOPE_CPU),
        "shape_role": role,
        "claim_scope": doc["claim_scope"],
        "arms": perf,
        "post_measure_input_unmodified": post_intact,
    }
    if len(arms) == 2:
        base_key = perf["baseline"]["metric"]
        base_runs = [r[base_key] for r in rows["baseline"]]
        cand_runs = [r[base_key] for r in rows["candidate"]]
        paired = [
            round((c / b - 1) * 100, 3) if b else None
            for b, c in zip(cand_runs, base_runs, strict=True)
        ]
        doc["perf"]["paired_delta_pct"] = paired
        doc["perf"]["paired_delta_pct_median"] = (
            round(statistics.median([p for p in paired if p is not None]), 3)
            if any(p is not None for p in paired)
            else None
        )
        base_us = perf["baseline"]["us_median"]
        doc["perf"]["delta_pct"] = round(
            (perf["candidate"]["us_median"] / base_us - 1) * 100, 3
        )
    doc["ok"] = post_intact
    return doc


# ---------------------------------------------------------------- 主流程
def parse_list(spec: str, conv):
    return [conv(x) for x in str(spec).split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--candidate",
        default=DEFAULT_CANDIDATE,
        help="候选实现 <module|file.py>:<attr>（默认 %(default)s）",
    )
    ap.add_argument(
        "--candidate-arg-style",
        choices=["packed", "dims"],
        default=DEFAULT_CANDIDATE_ARG_STYLE,
        help="packed=fn(mixed_qkv)；dims=额外给 q/k/v/head 维关键字"
        "（默认 %(default)s）",
    )
    ap.add_argument(
        "--baseline-only",
        action="store_true",
        help="只测基线（产物显式标记无 A/B，不得当配对结论）",
    )
    ap.add_argument("--device", choices=["auto", "npu", "cpu"], default="auto")
    ap.add_argument("--tokens", default=DEFAULT_TOKENS, help="token 档位（逗号分隔）")
    ap.add_argument(
        "--row-strides",
        default=DEFAULT_ROW_STRIDES,
        help="输入行跨距档位（4096=连续实测布局；6176=合成补充视图）",
    )
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--q-dim", type=int, default=1024)
    ap.add_argument("--k-dim", type=int, default=1024)
    ap.add_argument("--v-dim", type=int, default=2048)
    ap.add_argument("--head-k-dim", type=int, default=128)
    ap.add_argument("--head-v-dim", type=int, default=128)
    ap.add_argument(
        "--supplement-dtypes",
        default="float16,float32",
        help="补充正确性精度（不测时间）",
    )
    ap.add_argument(
        "--wall-budget-min",
        type=float,
        default=20.0,
        help="cell 之间软停的墙钟上限（不打断编译/校验/测量；硬上限用外层"
        " timeout；不重试）",
    )
    ap.add_argument("--out", default="", help="结果 json 输出路径")
    args = ap.parse_args(argv)

    global _DIMS
    _DIMS = Dims(args.q_dim, args.k_dim, args.v_dim, args.head_k_dim, args.head_v_dim)
    d = _DIMS
    if args.iters <= 0 or args.repeats <= 0 or args.warmup < 0:
        return die_usage("--iters/--repeats 必须 >0，--warmup 必须 >=0")
    tokens = parse_list(args.tokens, int)
    strides = parse_list(args.row_strides, int)
    if not tokens or not strides:
        return die_usage("--tokens/--row-strides 不能为空")
    if any(t <= 0 for t in tokens) or any(s < d.width for s in strides):
        return die_usage(f"token 必须 >0；行跨距必须 >= packed 宽度 {d.width}")
    supplement = parse_list(args.supplement_dtypes, str)
    unknown = [x for x in supplement if x not in SUPPLEMENT_DTYPES]
    if unknown or PERF_DTYPE in supplement:
        return die_usage(
            f"补充精度只能是 {list(SUPPLEMENT_DTYPES)}（收到 {supplement}）"
        )

    budget = Budget(args.wall_budget_min)
    t_start = time.monotonic()
    product: dict = {
        "schema": SCHEMA,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "ok",
        "command": {
            "argv": list(sys.argv if argv is None else argv),
            "cwd": os.getcwd(),
            "python": sys.executable,
        },
        "goal": {
            "op": "rearrange_mixed_qkv (GDN mixed QKV unpack)",
            "baseline": "split + reshape x3 + cat = 4 launch（逐字上游实现）",
            "candidate": "单次 copy（同契约）",
            "callsite": {
                "canonical": f"{CANONICAL}:{CANONICAL_LINES}",
                "callers": list(CALLSITES),
                "source": "安装树直读（非窗口产物）",
                "tree": CALLSITE_TREE,
                "note": CALLSITE_NOTE,
            },
        },
        "fidelity": {
            "model": MODEL,
            "precision": {"perf": PERF_DTYPE, "supplement_correctness": supplement},
            "weight_format": {
                "declared": "unspecified",
                "note": "本算子只搬数据不读权重；权重格式不适用",
                "do_not_infer": "NZ/ND",
            },
        },
        "layer_dims": d.as_dict(),
        "cells": [],
        "budget": budget.as_dict(),
        "not_claimed": [
            "不宣称任何 e2e/吞吐/图指标收益；L0 数字不换算成端到端",
            "计时是 eager 链流程局部延迟（含链内 host 供给空洞），"
            "不是 FULL/图模式时长，也不是 L4/all-kernel 占比",
            "T=1/2/17 与行跨距 6176 为合成覆盖，不作生产范围、不称实测 shape 收益",
        ],
    }

    try:
        device, hw = resolve_device(args.device)
        product["hardware"] = hw
        product["caliber"] = {
            "timing": {
                "per_iter": "us",
                "iters_per_round": args.iters,
                "repeats": args.repeats,
                "warmup": args.warmup,
                "timer_scope": {"npu": TIMER_SCOPE_NPU, "cpu": TIMER_SCOPE_CPU},
                "npu": "torch.npu.Event 包夹 iters 次 eager 调用 → /iters/"
                "（event_elapsed_us_total、us_per_iter_event_elapsed）",
                "host_enqueue": "perf_counter 同步前包夹 → /iters",
                "host_total": "perf_counter 含末次 synchronize → /iters",
                "measures": "同一 shape 下独立 eager 调用链的流程局部延迟"
                "（per call）；Event 只包住 iters 次调用，"
                "链内的 host 供给空洞计入",
                "does_not_measure": [
                    "FULL/图模式时长",
                    "L4 all-kernel 占比",
                    "e2e/吞吐",
                ],
                "pairing": "偶数轮正序、奇数轮反序；两臂同轮相邻测量",
                "device_available": device == "npu",
            },
            "format": {
                "declared": "ND",
                "source": "窗口 1017-093650 kernel_details.csv "
                "Input/Output Formats（rearrange 的 4 个 "
                "launch 均为 ND）",
                "note": "L0 自造输入同为 ND（连续或 strided view）",
            },
            "input_layouts": [
                {
                    "row_stride": MEASURED_ROW_STRIDE,
                    "label": "measured-layout",
                    "source": "窗口 1017-093650: aclnnInplaceCopy_Slice in='T,4096' → "
                    "out='T,1024'（q/k）与 out='T,2048'（v）",
                },
                {
                    "row_stride": 6176,
                    "label": "synthetic-supplementary",
                    "source": "窗口同族里存在 6176 宽缓冲视图"
                    "（in='T,6176' → 'T,6144'），"
                    "本档只复用行跨距不等于宽度的布局不变量，不是对生产的复现",
                },
            ],
        }

        arms: tuple[str, ...] = (
            ("baseline",) if args.baseline_only else ("baseline", "candidate")
        )
        cand_info: dict = {}
        candidate = None
        if not args.baseline_only:
            budget.check("resolve-candidate")
            try:
                candidate, cand_info = resolve_candidate(
                    args.candidate, args.candidate_arg_style, d
                )
            except Exception as exc:  # noqa: BLE001 - 候选缺失必须如实失败
                product["status"] = "candidate-missing"
                product["candidate"] = {"spec": args.candidate, "error": repr(exc)}
                print(f"候选不可用（不回落基线、不产出 A/B）: {exc!r}", file=sys.stderr)
                _finish(product, budget, t_start, args.out)
                return RC_FAIL
        product["arms"] = {
            "requested": list(arms),
            "candidate_spec": (None if args.baseline_only else args.candidate),
            "candidate_arg_style": (
                None if args.baseline_only else args.candidate_arg_style
            ),
        }
        product["code"] = code_provenance(cand_info)

        cells = []
        for tokens_i in tokens:
            for stride in strides:
                cells.append(
                    {
                        "id": f"T{tokens_i}-rs{stride}",
                        "T": tokens_i,
                        "row_stride": stride,
                        "phase": DECLARED_PHASES.get(tokens_i),
                        "phase_source": (
                            "declared"
                            if tokens_i in DECLARED_PHASES
                            else "undeclared (correctness-only)"
                        ),
                        "shape_role": shape_role(tokens_i, stride),
                    }
                )
        product["shape_scope"] = {
            "rule": "只有『已声明相位 且 行跨距=4096』算实测形状；其余只作合成覆盖",
            "measured": [c["id"] for c in cells if c["shape_role"] == SHAPE_MEASURED],
            "synthetic": [c["id"] for c in cells if c["shape_role"] == SHAPE_SYNTHETIC],
            "claim": SCOPE_MEASURED,
            "synthetic_claim": SCOPE_SYNTHETIC,
        }
        ctx = {
            "dims": d,
            "device": device,
            "seed": args.seed,
            "iters": args.iters,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "supplement": tuple(supplement),
            "arms": arms,
            "schedule": ab_schedule(arms, args.repeats),
            "candidate": candidate,
        }
        for cell in cells:
            budget.check(f"before cell {cell['id']}")
            try:
                doc = run_cell(cell, ctx)
            except BudgetExhausted:
                raise
            except Exception as exc:  # noqa: BLE001 - 失败立即记录，不换配置重试
                product["status"] = "failed"
                product["cells"].append(
                    {
                        "id": cell["id"],
                        "T": cell["T"],
                        "row_stride": cell["row_stride"],
                        "error": repr(exc),
                        "ok": False,
                    }
                )
                print(f"  {cell['id']} ERROR {exc!r}", file=sys.stderr)
                _finish(product, budget, t_start, args.out)
                return RC_FAIL
            product["cells"].append(doc)
            if not doc.get("ok", False):
                product["status"] = "failed"
                print(
                    f"  {cell['id']} 契约/数值不过 → 立即停止（不换配置重试）",
                    file=sys.stderr,
                )
                _finish(product, budget, t_start, args.out)
                return RC_FAIL
            _print_cell(doc, arms)
        if args.baseline_only:
            product["status"] = "baseline-only"
        _finish(product, budget, t_start, args.out)
        return RC_OK
    except BudgetExhausted as exc:
        product["status"] = "time-limited"
        product["budget"] = budget.as_dict()
        print(
            f"墙钟软预算耗尽（{exc}）→ cell 边界停止，不重试；硬上限看外层 timeout",
            file=sys.stderr,
        )
        _finish(product, budget, t_start, args.out)
        return RC_BUDGET
    except Exception as exc:  # noqa: BLE001 - 工具错也要留产物
        product["status"] = "failed"
        product["error"] = repr(exc)
        print(f"工具失败: {exc!r}", file=sys.stderr)
        _finish(product, budget, t_start, args.out)
        return RC_FAIL


def _print_cell(doc: dict, arms: tuple[str, ...]) -> None:
    head = (
        f"  {doc['id']:16s} {str(doc.get('phase')):9s} "
        f"[{doc.get('shape_role')}] "
        f"in={doc['in_shape']} stride={doc['in_stride']}"
    )
    perf = doc.get("perf")
    if not perf:
        print(f"{head}  numeric={ {k: v['ok'] for k, v in doc['numeric'].items()} }")
        return
    row = "  ".join(f"{a}={perf['arms'][a]['us_median']:.2f}us" for a in arms)
    delta = perf.get("delta_pct")
    paired = perf.get("paired_delta_pct_median")
    extra = "" if delta is None else f"  paired_median={paired}%"
    print(f"{head}  {row}{extra}")


def _finish(product: dict, budget: Budget, t_start: float, out: str) -> None:
    product["budget"] = {
        **budget.as_dict(),
        "enforcement": "soft: cell 边界；硬上限由外层 timeout 保护，单进程不重试",
        "elapsed_s": round(time.monotonic() - t_start, 2),
    }
    if out:
        with open(out, "w") as f:
            json.dump(product, f, indent=1, ensure_ascii=False)
        print(f"saved -> {out}  status={product['status']}")


if __name__ == "__main__":
    sys.exit(main())
