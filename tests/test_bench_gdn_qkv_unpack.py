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

"""CPU 门禁：``tools/bench_gdn_qkv_unpack.py``（GDN QKV unpack L0 配对 A/B）。

覆盖四类必须先证明在 CPU 上成立的性质（真机 L0 在实现 agent 防护到位后由主会话执行）：

1. **配对调度与计时口径**：偶数轮臂序正序、奇数轮反序；每轮两臂各测一次；计时 scope
   元数据必须逐字为 ``npu_event_elapsed_eager_chain_including_host_submission_gaps``，
   且显式声明不测 FULL/图模式与 L4；
2. **输出契约**：同一 backing + 片段不重叠 + shape/stride 同 + 输入无修改；
   bitwise 比较必须区分 ±0.0 与 NaN 位图（含符号位）；
3. **失败 rc**：候选缺失 → 1（不回落基线、不产 A/B）；候选抛错/契约不过 → 1
   且**立即停**（不换配置重试）；``--baseline-only`` → 0 且产物显式标记无 A/B；
4. **预算**：只在 cell 边界软停（不打断编译/校验/测量），rc 5 + status=time-limited，
   留完整 cell、不留半截 cell 数字。

除「端到端一次真实 CLI」与「派发接线」两条走子进程外，其余直接调 ``main()``
（``sys.exit`` 的返回值就是 rc），因为每次子进程都要重付 ~10s 的 torch import。

形状范围：只有「已声明相位 且 行跨距=4096」算 measured-shape，其余（T=1/2/17、6176
行跨距）只作 synthetic-coverage，不混作生产范围。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="bench tool needs torch")

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bench_gdn_qkv_unpack.py"
LADDER = ROOT / "tools" / "bench-ladder.sh"

GOOD_CANDIDATE = """
import torch


def unpack_mixed_qkv(mixed_qkv):
    q_dim, k_dim, v_dim, head_k, head_v = 1024, 1024, 2048, 128, 128
    tokens = mixed_qkv.shape[0]
    qs, ks = tokens * q_dim, tokens * k_dim
    fused = torch.empty(qs + ks + tokens * v_dim, dtype=mixed_qkv.dtype,
                        device=mixed_qkv.device)
    q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    fused[0:qs] = q.reshape(-1)
    fused[qs:qs + ks] = k.reshape(-1)
    fused[qs + ks:] = v.reshape(-1)
    return (fused[0:qs].view(1, tokens, -1, head_k),
            fused[qs:qs + ks].view(1, tokens, -1, head_k),
            fused[qs + ks:].view(1, tokens, -1, head_v))
"""

SEPARATE_BUFFERS_CANDIDATE = """
import torch


def unpack_mixed_qkv(mixed_qkv):
    q_dim, k_dim, v_dim, head_k, head_v = 1024, 1024, 2048, 128, 128
    tokens = mixed_qkv.shape[0]
    q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    return (q.reshape(-1).clone().view(1, tokens, -1, head_k),
            k.reshape(-1).clone().view(1, tokens, -1, head_k),
            v.reshape(-1).clone().view(1, tokens, -1, head_v))
"""

SWAPPED_QKV_CANDIDATE = """
import torch


def unpack_mixed_qkv(mixed_qkv):
    q_dim, k_dim, v_dim, head_k, head_v = 1024, 1024, 2048, 128, 128
    tokens = mixed_qkv.shape[0]
    q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    fused = torch.cat([k.reshape(-1), q.reshape(-1), v.reshape(-1)], dim=0)
    qs, ks = tokens * q_dim, tokens * k_dim
    return (fused[0:qs].view(1, tokens, -1, head_k),
            fused[qs:qs + ks].view(1, tokens, -1, head_k),
            fused[qs + ks:].view(1, tokens, -1, head_v))
"""

RAISING_CANDIDATE = """
def unpack_mixed_qkv(mixed_qkv):
    raise RuntimeError("candidate is not implemented yet")
"""

SLOW_CANDIDATE = """
import time

import torch


def unpack_mixed_qkv(mixed_qkv):
    time.sleep(2.0)
    q_dim, k_dim, v_dim, head_k, head_v = 1024, 1024, 2048, 128, 128
    tokens = mixed_qkv.shape[0]
    qs, ks = tokens * q_dim, tokens * k_dim
    fused = torch.empty(qs + ks + tokens * v_dim, dtype=mixed_qkv.dtype,
                        device=mixed_qkv.device)
    q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    fused[0:qs] = q.reshape(-1)
    fused[qs:qs + ks] = k.reshape(-1)
    fused[qs + ks:] = v.reshape(-1)
    return (fused[0:qs].view(1, tokens, -1, head_k),
            fused[qs:qs + ks].view(1, tokens, -1, head_k),
            fused[qs + ks:].view(1, tokens, -1, head_v))
"""


@pytest.fixture(scope="module")
def tool():
    name = "bench_gdn_qkv_unpack"
    spec = importlib.util.spec_from_file_location(name, TOOL)
    module = importlib.util.module_from_spec(spec)
    # dataclass 解析注解要经 sys.modules 找到模块本身
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _reset_dims(tool):
    tool._DIMS = tool.Dims()
    yield
    tool._DIMS = tool.Dims()


def write_candidate(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return f"{path}:unpack_mixed_qkv"


def run_inprocess(tool, tmp_path: Path, *args: str, out_name: str = "product.json"):
    """直接调 main()（rc 与子进程 sys.exit 一致），返回 (rc, product|None)。"""
    out = tmp_path / out_name
    rc = tool.main([*args, "--out", str(out)])
    doc = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
    return rc, doc


SMALL = (
    "--device",
    "cpu",
    "--tokens",
    "2",
    "--row-strides",
    "4096",
    "--iters",
    "2",
    "--repeats",
    "1",
    "--warmup",
    "0",
)
# .py 假候选都是自包含的 fn(mixed_qkv) 契约
PACKED = ("--candidate-arg-style", "packed")


# ------------------------------------------------------------------ 配对调度
def test_ab_schedule_alternates_order(tool) -> None:
    sched = tool.ab_schedule(("baseline", "candidate"), 3)
    assert sched == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
    ]
    assert tool.ab_schedule(("baseline",), 2) == [["baseline"], ["baseline"]]


def test_timer_scope_is_explicit_event_chain(tool) -> None:
    """设备轴不是纯 kernel 时长：scope 逐字声明含 host 供给空洞的 eager 链。"""
    assert (
        tool.TIMER_SCOPE_NPU
        == "npu_event_elapsed_eager_chain_including_host_submission_gaps"
    )
    row = tool.timed(lambda: None, 3, "cpu")
    assert row["timer_scope"] == tool.TIMER_SCOPE_CPU
    assert row["iters"] == 3
    assert row["event_elapsed_us_total"] is None
    assert row["us_per_iter_event_elapsed"] is None
    fake = [{"us_per_iter_event_elapsed": 4.0, "us_host_total": 9.0}] * 3
    med, key = tool.median_metric(fake, "npu")
    assert key == tool.METRIC_NPU and med == 4.0
    med, key = tool.median_metric(fake, "cpu")
    assert key == tool.METRIC_CPU and med == 9.0


def test_shape_role_splits_measured_from_synthetic(tool) -> None:
    assert tool.shape_role(16, 4096) == tool.SHAPE_MEASURED
    assert tool.shape_role(2048, 4096) == tool.SHAPE_MEASURED
    for tokens, stride in ((1, 4096), (2, 4096), (17, 4096), (16, 6176)):
        assert tool.shape_role(tokens, stride) == tool.SHAPE_SYNTHETIC, (tokens, stride)


# ------------------------------------------------------------------ 数值/契约
def test_bitwise_comparator_distinguishes_zero_sign_and_nan(tool) -> None:
    base = torch.zeros(4, dtype=torch.bfloat16)
    same = base.clone()
    assert tool.compare_tensors(base, same, "bfloat16")["bitwise_ok"] is True

    neg_zero = base.clone()
    neg_zero[0] = -0.0
    assert tool.compare_tensors(base, neg_zero, "bfloat16")["bitwise_ok"] is False

    nan_pos = base.clone()
    nan_pos[1] = float("nan")
    nan_neg = nan_pos.clone()
    nan_neg.view(torch.int16)[1] = tool._signed(0xFFC0, torch.int16)
    assert tool.compare_tensors(nan_pos, nan_neg, "bfloat16")["bitwise_ok"] is False
    assert tool.bit_stats(nan_neg, "bfloat16")["nan_count"] == 1


def test_contract_accepts_single_backing(tool) -> None:
    dims = tool.Dims()
    outs = tool.baseline_unpack(torch.randn(4, dims.width, dtype=torch.bfloat16), dims)
    doc = tool.output_contract(outs, 4, dims)
    assert doc["ok"] is True
    assert doc["same_backing"] is True
    assert doc["segments_disjoint"] is True


def test_contract_rejects_separate_buffers(tool) -> None:
    dims = tool.Dims()
    outs = (
        torch.zeros(1, 4, 8, 128, dtype=torch.bfloat16),
        torch.zeros(1, 4, 8, 128, dtype=torch.bfloat16),
        torch.zeros(1, 4, 16, 128, dtype=torch.bfloat16),
    )
    doc = tool.output_contract(outs, 4, dims)
    assert doc["same_backing"] is False
    assert doc["ok"] is False


def test_baseline_matches_slice_truth_for_both_layouts(tool) -> None:
    dims = tool.Dims()
    for row_stride, layout in ((dims.width, "contiguous"), (6176, "strided-view")):
        x = tool.build_input(4, row_stride, "bfloat16", 7, True, "cpu")
        assert x.stride() == (row_stride, 1), layout
        base = tool.baseline_unpack(x, dims)
        truth = tool.truth_unpack(x, dims)
        cmp = tool.compare_arm(truth, base, "bfloat16", False)
        assert cmp["ok"] is True, (layout, cmp)
        # T=4 覆盖 specials 的前 5 个位置：qnan/neg_qnan 落在 q 的前两行
        stats = tool.bit_stats(base[0], "bfloat16")
        assert stats["nan_count"] == 2 and stats["neg_zero_count"] == 1


def test_input_specials_are_injected(tool) -> None:
    dims = tool.Dims()
    x = tool.build_input(8, dims.width, "bfloat16", 0, True, "cpu")
    flat = tool.int_view_cpu(x, "bfloat16")
    bits = tool.SPECIAL_BITS["bfloat16"]
    for offset, name in (
        (0, "pos_zero"),
        (1, "neg_zero"),
        (dims.width + 2, "qnan"),
        (2 * dims.width + 3, "neg_qnan"),
    ):
        assert int(flat[offset]) == tool._signed(bits[name], torch.int16)


# ------------------------------------------------------------------ 端到端一次
def test_cli_ok_product_carries_shape_facts(tmp_path: Path) -> None:
    """真子进程跑一次完整 CLI：产物自包含 shape/相位/callsite/标签/代码/计时 scope。"""
    cand = write_candidate(tmp_path, "good", GOOD_CANDIDATE)
    out = tmp_path / "ok.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--candidate",
            cand,
            *PACKED,
            "--device",
            "cpu",
            "--tokens",
            "2,16",
            "--row-strides",
            "4096,6176",
            "--iters",
            "2",
            "--repeats",
            "3",
            "--warmup",
            "0",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))

    assert doc["schema"] == "opt-ops-l0-gdn-qkv-unpack.v1"
    assert doc["status"] == "ok"
    assert doc["arms"]["requested"] == ["baseline", "candidate"]
    assert len(doc["code"]["tool_sha256"]) == 64
    assert doc["hardware"]["device"] == "cpu"
    assert doc["fidelity"]["model"] == "Qwen3.5-35B-A3B"
    assert doc["fidelity"]["precision"]["perf"] == "bfloat16"
    assert doc["fidelity"]["precision"]["supplement_correctness"] == [
        "float16",
        "float32",
    ]
    assert doc["fidelity"]["weight_format"]["do_not_infer"] == "NZ/ND"
    assert doc["caliber"]["format"]["declared"] == "ND"
    assert "890-924" in doc["goal"]["callsite"]["canonical"]

    # 计时口径：Event 是 eager 链时长（含 host 供给空洞），不冒充 FULL/L4
    timing = doc["caliber"]["timing"]
    assert (
        timing["timer_scope"]["npu"]
        == "npu_event_elapsed_eager_chain_including_host_submission_gaps"
    )
    assert timing["device_available"] is False
    assert "FULL/图模式时长" in timing["does_not_measure"]
    assert any("eager" in note for note in doc["not_claimed"])

    # callsite 读安装树（479/480/535），不抄仓内 clone 的 532/533
    callers = doc["goal"]["callsite"]["callers"]
    assert len(callers) == 3
    assert any(":479 " in c for c in callers)
    assert any(":535 " in c for c in callers)
    assert not any(":532" in c or ":533" in c for c in callers)
    assert "532/533" in doc["goal"]["callsite"]["note"]

    # 代码溯源：tmp 候选在仓外 → 不 hash、明确标 inside_repo False
    code_cand = doc["code"]["candidate"]
    assert code_cand["inside_repo"] is False
    assert code_cand["package"]["inside_repo"] is False
    assert code_cand["package"]["files"] == {}

    labels = {
        layout["row_stride"]: layout["label"]
        for layout in doc["caliber"]["input_layouts"]
    }
    assert labels == {4096: "measured-layout", 6176: "synthetic-supplementary"}
    scope = doc["shape_scope"]
    assert scope["measured"] == ["T16-rs4096"]
    assert sorted(scope["synthetic"]) == ["T16-rs6176", "T2-rs4096", "T2-rs6176"]
    assert "not a production-shape gain" in scope["synthetic_claim"]

    cells = {(c["T"], c["row_stride"]): c for c in doc["cells"]}
    assert set(cells) == {(2, 4096), (2, 6176), (16, 4096), (16, 6176)}
    c16 = cells[(16, 4096)]
    assert c16["phase"] == "decode" and c16["phase_source"] == "declared"
    assert c16["shape_role"] == "measured-shape"
    assert cells[(16, 6176)]["shape_role"] == "synthetic-coverage"
    assert c16["in_shape"] == [16, 4096] and c16["in_stride"] == [4096, 1]
    assert cells[(16, 6176)]["in_stride"] == [6176, 1]
    assert cells[(2, 4096)]["phase"] is None
    assert set(c16["numeric"]) == {"bfloat16", "float16", "float32"}
    for dtype_doc in c16["numeric"].values():
        assert dtype_doc["ok"] is True
        assert dtype_doc["input_unmodified"] is True
        assert dtype_doc["contract"]["candidate"]["same_backing"] is True

    perf = c16["perf"]
    assert perf["dtype"] == "bfloat16"
    assert (
        perf["timer_scope"]
        == "host_wall_perf_counter_elapsed_including_host_submission"
    )
    assert perf["shape_role"] == "measured-shape"
    assert perf["schedule"] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
    ]
    for arm in ("baseline", "candidate"):
        assert len(perf["arms"][arm]["runs"]) == 3
    assert len(perf["paired_delta_pct"]) == 3
    assert perf["arms"]["baseline"]["metric"] == "us_host_total"
    assert perf["post_measure_input_unmodified"] is True
    assert "soft: cell 边界" in doc["budget"]["enforcement"]


def test_cli_baseline_only_declares_no_ab(tool, tmp_path: Path) -> None:
    rc, doc = run_inprocess(
        tool, tmp_path, "--baseline-only", *SMALL, out_name="base.json"
    )
    assert rc == 0
    assert doc["status"] == "baseline-only"
    assert doc["arms"]["requested"] == ["baseline"]
    assert doc["arms"]["candidate_spec"] is None
    assert "candidate" not in doc["cells"][0]["perf"]["arms"]
    assert "delta_pct" not in doc["cells"][0]["perf"]


def test_cli_candidate_missing_is_rc1_without_numbers(tool, tmp_path: Path) -> None:
    rc, doc = run_inprocess(
        tool,
        tmp_path,
        "--candidate",
        "no_such_module_xyz:unpack",
        *SMALL,
        out_name="missing.json",
    )
    assert rc == 1
    assert doc["status"] == "candidate-missing"
    assert doc["cells"] == []
    assert doc["candidate"]["error"]


def test_cli_contract_violation_is_rc1(tool, tmp_path: Path) -> None:
    cand = write_candidate(tmp_path, "sep", SEPARATE_BUFFERS_CANDIDATE)
    rc, doc = run_inprocess(
        tool, tmp_path, "--candidate", cand, *PACKED, *SMALL, out_name="sep.json"
    )
    assert rc == 1
    assert doc["status"] == "failed"
    entry = doc["cells"][0]["numeric"]["bfloat16"]
    assert entry["ok"] is False
    assert entry["contract"]["candidate"]["same_backing"] is False


def test_cli_value_mismatch_is_rc1(tool, tmp_path: Path) -> None:
    cand = write_candidate(tmp_path, "swap", SWAPPED_QKV_CANDIDATE)
    rc, doc = run_inprocess(
        tool, tmp_path, "--candidate", cand, *PACKED, *SMALL, out_name="swap.json"
    )
    assert rc == 1
    entry = doc["cells"][0]["numeric"]["bfloat16"]
    assert entry["candidate_vs_baseline"]["ok"] is False
    assert entry["candidate_vs_baseline"]["parts"][0]["bitwise_ok"] is False


def test_cli_candidate_error_stops_without_retry(tool, tmp_path: Path) -> None:
    cand = write_candidate(tmp_path, "raise", RAISING_CANDIDATE)
    rc, doc = run_inprocess(
        tool,
        tmp_path,
        "--candidate",
        cand,
        *PACKED,
        "--device",
        "cpu",
        "--tokens",
        "2,4",
        "--row-strides",
        "4096,6176",
        "--iters",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        out_name="raise.json",
    )
    assert rc == 1
    assert doc["status"] == "failed"
    assert len(doc["cells"]) == 1, "失败必须立即停（不换配置重试）"
    assert "candidate is not implemented" in doc["cells"][0]["error"]


def test_cli_dims_arg_style_is_supported(tool, tmp_path: Path) -> None:
    """候选也可收 shape 关键字（单入口两种 documented 契约，不静默重试）。"""
    body = GOOD_CANDIDATE.replace(
        "def unpack_mixed_qkv(mixed_qkv):",
        "def unpack_mixed_qkv(mixed_qkv, q_dim=1024, k_dim=1024, v_dim=2048,\n"
        "                     head_k_dim=128, head_v_dim=128):",
        1,
    )
    body = body.replace(
        "    q_dim, k_dim, v_dim, head_k, head_v = 1024, 1024, 2048, 128, 128\n",
        "    head_k, head_v = head_k_dim, head_v_dim\n",
        1,
    )
    cand = write_candidate(tmp_path, "dims", body)
    rc, doc = run_inprocess(
        tool,
        tmp_path,
        "--candidate",
        cand,
        "--candidate-arg-style",
        "dims",
        *SMALL,
        out_name="dims.json",
    )
    assert rc == 0, doc
    assert doc["status"] == "ok"
    assert doc["cells"][0]["numeric"]["bfloat16"]["ok"] is True


# ------------------------------------------------------------------ 预算
def test_cli_budget_stop_is_time_limited(tool, tmp_path: Path) -> None:
    rc, doc = run_inprocess(
        tool,
        tmp_path,
        "--baseline-only",
        "--device",
        "cpu",
        "--tokens",
        "2",
        "--row-strides",
        "4096",
        "--iters",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--wall-budget-min",
        "0.0001",
        out_name="budget.json",
    )
    assert rc == 5
    assert doc["status"] == "time-limited"
    assert doc["budget"]["exhausted"] is True
    assert doc["budget"]["stop_reason"]
    assert doc["cells"] == []
    assert "elapsed_s" in doc["budget"]


def test_cli_budget_stop_mid_run_keeps_only_complete_cells(
    tool, tmp_path: Path
) -> None:
    """软预算只在 cell 边界生效：cell 内的编译/校验/测量绝不被截断。"""
    cand = write_candidate(tmp_path, "slow", SLOW_CANDIDATE)
    rc, doc = run_inprocess(
        tool,
        tmp_path,
        "--candidate",
        cand,
        *PACKED,
        "--device",
        "cpu",
        "--tokens",
        "2,4",
        "--row-strides",
        "4096",
        "--iters",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--wall-budget-min",
        "0.05",
        out_name="slow.json",
    )
    assert rc == 5
    assert doc["status"] == "time-limited"
    assert doc["budget"]["exhausted"] is True
    assert "before cell" in doc["budget"]["stop_reason"]
    # 后到 cell 不启动；已记的 cell 必须完整（有 perf 且 ok）
    assert 0 < len(doc["cells"]) < 2
    for cell in doc["cells"]:
        assert cell["ok"] is True
        assert "perf" in cell


def test_cli_usage_errors_are_rc2(tool, tmp_path: Path) -> None:
    rc, doc = run_inprocess(
        tool,
        tmp_path,
        "--baseline-only",
        "--device",
        "cpu",
        "--tokens",
        "0",
        "--iters",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        out_name="u1.json",
    )
    assert rc == 2 and doc is None
    rc, doc = run_inprocess(
        tool,
        tmp_path,
        "--baseline-only",
        "--device",
        "cpu",
        "--tokens",
        "2",
        "--row-strides",
        "1024",
        "--iters",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        out_name="u2.json",
    )
    assert rc == 2 and doc is None


def test_default_candidate_wiring_resolves_on_cpu(tool) -> None:
    """默认候选指向仓内单遍实现；解析（不调用）在 CPU 上必须成立且记录来源 sha。"""
    pytest.importorskip(
        "vllm_ascend_split_batch.gdn_qkv_unpack", reason="候选实现包尚未落地"
    )
    assert tool.DEFAULT_CANDIDATE.endswith(":rearrange_mixed_qkv")
    fn, info = tool.resolve_candidate(
        tool.DEFAULT_CANDIDATE, tool.DEFAULT_CANDIDATE_ARG_STYLE, tool.Dims()
    )
    assert callable(fn)
    assert info["callable_module_file"].endswith("gdn_qkv_unpack/api.py")
    assert info["module_file"].endswith("gdn_qkv_unpack/__init__.py")


def test_code_provenance_hashes_whole_candidate_package(tool) -> None:
    """溯源必须覆盖候选包全部 .py（含 kernel），且确认路径属本仓。"""
    pytest.importorskip(
        "vllm_ascend_split_batch.gdn_qkv_unpack", reason="候选实现包尚未落地"
    )
    _, info = tool.resolve_candidate(
        tool.DEFAULT_CANDIDATE, tool.DEFAULT_CANDIDATE_ARG_STYLE, tool.Dims()
    )
    prov = tool.code_provenance(info)
    doc = prov["candidate"]
    assert doc["inside_repo"] is True and doc["callable_inside_repo"] is True
    pkg = doc["package"]
    assert pkg["inside_repo"] is True
    assert pkg["dir"].startswith(prov["repo"])
    files = pkg["files"]
    assert set(files) == {
        "src/vllm_ascend_split_batch/gdn_qkv_unpack/__init__.py",
        "src/vllm_ascend_split_batch/gdn_qkv_unpack/api.py",
        "src/vllm_ascend_split_batch/gdn_qkv_unpack/kernels.py",
        "src/vllm_ascend_split_batch/gdn_qkv_unpack/reference.py",
    }
    for rel, digest in files.items():
        assert rel.endswith(".py") and len(digest) == 64
    assert files[
        "src/vllm_ascend_split_batch/gdn_qkv_unpack/kernels.py"
    ] == tool.sha256_file(f"{pkg['dir']}/kernels.py")


# ------------------------------------------------------------------ 派发接线
def test_ladder_dispatches_gdn_qkv_unpack() -> None:
    body = LADDER.read_text(encoding="utf-8")
    assert 'if [ "$op" = "GdnQkvUnpack" ]' in body
    assert "bench_gdn_qkv_unpack.py" in body
    # 其余算子仍走 bench-l0.py（行为不变）
    assert '"$PY" "$SCRIPTS/bench-l0.py" --op "$op" "$@"' in body
    proc = subprocess.run(
        ["bash", str(LADDER), "l0", "GdnQkvUnpack", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "bench_gdn_qkv_unpack" in proc.stdout


def test_paired_changes_use_candidate_over_baseline(tool):
    assert tool.paired_changes([100.0, 200.0, 0.0], [50.0, 300.0, 1.0]) == [
        -50.0,
        50.0,
        None,
    ]
    with pytest.raises(ValueError):
        tool.paired_changes([100.0], [])


def shape_source_fixture(tmp_path, model="Qwen3.5-35B-A3B", precision="bf16"):
    import csv

    rank = tmp_path / "rank0_ascend_pt"
    output = rank / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    kernel = output / "kernel_details.csv"
    with kernel.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Input Shapes", "Output Shapes", "Input Data Types"])
        writer.writerow(
            ["ConcatD", "1024;1024;2048", "4096", "DT_BF16;DT_BF16;DT_BF16"]
        )
        writer.writerow(
            [
                "RecurrentGatedDeltaRule",
                "1,8,128;1,8,128;1,16,128;1,16",
                "1,16,128",
                "DT_BF16",
            ]
        )
    opstat = tmp_path / "source-opstat.json"
    opstat.write_text(
        json.dumps(
            {"asc_dir": str(rank), "cohort": {"model": model, "precision": precision}}
        )
    )
    return opstat, kernel


def test_shape_source_binds_new_t1_run_without_relabelling_defaults(tool, tmp_path):
    source, _ = shape_source_fixture(tmp_path)
    evidence = tool.decode_shape_source(str(source), tool.Dims())
    assert evidence["matched_calls"] == 1
    assert evidence["kernel_details"]["sha256"]
    assert tool.shape_role(1, 4096) == tool.SHAPE_SYNTHETIC
    assert tool.shape_role(1, 4096, evidence) == tool.SHAPE_MEASURED
    assert tool.shape_role(1, 6176, evidence) == tool.SHAPE_SYNTHETIC
    candidate = write_candidate(tmp_path, "bound_candidate", GOOD_CANDIDATE)
    rc, doc = run_inprocess(
        tool,
        tmp_path,
        *SMALL,
        *PACKED,
        "--tokens",
        "1",
        "--candidate",
        candidate,
        "--shape-source-opstat",
        str(source),
    )
    assert rc == 0
    assert doc["cells"][0]["phase"] == "decode"
    assert doc["cells"][0]["phase_source"] == "shape_source"
    assert doc["cells"][0]["perf"]["shape_role"] == tool.SHAPE_MEASURED
    assert doc["shape_scope"]["measured"] == ["T1-rs4096"]


@pytest.mark.parametrize(
    "model,precision", [("other", "bf16"), ("Qwen3.5-35B-A3B", "fp16")]
)
def test_shape_source_rejects_weight_mismatch(tool, tmp_path, model, precision):
    source, _ = shape_source_fixture(tmp_path, model, precision)
    with pytest.raises(ValueError, match="model/precision mismatch"):
        tool.decode_shape_source(str(source), tool.Dims())


def test_shape_source_requires_matching_decode_and_qkv_shapes(tool, tmp_path):
    source, kernel = shape_source_fixture(tmp_path)
    kernel.write_text(kernel.read_text().replace("1,8,128", "16,8,128"))
    with pytest.raises(ValueError, match="matched T1"):
        tool.decode_shape_source(str(source), tool.Dims())
