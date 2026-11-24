"""Native routed-experts request probe: self-check now, cards only after review.

Default invocation (no arguments) is a read-only self-check: it re-verifies the
SHA256 fingerprints registered in ``docs/optimization-history/gmm1-swiglu-screen.json``
for the archived diagnostic config and the reference responses, cross-checks both
host commits and the collection shape, and prints the pending-review state plus
the single next step. It imports no tensor stack, writes nothing and touches no
device. ``--run`` is the future device-bound path: it is refused until
``docs/optimization-next.json`` registers this exact plan in ``reviewed_plan``
(path plus sha256, re-checked with :func:`tools.operator_tools.next.check_hash`);
no flag bypasses that gate and passing tests never counts as review.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.operator_tools import next as handoff  # noqa: E402
from tools.operator_tools import serving_collect as collect  # noqa: E402

SCREEN = ROOT / "docs/optimization-history/gmm1-swiglu-screen.json"
HANDOFF = ROOT / "docs/optimization-next.json"
#: 审批只认本计划；换路径等于换一份待评审文档。
PLAN = "experiments/native-route-probe/EXPERIMENT_PLAN.md"
ROUTE_FLAG = "--enable-return-routed-experts"
ARCHIVES = Path("/root/vllm-workspace/benchmark-archives")
#: --run 只绑定共用 serving-longrun.lock 的权威 checkout；独立 clone 的同名锁
#: 文件互不相干，所以不能只查 .git 与 common dir 是否同一棵树。
AUTHORITATIVE_ROOT = Path("/root/vllm-workspace/vllm-ascend-split-batch-hust")
LOCKFILE = ROOT / ".openbitfun/serving-longrun.lock"
FROZEN_TOOL = re.compile(r"native_route_probe\.py sha256 ([0-9a-f]{64})")
HOST_KEYS = {"vllm", "vllm-ascend"}
COMMIT = re.compile(r"[0-9a-f]{40}")
WARMUP_PROMPT = "Explain RMSNorm briefly."
RUN_BUDGET_S = 1350
#: 只复用登记过指纹的只读输入；登记映射缺一即拒绝。
DIAGNOSTIC_INPUTS = (
    "config.json",
    "diagnostic-plan.json",
    "host_preflight.json",
    "route-status.json",
)
REFERENCE_INPUTS = (
    "collection_manifest.json",
    "round_00_off/invocation.json",
    "round_00_off/profile_responses.json",
)


def load(path) -> object:
    return json.loads(Path(path).read_bytes())


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def archive_dir(uri) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc or not Path(parsed.path).is_absolute():
        raise ValueError(f"需要本机绝对路径的 file URI 归档：{uri}")
    return Path(unquote(parsed.path))


def complete_hosts(hosts, where) -> dict:
    if set(hosts) != HOST_KEYS or not all(
        isinstance(value, str) and COMMIT.fullmatch(value) for value in hosts.values()
    ):
        raise ValueError(f"{where} 的双宿主提交不完整或不合规：{hosts}")
    return dict(hosts)


def port_of(command) -> int:
    if "--port" not in command:
        raise ValueError("服务命令缺 --port")
    return int(command[command.index("--port") + 1])


def merged_invocation_ok(config, invocation) -> None:
    """Base command + guard flags, minus the single added route flag, must equal
    the historical reference window's invocation without its profiler part."""
    if "--profiler-config" not in invocation:
        raise ValueError("历史参考 invocation 缺 --profiler-config 截断点")
    if ROUTE_FLAG not in config["guard_server_args"]:
        raise ValueError(f"诊断配置 guard_server_args 缺路由 flag {ROUTE_FLAG}")
    guard = list(config["guard_server_args"])
    guard.remove(ROUTE_FLAG)  # 诊断窗口在历史工况之上单加的路由 flag
    merged = list(config["server_command"]) + guard
    cut = list(invocation[: invocation.index("--profiler-config")])
    cut[cut.index("--port") + 1] = str(port_of(merged))
    if merged != cut:
        raise ValueError("诊断配置合并后的服务命令与历史参考 invocation 工况不一致")


def environment_ok(config) -> None:
    env = config.get("environment", {})
    entries = env.get("PYTHONPATH", "").split(":")
    if any(not entry for entry in entries):
        raise ValueError("config PYTHONPATH 含空组件；服务解释器搜索路径不可为空")
    if any(
        entry == str(ROOT) or entry.startswith(str(ROOT) + os.sep) for entry in entries
    ):
        raise ValueError("config PYTHONPATH 含仓库源码路径，会遮蔽已安装宿主")
    if "ASCEND_RT_VISIBLE_DEVICES" in env:
        raise ValueError("config 携带历史设备绑定 ASCEND_RT_VISIBLE_DEVICES，须先清除")


def reference_rows_ok(responses, shape) -> None:
    count = int(shape["profile_requests"])
    if not isinstance(responses, list) or len(responses) != count:
        raise ValueError(f"参考响应条数与 profile_requests={count} 不符")
    ids = [row.get("id") for row in responses]
    if (
        any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != count
    ):
        raise ValueError("参考响应 id 缺失、为空或重复")
    for row in responses:
        if len(row["choices"]) != 1:
            raise ValueError("参考响应必须是单 choice；多 choice 参考档不可比")
        choice = row["choices"][0]
        if not isinstance(choice.get("text"), str) or not choice["text"]:
            raise ValueError("参考响应缺非空贪心文本")
        if choice.get("routed_experts") not in (None, [], ""):
            raise ValueError("参考响应携带 routed_experts；原窗口路由 flag 未开启")
        usage = row["usage"]
        if (
            usage["completion_tokens"] != shape["profile_tokens"]
            or usage["prompt_tokens"] < 1
        ):
            raise ValueError("参考响应 token 记录与工况不符")


def gather(next_record) -> dict:
    """Verify the registered read-only inputs and return everything the probe reuses."""
    hosts = complete_hosts(next_record["host_commits"], "docs/optimization-next.json")
    screen = load(SCREEN)
    progress = screen["next_diagnostic"]["progress"]
    diagnostic = archive_dir(progress["archive_uri"])
    reference = archive_dir(screen["archive_uri"])
    for mapping, names, archive in (
        (progress["sha256"], DIAGNOSTIC_INPUTS, diagnostic),
        (screen["observations"]["sha256"], REFERENCE_INPUTS, reference),
    ):
        for name in names:
            if name not in mapping:
                raise ValueError(f"登记指纹缺少 {name}")
            handoff.check_hash(archive, name, mapping[name])
    config = load(diagnostic / "config.json")
    plan = load(diagnostic / "diagnostic-plan.json")
    expected = plan["expected"]
    manifest = load(reference / "collection_manifest.json")
    invocation = load(reference / "round_00_off" / "invocation.json")["command"]
    responses = load(reference / "round_00_off" / "profile_responses.json")
    for where, other in (
        ("gmm1-swiglu-screen.json", screen["host_commits"]),
        ("collection_manifest.json", manifest["host_commits"]),
        ("config.json", config["host_commits"]),
        ("diagnostic-plan.json", expected["host_commits"]),
    ):
        if complete_hosts(other, where) != hosts:
            raise ValueError(f"{where} 的双宿主与 docs/optimization-next.json 不一致")
    shape = config["collection_shape"]
    if shape != manifest["collection_shape"] or shape != expected["shape"]:
        raise ValueError(
            "诊断配置、参考 manifest 与诊断计划的工况（collection_shape）不一致"
        )
    if ROUTE_FLAG not in config["guard_server_args"]:
        raise ValueError(f"诊断配置 guard_server_args 缺路由 flag {ROUTE_FLAG}")
    if plan["single_server_flag"] != ROUTE_FLAG:
        raise ValueError("登记诊断计划的路由 flag 与配置不一致")
    counts = {
        key: int(expected[key]) for key in ("model_layers", "expert_count", "top_k")
    }
    if any(value < 1 for value in counts.values()):
        raise ValueError(f"诊断计划的层/专家数/top-k 不合规：{counts}")
    if config["server_command"][0] != config["python"]:
        raise ValueError("server_command 必须使用配置内的服务解释器")
    merged_invocation_ok(config, invocation)
    environment_ok(config)
    reference_rows_ok(responses, shape)
    preflight = load(diagnostic / "host_preflight.json")
    if preflight["returncode"] != 0:
        raise ValueError("归档内宿主预检回执非零")
    observed = json.loads(preflight["stdout"])
    if any(observed[host]["revision"] != commit for host, commit in hosts.items()):
        raise ValueError("归档内宿主预检的 revision 与登记双宿主不符")
    if not Path(config["python"]).is_file():
        raise FileNotFoundError(f"服务解释器不存在：{config['python']}")
    if not Path(config["model_path"]).is_dir():
        raise FileNotFoundError(f"模型路径不存在：{config['model_path']}")
    return {
        "hosts": hosts,
        "config": config,
        "expected": counts,
        "shape": shape,
        "responses": responses,
        "reference": reference,
        "diagnostic": diagnostic,
        "invocation_port": port_of(invocation),
        "old_stage": load(diagnostic / "route-status.json").get("stage"),
    }


def plan_file() -> Path:
    path = ROOT / PLAN
    if not path.is_file():
        raise FileNotFoundError(f"缺待评审执行计划 {PLAN}")
    return path


def frozen_matches(path) -> None:
    frozen = FROZEN_TOOL.search(path.read_text())
    if not frozen:
        raise ValueError(f"{PLAN} 缺源码冻结摘要（native_route_probe.py sha256 …）")
    if frozen.group(1) != file_sha(Path(__file__).resolve()):
        raise ValueError("工具源码与计划冻结摘要不符；先重新冻结并重新评审")


def approved_plan(next_record) -> dict:
    """审批只从 docs/optimization-next.json 的 reviewed_plan 读取并逐字核对。"""
    entry = next_record.get("reviewed_plan")
    if entry is None:
        raise ValueError(
            "docs/optimization-next.json 未登记 reviewed_plan；--run 未获批，拒绝执行"
        )
    if not isinstance(entry, dict) or entry.get("path") != PLAN:
        raise ValueError(f"reviewed_plan 必须指向 {PLAN}，当前为 {entry!r}")
    sha = entry.get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("reviewed_plan 需要本计划的完整 sha256")
    return {"path": PLAN, "sha256": sha, "file": handoff.check_hash(ROOT, PLAN, sha)}


def report(inputs, next_record) -> int:
    plan = plan_file()
    registered = next_record.get("reviewed_plan") is not None
    if registered:
        approved_plan(next_record)
    frozen_matches(plan)
    config = inputs["config"]
    print(
        "状态：只读自检通过；未占卡、未起服、未采集，本探针还没有任何真实实测。",
        "已核验：本计划引用的输入指纹全部匹配、双宿主一致（optimization-next、"
        "筛查记录、诊断配置、参考 manifest、诊断计划）、工况三源一致"
        f"（诊断配置、参考 manifest、诊断计划）、guard 含 {ROUTE_FLAG}、"
        f"归档宿主预检回执、旧诊断止步 {inputs['old_stage']}（无请求级回执）；"
        "未核验：当前运行时宿主与设备（只在占卡预检）。",
        f"计划：{PLAN}（sha256={file_sha(plan)}；占卡命令、准入与判据都在计划内）。",
        sep="\n",
        flush=True,
    )
    if registered:
        print(
            "唯一动作：按计划占卡命令执行（一次窗口，无扩采）："
            f"{config['python']} -B -m tools.operator_tools.native_route_probe --run",
            flush=True,
        )
    else:
        print(
            "唯一动作：评审这份计划（见上一行链接与 sha256，登记值直接取该行，"
            "不要求手抄长哈希）；无审批不执行占卡命令。",
            flush=True,
        )
    return 0


def stage(archive, name, **fields) -> None:
    path = archive / "route-status.json"
    tmp = path.with_suffix(".tmp")
    collect.save(
        tmp, {"stage": name, "pid": os.getpid(), "updated_at": utc_now(), **fields}
    )
    tmp.replace(path)


def allocate_archive(explicit) -> Path:
    if not ARCHIVES.is_dir():
        raise FileNotFoundError(f"归档根目录不存在：{ARCHIVES}")
    if explicit is not None:
        target = Path(explicit)
        if not target.is_absolute() or target.parent.resolve() != ARCHIVES.resolve():
            raise ValueError(
                f"--archive-dir 必须是 {ARCHIVES} 下不存在的受控子目录：{explicit}"
            )
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = ARCHIVES / f"native-route-probe-{stamp}-p{os.getpid()}"
    target.mkdir(exist_ok=False)  # 唯一、不可覆盖
    return target


def pick_port(explicit, forbidden) -> int:
    if explicit is not None:
        if explicit in forbidden:
            raise ValueError(f"--port 不得复用历史端口 {explicit}")
        return int(explicit)
    for _ in range(5):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            chosen = int(sock.getsockname()[1])
        if chosen not in forbidden:
            return chosen
    raise RuntimeError("未能分配到非历史端口")


def parse_devices(text):
    """`--devices` 只是显式物理卡号的核验输入；格式必须严格，不做字面修复。

    恰好两个纯数字字段（规范化十进制、无空格/空字段/尾逗号/数值重复），
    ``4,,6``、``4,6,``、``,`、带空格或 ``4,04`` 一律拒绝；缺省（None）保持
    不指定、按当前映射自动核对。
    """
    if text is None:
        return None
    items = text.split(",")
    numbers = [int(item) for item in items if item.isdigit() and str(int(item)) == item]
    if len(items) != 2 or len(numbers) != 2 or len(set(numbers)) != 2:
        raise ValueError(
            f"--devices 需要恰好两张物理卡号（严格 N,M 纯数字、无空字段"
            f"/空格/尾逗号/数值重复）：{text!r}"
        )
    return numbers


def npu_text(argv) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{argv[0]} 退出 {proc.returncode}: {proc.stderr.strip()[-200:]}"
        )
    return proc.stdout


def resolve_devices(explicit, archive) -> list[int]:
    """设备号只用于忙卡检查与映射核对，不作运行时可见性掩码。

    本计划只覆盖容器直通两卡（逻辑 0/1）且 ``ASCEND_RT_VISIBLE_DEVICES``
    键未设置的工况。该键只要已设置就拒绝——包括空串、纯逗号、单卡、
    物理/逻辑号，不因字面可转换而放行；也不覆盖、不建议 unset 绕过平台
    分配。平台必须设置时停止执行并回 CPU 另核该映射，不猜物理/逻辑号。
    """
    info = npu_text(["npu-smi", "info"])
    (archive / "npu-before.txt").write_text(info)
    mapping = npu_text(["npu-smi", "info", "-m"])
    (archive / "device-mapping.txt").write_text(mapping)
    pairs = {
        (parts[0], parts[2])
        for line in mapping.splitlines()
        if len(parts := line.split()) > 2 and parts[1] == "0"
    }
    logic = {chip_logic: npu for npu, chip_logic in pairs}
    if len(pairs) != 2 or set(logic) != {"0", "1"}:
        raise ValueError(f"要求恰好两张计算卡映射容器逻辑 0/1；当前 {sorted(pairs)}")
    physical = [int(logic[key]) for key in ("0", "1")]
    if len(set(physical)) != 2:
        raise ValueError(f"映射到逻辑 0/1 的是同一张物理卡 {physical}，拒绝执行")
    if explicit is not None and sorted(explicit) != sorted(physical):
        raise ValueError(
            f"--devices {explicit} 与当前真实映射 {physical} 不符，拒绝猜测"
        )
    for device in physical:
        if f"No running processes found in NPU {device} " not in info:
            raise ValueError(f"NPU {device} 忙或有进程占用，拒绝起服")
    ambient = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if ambient is not None:
        raise ValueError(
            f"ASCEND_RT_VISIBLE_DEVICES 已设置（{ambient!r}）；本计划只覆盖该键"
            "未设置、容器直通两卡（逻辑 0/1）的工况，任何已设置取值一律拒绝，"
            "不覆盖、不建议绕过平台分配。平台必须设置时停止执行并回 CPU 另核"
            "该映射，不猜物理/逻辑号"
        )
    return physical


def decode_ids(row, expected):
    """Decode one routed_experts payload; numpy enters here and only here."""
    choice = row["choices"][0]
    usage = row["usage"]
    payload = choice.get("routed_experts")
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"请求 {row.get('id')!r} 的 routed_experts 为空；不得当成功")
    import numpy as np

    try:
        ids = np.load(
            io.BytesIO(base64.b64decode(payload, validate=True)), allow_pickle=False
        )
    except (ValueError, OSError, EOFError) as exc:
        raise ValueError(f"请求 {row.get('id')!r} 的 NPY 解码失败：{exc}") from exc
    if not isinstance(ids, np.ndarray):
        raise ValueError(f"请求 {row.get('id')!r} 的 routed_experts 不是单个 NPY 数组")
    tokens = usage["prompt_tokens"] + usage["completion_tokens"] - 1
    wanted = (tokens, expected["model_layers"], expected["top_k"])
    if ids.shape != wanted:
        raise ValueError(f"请求 {row.get('id')!r} 形状 {ids.shape} != {wanted}")
    if ids.dtype.kind not in "iu":
        raise ValueError(f"请求 {row.get('id')!r} dtype {ids.dtype} 不是整数")
    if int(ids.min()) < 0 or int(ids.max()) >= expected["expert_count"]:
        raise ValueError(
            f"请求 {row.get('id')!r} 专家 id 超出 [0, {expected['expert_count']})"
        )
    ordered = np.sort(ids, axis=-1)
    if not np.all(ordered[..., 1:] != ordered[..., :-1]):
        raise ValueError(f"请求 {row.get('id')!r} 同一 token×层 top-k 有重复专家")
    return ids


def validate(rows, reference, shape, expected, archive) -> dict:
    count = int(shape["profile_requests"])
    if not isinstance(rows, list) or len(rows) != count or len(rows) != len(reference):
        raise ValueError(f"请求条数与 profile_requests={count} 或参考档不符")
    ids = [row.get("id") for row in rows]
    if (
        any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != count
    ):
        raise ValueError("请求 id 缺失、为空或重复")
    if any(len(row["choices"]) != 1 for row in rows):
        raise ValueError("本窗口响应必须是单 choice")
    summaries, matched = [], []
    for row, ref in zip(rows, reference, strict=True):
        ids_array = decode_ids(row, expected)
        usage = row["usage"]
        if usage["completion_tokens"] != shape["profile_tokens"]:
            raise ValueError(f"请求 {row.get('id')!r} completion_tokens 与工况不符")
        if usage["prompt_tokens"] != ref["usage"]["prompt_tokens"]:
            raise ValueError(
                f"请求 {row.get('id')!r} prompt_tokens 与参考档不一致；"
                "参考档不是同一提示集，不可比"
            )
        summaries.append(
            {
                "request": row["id"],
                "shape": list(ids_array.shape),
                "dtype": str(ids_array.dtype),
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "min_id": int(ids_array.min()),
                "max_id": int(ids_array.max()),
            }
        )
        matched.append(row["choices"][0]["text"] == ref["choices"][0]["text"])
    result = {
        "scope": "Request-level logical top-k only; no per-step/rank group_list "
        "reconciliation and no performance claim.",
        "requests": summaries,
        "matching_greedy_text_by_request": matched,
        "per_step_rank_verified": False,
    }
    collect.save(archive / "logical-topk.json", result)
    if not all(matched):
        raise ValueError("逐请求贪心文本与参考档不一致；本窗口不可比，原始结果已保留")
    return result


def write_manifest(archive, inputs, approval) -> None:
    collect.save(
        archive / "collection_manifest.json",
        {
            "schema": "vllm-hust/collection/collection-manifest/v1",
            "produced_at": utc_now(),
            "run_id": archive.name,
            "host_commits": inputs["hosts"],
            "collection_shape": inputs["shape"],
            "route_flag": ROUTE_FLAG,
            "diagnostic_only": True,
            "artifact_contract": "strict",
            "reviewed_plan": {
                "path": approval["path"],
                "sha256": approval["sha256"],
            },
            "reference_archive": str(inputs["reference"]),
            "diagnostic_input_archive": str(inputs["diagnostic"]),
            "artifacts": [
                "collection_manifest.json",
                "config.json",
                "route-status.json",
                "npu-before.txt",
                "device-mapping.txt",
                "host_preflight.json",
                "native/invocation.json",
                "native/exit.json",
                "requests/request-*.json",
                "responses.json",
                "logical-topk.json",
            ],
        },
    )


def git_status() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout


def authoritative_root() -> None:
    """占卡执行只绑定共用 serving-longrun.lock 的权威 checkout。

    自检（默认路径）不需要该约束；``--run`` 需要。只查私有 ``.git`` 等于
    common dir 仍会让任一独立 clone 拿到不同的锁文件，因此运行路径直接
    绑定当前共用锁的那棵 checkout。
    """
    if ROOT.resolve() != AUTHORITATIVE_ROOT.resolve():
        raise ValueError(
            f"占卡执行只绑定共用 serving-longrun.lock 的权威 checkout "
            f"{AUTHORITATIVE_ROOT}；当前检出 {ROOT} 是独立 clone/worktree，"
            "锁互不相干，拒绝执行"
        )


def clean_tree() -> None:
    if git_status().strip():
        raise ValueError("工作区不干净；评审与占卡执行必须基于同一已提交代码版本")


class ProbeInterrupted(RuntimeError):
    """Signal/alarm cancellation.

    Deliberately not a ``TimeoutError``: serving_collect's server startup
    retry loop catches ``TimeoutError`` and would swallow the cancellation.
    """


def _interrupt(signum, _frame):
    raise ProbeInterrupted(f"探针被信号 {signum} 打断或超出 {RUN_BUDGET_S}s 时限")


#: 服务退出回执的必备字段；缺一即不判成功。
EXIT_KEYS = (
    "returncode",
    "returncode_before_shutdown",
    "shutdown_requested",
    "forced_kill",
    "collection_completed",
)
EXIT_BOOLEANS = ("shutdown_requested", "forced_kill", "collection_completed")


def exit_receipt_ok(receipt) -> None:
    for key in EXIT_KEYS:
        if key not in receipt:
            raise ValueError(f"服务退出回执缺字段 {key}：{receipt}")
    for key in EXIT_BOOLEANS:
        if not isinstance(receipt[key], bool):
            raise ValueError(f"服务退出回执 {key} 不是布尔：{receipt[key]!r}")
    # bool 是 int 子类：退出码必须是真的 int，不接受布尔或浮点。
    rc = receipt["returncode"]
    if (
        isinstance(rc, bool)
        or not isinstance(rc, int)
        or rc not in (0, -signal.SIGTERM)
    ):
        raise ValueError(f"服务退出码 {rc!r} 不是整数 0 或 SIGTERM 关闭 -15：{receipt}")
    if (
        receipt["returncode_before_shutdown"] is not None
        or not receipt["shutdown_requested"]
    ):
        raise ValueError(f"服务在请求关闭前提前退出：{receipt}")
    if receipt["forced_kill"] or not receipt["collection_completed"]:
        raise ValueError(f"服务被硬杀或未完成采集，不判成功：{receipt}")


def request_sink(archive):
    """Per-request archive sink: finished requests survive a partial failure.

    Each response is written to a ``.json.tmp`` sibling and renamed into
    place, so an interrupt cannot leave a half-written file under the
    final ``request-NN.json`` name.
    """

    directory = archive / "requests"
    directory.mkdir(exist_ok=False)

    def sink(index, response):
        final = directory / f"request-{index:02d}.json"
        tmp = final.with_name(final.name + ".tmp")
        collect.save(tmp, response)
        tmp.replace(final)

    return sink


def saved_requests(archive) -> int:
    directory = archive / "requests"
    return len(list(directory.glob("request-*.json"))) if directory.is_dir() else 0


def window(args, inputs, archive, approval) -> None:
    """One reviewed window: device preflight, host identity, one request batch."""
    config = inputs["config"]
    shape = inputs["shape"]
    # profile=False 时不追加 guard_server_args；不并入真实命令就丢路由 flag。
    config["server_command"] = list(config["server_command"]) + list(
        config["guard_server_args"]
    )
    forbidden_ports = {int(config["port"]), inputs["invocation_port"]}
    write_manifest(archive, inputs, approval)
    stage(archive, "archive_created", run_id=archive.name)
    try:
        with LOCKFILE.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stage(archive, "device_preflight")
            resolve_devices(parse_devices(args.devices), archive)
            # gate=0 并入 config environment：verify_host 与服务看到同一最终关键
            # 环境；可见性不做掩码——本计划只覆盖 ASCEND_RT_VISIBLE_DEVICES 键
            # 未设置的直通两卡工况，已设置即在预检拒绝。
            config["environment"] = dict(config["environment"])
            config["environment"][collect.gate_env_name(config)] = "0"
            stage(archive, "host_identity")
            collect.verify_host(config, archive)
            port = pick_port(args.port, forbidden_ports)
            config["server_command"][config["server_command"].index("--port") + 1] = (
                str(port)
            )
            config["port"] = port
            collect.save(archive / "config.json", config)
            stage(archive, "starting_native_service")
            with collect.server(config, archive / "native", "off", profile=False):
                stage(archive, "warmup")
                for _ in range(int(shape["warmup_repeats"])):
                    collect.completion(
                        config, WARMUP_PROMPT, int(shape["warmup_tokens"])
                    )
                stage(archive, "request_window")
                prompts = collect.build_profile_prompts(
                    int(shape["profile_requests"]), ""
                )
                sink = request_sink(archive)
                rows = collect.run_profile_requests(
                    config,
                    prompts,
                    int(shape["profile_tokens"]),
                    int(shape["concurrency"]),
                    sink=sink,
                )
                collect.save(archive / "responses.json", rows)
        stage(archive, "decode_and_validate")
        result = validate(rows, inputs["responses"], shape, inputs["expected"], archive)
        receipt = load(archive / "native" / "exit.json")
        exit_receipt_ok(receipt)
        result["service_exit"] = receipt
        collect.save(archive / "logical-topk.json", result)
        stage(
            archive,
            "complete_request_level_only",
            valid_requests=len(rows),
            per_step_rank_verified=False,
        )
        print(
            f"探针完成：{len(rows)} 条请求全部校验通过，仅请求级结论，"
            f"未做逐卡逐调度步对账，无服务收益结论。归档 {archive}",
            flush=True,
        )
    except BaseException as exc:
        previous = load(archive / "route-status.json")
        fields = {
            "last_stage": previous.get("stage"),
            "saved_requests": saved_requests(archive),
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ProbeInterrupted):
            fields["drain_note"] = (
                "已取消未启动请求；在途请求按其自身请求超时返回，"
                f"{RUN_BUDGET_S}s 时限不是硬上界"
            )
        stage(archive, "failed", **fields)
        raise


def execute(args, next_record) -> int:
    approval = approved_plan(next_record)
    frozen_matches(approval["file"])
    authoritative_root()
    clean_tree()
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGALRM, _interrupt)
    signal.alarm(RUN_BUDGET_S)
    try:
        inputs = gather(next_record)
        archive = allocate_archive(args.archive_dir)
        window(args, inputs, archive, approval)
    finally:
        signal.alarm(0)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="原生路由请求探针：默认只读自检；--run 须先登记 reviewed_plan。"
    )
    parser.add_argument(
        "--run", action="store_true", help="占卡执行已评审计划；未评审即拒绝"
    )
    parser.add_argument(
        "--devices",
        help="显式两张物理卡号（严格 N,M 纯数字），仅核验、须与当前 npu-smi 映射一致",
    )
    parser.add_argument("--port", type=int, help="显式服务端口；不得复用历史端口")
    parser.add_argument(
        "--archive-dir", help="归档根下不存在的受控子目录；缺省自动分配"
    )
    args = parser.parse_args(argv)
    try:
        next_record = load(HANDOFF)
        if args.run:
            return execute(args, next_record)
        return report(gather(next_record), next_record)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"native_route_probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
