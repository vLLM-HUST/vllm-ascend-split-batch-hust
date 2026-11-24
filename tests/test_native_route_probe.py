"""The native route probe refuses unreviewed runs and verifies registered inputs."""

import ast
import base64
import hashlib
import io
import json
import signal
import socket
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.operator_tools import artifact_contract  # noqa: E402
from tools.operator_tools import native_route_probe as probe  # noqa: E402

HOSTS = {
    "vllm": "f18cf803c5f63625e2c71253ddaf8b0bad0bad1a",
    "vllm-ascend": "320eced8c1bba71c0329bf59caa721a2382e7499",
}
LAYERS, TOP_K, EXPERTS = 40, 8, 256
NPU_INFO = (
    "| NPU   Name   | Health |\n"
    "| 4     910B2  | OK     |\n"
    "| No running processes found in NPU 4                 |\n"
    "| No running processes found in NPU 6                 |\n"
)
NPU_MAPPING = (
    "NPU ID   Chip ID   Chip Logic ID   Chip Name\n"
    "4        0         0                Ascend 910B2\n"
    "4        1         -                Mcu\n"
    "6        0         1                Ascend 910B2\n"
    "6        1         -                Mcu\n"
)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_row(index):
    return {
        "id": f"cmpl-ref-{index}",
        "choices": [
            {"text": f"\n\nVariant {index} of a binary heap.", "routed_experts": None}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 16},
    }


def make_world(tmp_path, config_tweak=None):
    diagnostic = tmp_path / "archives" / "diagnostic"
    reference = tmp_path / "archives" / "reference"
    python = tmp_path / "venv" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    model = tmp_path / "model"
    model.mkdir()
    server_command = [
        str(python),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8350",
        "--served-model-name",
        "qwen35-35b",
        "--tensor-parallel-size",
        "2",
    ]
    guard = ["--no-enable-prefix-caching", probe.ROUTE_FLAG]
    shape = {
        "prefix_caching": 0,
        "prefix_caching_flag": "--no-enable-prefix-caching",
        "warmup_tokens": 16,
        "l3_tokens": 128,
        "profile_tokens": 16,
        "warmup_repeats": 10,
        "profile_requests": 16,
        "concurrency": 16,
    }
    config = {
        "python": str(python),
        "server_command": server_command,
        "guard_server_args": guard,
        "host_commits": HOSTS,
        "gate": "VLLM_HUST_GDN_SPLIT_BA",
        "port": 18371,
        "model_alias": "qwen35-35b",
        "model_path": str(model),
        "collection_shape": shape,
        "startup_timeout": 900,
        "environment": {"PYTHONPATH": "/usr/local/Ascend/cann/lib"},
        "sync_mode": "none",
    }
    historical_command = list(server_command) + ["--no-enable-prefix-caching"]
    if config_tweak is not None:
        config_tweak(config)
    diagnostic_sha = {
        "config.json": write_json(diagnostic / "config.json", config),
        "diagnostic-plan.json": write_json(
            diagnostic / "diagnostic-plan.json",
            {
                "expected": {
                    "host_commits": HOSTS,
                    "shape": shape,
                    "model_layers": LAYERS,
                    "expert_count": EXPERTS,
                    "top_k": TOP_K,
                },
                "single_server_flag": probe.ROUTE_FLAG,
            },
        ),
        "host_preflight.json": write_json(
            diagnostic / "host_preflight.json",
            {
                "returncode": 0,
                "stdout": json.dumps(
                    {name: {"revision": commit} for name, commit in HOSTS.items()}
                ),
            },
        ),
        "route-status.json": write_json(
            diagnostic / "route-status.json", {"stage": "starting_native_service"}
        ),
    }
    responses = [reference_row(index) for index in range(shape["profile_requests"])]
    reference_sha = {
        "collection_manifest.json": write_json(
            reference / "collection_manifest.json",
            {"host_commits": HOSTS, "collection_shape": shape},
        ),
        "round_00_off/invocation.json": write_json(
            reference / "round_00_off" / "invocation.json",
            {"command": historical_command + ["--profiler-config", "{}"]},
        ),
        "round_00_off/profile_responses.json": write_json(
            reference / "round_00_off" / "profile_responses.json", responses
        ),
    }
    screen_path = tmp_path / "screen.json"
    write_json(
        screen_path,
        {
            "host_commits": HOSTS,
            "archive_uri": reference.as_uri(),
            "next_diagnostic": {
                "progress": {
                    "archive_uri": diagnostic.as_uri(),
                    "sha256": diagnostic_sha,
                }
            },
            "observations": {"sha256": reference_sha},
        },
    )
    handoff_path = tmp_path / "optimization-next.json"
    write_json(handoff_path, {"host_commits": HOSTS, "reviewed_plan": None})
    return {
        "screen": screen_path,
        "handoff": handoff_path,
        "archives_root": tmp_path / "archives",
        "diagnostic": diagnostic,
        "reference": reference,
        "config": config,
        "responses": responses,
    }


@pytest.fixture
def world(tmp_path, monkeypatch):
    built = make_world(tmp_path)
    monkeypatch.setattr(probe, "SCREEN", built["screen"])
    monkeypatch.setattr(probe, "HANDOFF", built["handoff"])
    monkeypatch.setattr(probe, "ARCHIVES", built["archives_root"])
    monkeypatch.setattr(probe, "LOCKFILE", tmp_path / "serving-longrun.lock")
    return built


def register_plan(world, entry=None):
    record = json.loads(world["handoff"].read_text())
    if entry is None:
        entry = {
            "path": probe.PLAN,
            "sha256": hashlib.sha256((ROOT / probe.PLAN).read_bytes()).hexdigest(),
        }
    record["reviewed_plan"] = entry
    world["handoff"].write_text(json.dumps(record, indent=1) + "\n")


def fake_npu(argv):
    if argv[1:] == ["info"]:
        return NPU_INFO
    if argv[1:] == ["info", "-m"]:
        return NPU_MAPPING
    raise AssertionError(f"unexpected npu-smi call {argv}")


def native_payload(np, prompt_tokens, completion_tokens, seed):
    steps = prompt_tokens + completion_tokens - 1
    ids = np.empty((steps, LAYERS, TOP_K), dtype=np.uint8)
    for step in range(steps):
        for layer in range(LAYERS):
            ids[step, layer] = [
                (step * 7 + layer * 3 + slot + seed) % EXPERTS for slot in range(TOP_K)
            ]
    buffer = io.BytesIO()
    np.save(buffer, ids, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode()


def native_row(np, index, prompt_tokens=10, completion_tokens=16, seed=None):
    return {
        "id": f"cmpl-native-{index}",
        "choices": [
            {
                "text": f"\n\nVariant {index} of a binary heap.",
                "routed_experts": native_payload(
                    np,
                    prompt_tokens,
                    completion_tokens,
                    index if seed is None else seed,
                ),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def side_effect_free(world):
    return {item.name for item in world["archives_root"].iterdir()} == {
        "diagnostic",
        "reference",
    }


def test_plan_freezes_the_current_tool_source():
    frozen = probe.FROZEN_TOOL.search((ROOT / probe.PLAN).read_text())
    assert frozen, "计划缺源码冻结摘要"
    tool = ROOT / "tools/operator_tools/native_route_probe.py"
    assert frozen.group(1) == hashlib.sha256(tool.read_bytes()).hexdigest()


def test_module_never_imports_tensor_stacks_at_load_time():
    tree = ast.parse((ROOT / "tools/operator_tools/native_route_probe.py").read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"numpy", "torch"}


def test_default_self_check_is_ready_and_read_only(world, capsys):
    assert probe.main([]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 4, out
    assert "未占卡、未起服、未采集" in out and "没有任何真实实测" in out
    assert "本计划引用的输入指纹全部匹配" in out
    assert "工况三源一致（诊断配置、参考 manifest、诊断计划）" in out
    assert "未核验：当前运行时宿主与设备" in out
    assert probe.PLAN in out and "sha256=" in out
    # 未评审时唯一动作是评审计划本身；不把手工粘长哈希当主要动作。
    assert "唯一动作：评审这份计划" in out
    assert "手抄长哈希" in out and '{"path"' not in out
    assert "无审批不执行占卡命令" in out
    # 无审批时不得打印可起服命令，也不得声称已采集。
    assert "native_route_probe --run" not in out
    assert side_effect_free(world)


def test_default_reports_a_registered_plan(world, capsys):
    register_plan(world)
    assert probe.main([]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 4, out
    assert "唯一动作" in out and "native_route_probe --run" in out
    assert side_effect_free(world)


def test_default_refuses_tampered_registered_input(world, capsys):
    path = world["reference"] / "round_00_off" / "profile_responses.json"
    path.write_text(path.read_text() + " ")
    assert probe.main([]) == 1
    assert "SHA256" in capsys.readouterr().err
    assert side_effect_free(world)


@pytest.mark.parametrize(
    "tweak, message",
    [
        (lambda c: c["guard_server_args"].remove(probe.ROUTE_FLAG), "路由 flag"),
        (lambda c: c.update(environment={"PYTHONPATH": ""}), "空组件"),
        (
            lambda c: c.update(
                environment={"PYTHONPATH": f"{ROOT}:/usr/local/Ascend/cann/lib"}
            ),
            "仓库源码",
        ),
        (
            lambda c: c["environment"].update({"ASCEND_RT_VISIBLE_DEVICES": "4,6"}),
            "历史设备绑定",
        ),
        (lambda c: c.update(host_commits={"vllm": "0" * 40}), "双宿主"),
        (lambda c: c["guard_server_args"].append("--extra-shape-arg"), "工况"),
        (lambda c: c["server_command"].remove("--tensor-parallel-size"), "工况"),
        (lambda c: c["guard_server_args"].append(probe.ROUTE_FLAG), "工况"),
    ],
)
def test_default_refuses_broken_registered_config(
    tmp_path, monkeypatch, tweak, message
):
    built = make_world(tmp_path, config_tweak=tweak)
    monkeypatch.setattr(probe, "SCREEN", built["screen"])
    monkeypatch.setattr(probe, "HANDOFF", built["handoff"])
    assert probe.main([]) == 1


def test_default_refuses_reference_rows_with_routed_payload(
    tmp_path, monkeypatch, capsys
):
    built = make_world(tmp_path)
    rows = built["responses"]
    rows[0]["choices"][0]["routed_experts"] = "should-not-be-here"
    path = built["reference"] / "round_00_off" / "profile_responses.json"
    registered = json.loads(built["screen"].read_text())
    registered["observations"]["sha256"]["round_00_off/profile_responses.json"] = (
        write_json(path, rows)
    )
    write_json(built["screen"], registered)
    monkeypatch.setattr(probe, "SCREEN", built["screen"])
    monkeypatch.setattr(probe, "HANDOFF", built["handoff"])
    assert probe.main([]) == 1
    assert "routed_experts" in capsys.readouterr().err


def test_run_refused_without_review(world, capsys):
    assert probe.main(["--run"]) == 1
    assert "reviewed_plan" in capsys.readouterr().err
    assert side_effect_free(world)


def test_run_refused_when_review_points_elsewhere(world, capsys):
    register_plan(
        world, {"path": "experiments/other/EXPERIMENT_PLAN.md", "sha256": "0" * 64}
    )
    assert probe.main(["--run"]) == 1
    assert probe.PLAN in capsys.readouterr().err
    assert side_effect_free(world)


def test_run_refused_when_plan_hash_mismatches(world, capsys):
    register_plan(world, {"path": probe.PLAN, "sha256": "0" * 64})
    assert probe.main(["--run"]) == 1
    assert "SHA256" in capsys.readouterr().err
    assert side_effect_free(world)


def test_run_refused_when_tool_source_drifts_from_plan(world, monkeypatch, capsys):
    register_plan(world)
    monkeypatch.setattr(probe, "file_sha", lambda path: "0" * 64)
    assert probe.main(["--run"]) == 1
    assert "冻结摘要" in capsys.readouterr().err
    assert side_effect_free(world)


def test_run_refused_when_tree_is_dirty(world, monkeypatch, capsys):
    register_plan(world)
    monkeypatch.setattr(probe, "AUTHORITATIVE_ROOT", probe.ROOT)
    monkeypatch.setattr(probe, "git_status", lambda: " M tools/operator_tools/x.py\n")
    assert probe.main(["--run"]) == 1
    assert "不干净" in capsys.readouterr().err
    assert side_effect_free(world)


def test_run_refused_outside_authoritative_root(world, monkeypatch, capsys):
    # 独立 clone/worktree 的同名锁文件互不相干：--run 绑定共用锁的权威 checkout。
    register_plan(world)
    monkeypatch.setattr(
        probe, "AUTHORITATIVE_ROOT", probe.ROOT.parent / "another-checkout"
    )
    assert probe.main(["--run"]) == 1
    assert "权威 checkout" in capsys.readouterr().err
    assert side_effect_free(world)


def test_default_self_check_runs_outside_the_authoritative_root(
    world, monkeypatch, capsys
):
    # 自检不需要权威 checkout 约束；只读核验在任何检出上都可跑。
    monkeypatch.setattr(
        probe, "AUTHORITATIVE_ROOT", probe.ROOT.parent / "another-checkout"
    )
    assert probe.main([]) == 0
    assert "只读自检通过" in capsys.readouterr().out
    assert side_effect_free(world)


def test_no_flag_can_self_authorize_a_run(world):
    for flag in ("--yes", "--force", "--approved"):
        with pytest.raises(SystemExit):
            probe.main(["--run", flag])


def test_run_executes_the_reviewed_window_end_to_end(world, monkeypatch, capsys):
    np = pytest.importorskip("numpy")
    register_plan(world)
    monkeypatch.setattr(probe, "git_status", lambda: "")
    monkeypatch.setattr(probe, "AUTHORITATIVE_ROOT", probe.ROOT)
    # 占卡工况只覆盖该键未设置；测试环境同样以 unset 为准。
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(probe, "npu_text", fake_npu)
    monkeypatch.setattr(
        probe.collect,
        "verify_host",
        lambda config, archive: (archive / "host_preflight.json").write_text("{}"),
    )

    @contextmanager
    def fake_server(config, output, arm, profile=False):
        assert arm == "off" and profile is False
        assert probe.ROUTE_FLAG in config["server_command"]
        assert "--profiler-config" not in config["server_command"]
        assert config["environment"]["VLLM_HUST_GDN_SPLIT_BA"] == "0"
        write_json(
            output / "invocation.json",
            {"command": config["server_command"], "gate": "0"},
        )
        write_json(output / "owner.json", {"pid": 1})
        write_json(
            output / "exit.json",
            {
                "returncode": 0,
                "returncode_before_shutdown": None,
                "shutdown_requested": True,
                "forced_kill": False,
                "collection_completed": True,
            },
        )
        yield {}

    warmups = []
    monkeypatch.setattr(probe.collect, "server", fake_server)
    monkeypatch.setattr(
        probe.collect,
        "completion",
        lambda config, prompt, tokens: warmups.append((prompt, tokens)) or {},
    )

    def fake_requests(config, prompts, tokens, concurrency, sink=None):
        rows = [native_row(np, index) for index in range(len(prompts))]
        if sink is not None:
            for index, row in enumerate(rows):
                sink(index, row)
        return rows

    monkeypatch.setattr(probe.collect, "run_profile_requests", fake_requests)
    assert probe.main(["--run"]) == 0
    out = capsys.readouterr().out
    assert "仅请求级" in out
    created = [
        item
        for item in world["archives_root"].iterdir()
        if item.name.startswith("native-route-probe-")
    ]
    assert len(created) == 1
    archive = created[0]
    status = json.loads((archive / "route-status.json").read_text())
    assert status["stage"] == "complete_request_level_only"
    assert status["valid_requests"] == 16
    assert len(warmups) == 10 and all(
        prompt == "Explain RMSNorm briefly." and tokens == 16
        for prompt, tokens in warmups
    )
    run_config = json.loads((archive / "config.json").read_text())
    assert run_config["port"] not in (18371, 8350)
    assert run_config["server_command"][
        run_config["server_command"].index("--port") + 1
    ] == str(run_config["port"])
    # 可见性不做运行时掩码；verify_host 与服务同见 gate=0。
    assert "ASCEND_RT_VISIBLE_DEVICES" not in run_config["environment"]
    assert run_config["environment"]["VLLM_HUST_GDN_SPLIT_BA"] == "0"
    saved = sorted((archive / "requests").glob("request-*.json"))
    assert [p.name for p in saved] == [f"request-{i:02d}.json" for i in range(16)]
    result = json.loads((archive / "logical-topk.json").read_text())
    assert all(result["matching_greedy_text_by_request"])
    assert result["per_step_rank_verified"] is False
    assert result["service_exit"]["forced_kill"] is False
    assert artifact_contract.inspect(archive, strict=True).ok


def test_run_keeps_raw_results_when_validation_fails(world, monkeypatch, capsys):
    np = pytest.importorskip("numpy")
    register_plan(world)
    monkeypatch.setattr(probe, "git_status", lambda: "")
    monkeypatch.setattr(probe, "AUTHORITATIVE_ROOT", probe.ROOT)
    # 占卡工况只覆盖该键未设置；测试环境同样以 unset 为准。
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(probe, "npu_text", fake_npu)
    monkeypatch.setattr(
        probe.collect,
        "verify_host",
        lambda config, archive: (archive / "host_preflight.json").write_text("{}"),
    )

    @contextmanager
    def fake_server(config, output, arm, profile=False):
        write_json(
            output / "exit.json",
            {"forced_kill": False, "collection_completed": True, "returncode": 0},
        )
        yield {}

    mismatched = [native_row(np, index) for index in range(16)]
    mismatched[3]["choices"][0]["text"] = "different greedy text"
    monkeypatch.setattr(probe.collect, "server", fake_server)
    monkeypatch.setattr(probe.collect, "completion", lambda *a: {})
    monkeypatch.setattr(
        probe.collect,
        "run_profile_requests",
        lambda config, prompts, tokens, c, sink=None: mismatched,
    )
    assert probe.main(["--run"]) == 1
    assert "贪心文本" in capsys.readouterr().err
    archive = next(
        item
        for item in world["archives_root"].iterdir()
        if item.name.startswith("native-route-probe-")
    )
    status = json.loads((archive / "route-status.json").read_text())
    assert status["stage"] == "failed" and status["last_stage"] == "decode_and_validate"
    assert status["saved_requests"] == 0
    assert json.loads((archive / "responses.json").read_text()) == mismatched
    kept = json.loads((archive / "logical-topk.json").read_text())
    assert kept["matching_greedy_text_by_request"][3] is False


@pytest.mark.parametrize(
    "row, message",
    [
        (
            {
                "id": "empty",
                "choices": [{"text": "x", "routed_experts": ""}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 16},
            },
            "为空",
        ),
        (
            {
                "id": "missing",
                "choices": [{"text": "x"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 16},
            },
            "为空",
        ),
    ],
)
def test_validate_rejects_empty_payload(tmp_path, row, message):
    shape = {
        **make_world(tmp_path)["config"]["collection_shape"],
        "profile_requests": 1,
    }
    with pytest.raises(ValueError, match=message):
        probe.validate([row], [reference_row(0)], shape, expected_counts(), tmp_path)


def expected_counts():
    return {"model_layers": LAYERS, "expert_count": EXPERTS, "top_k": TOP_K}


@pytest.mark.parametrize(
    "kind, message",
    [
        ("shape", "形状"),
        ("dtype", "整数"),
        ("duplicate", "重复"),
        ("range", "超出"),
        ("tokens", "completion_tokens"),
    ],
)
def test_validate_rejects_malformed_payloads(tmp_path, kind, message):
    np = pytest.importorskip("numpy")
    row = (
        native_row(np, 0, completion_tokens=15)
        if kind == "tokens"
        else native_row(np, 0)
    )
    if kind != "tokens":
        payload = np.load(
            io.BytesIO(base64.b64decode(row["choices"][0]["routed_experts"])),
            allow_pickle=False,
        )
        if kind == "shape":
            payload = payload[:, :-1, :]
        elif kind == "dtype":
            payload = payload.astype(np.float32)
        elif kind == "duplicate":
            payload[0, 0, 1] = payload[0, 0, 0]
        elif kind == "range":
            payload = payload.astype(np.uint16)
            payload[0, 0, 0] = EXPERTS
        buffer = io.BytesIO()
        np.save(buffer, payload, allow_pickle=False)
        row["choices"][0]["routed_experts"] = base64.b64encode(
            buffer.getvalue()
        ).decode()
    shape = {
        **make_world(tmp_path)["config"]["collection_shape"],
        "profile_requests": 1,
    }
    with pytest.raises(ValueError, match=message):
        probe.validate([row], [reference_row(0)], shape, expected_counts(), tmp_path)


def test_devices_follow_the_current_mapping_only(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "npu_text", fake_npu)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    (tmp_path / "npu-before.txt").parent.mkdir(parents=True, exist_ok=True)
    assert probe.resolve_devices([4, 6], tmp_path) == [4, 6]
    assert (tmp_path / "npu-before.txt").is_file()
    assert (tmp_path / "device-mapping.txt").is_file()


@pytest.mark.parametrize(
    "ambient",
    [
        "",
        ",",
        "4,,6",
        "1",
        "4",
        "6,4",
        "4,6",
        " 4, 6 ",
        "4,6,8",
        "0,1",
    ],
)
def test_devices_refuse_any_set_visibility_key(tmp_path, monkeypatch, ambient):
    # 本计划只覆盖该键未设置的直通两卡工况：已设置一律拒绝，
    # 不因字面可转换（含空串、纯逗号、物理/逻辑号一致）而放行。
    monkeypatch.setattr(probe, "npu_text", fake_npu)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", ambient)
    with pytest.raises(ValueError, match="已设置"):
        probe.resolve_devices(None, tmp_path)


def test_devices_pass_only_when_visibility_key_is_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "npu_text", fake_npu)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    assert probe.resolve_devices(None, tmp_path) == [4, 6]


def test_parse_devices_default_keeps_no_explicit_numbers():
    assert probe.parse_devices(None) is None, "缺省 --devices 保持按映射自动核对"
    assert probe.parse_devices("4,6") == [4, 6]


@pytest.mark.parametrize(
    "text",
    ["4,,6", "4,6,", ",", " 4, 6 ", "4, 6", "4,", "4,4", "4,04", "4", "4,6,8", "4,x"],
)
def test_parse_devices_refuses_repairable_or_duplicate_literals(text):
    # 显式核验输入不做字面修复：空字段/尾逗号/空格/数值重复/非数字一律拒。
    with pytest.raises(ValueError, match="--devices"):
        probe.parse_devices(text)


def test_devices_refuse_duplicate_physical_mapping(tmp_path, monkeypatch):
    same_card = (
        "NPU ID   Chip ID   Chip Logic ID   Chip Name\n"
        "4        0         0                Ascend 910B2\n"
        "4        0         1                Ascend 910B2\n"
    )
    monkeypatch.setattr(
        probe,
        "npu_text",
        lambda argv: NPU_INFO if argv[1:] == ["info"] else same_card,
    )
    with pytest.raises(ValueError, match="同一张物理卡"):
        probe.resolve_devices(None, tmp_path)


def test_devices_refuse_busy_or_extra_cards(tmp_path, monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        probe,
        "npu_text",
        lambda argv: (
            NPU_INFO.replace("No running processes found in NPU 4", "| 1234 vllm")
            if argv[1:] == ["info"]
            else NPU_MAPPING
        ),
    )
    with pytest.raises(ValueError, match="忙"):
        probe.resolve_devices(None, tmp_path)
    monkeypatch.setattr(
        probe,
        "npu_text",
        lambda argv: (
            NPU_INFO
            if argv[1:] == ["info"]
            else NPU_MAPPING + "8  0  2  Ascend 910B2\n"
        ),
    )
    with pytest.raises(ValueError, match="恰好两张"):
        probe.resolve_devices(None, tmp_path)
    monkeypatch.setattr(probe, "npu_text", fake_npu)
    with pytest.raises(ValueError, match="不符"):
        probe.resolve_devices([4, 8], tmp_path)


def test_ports_never_reuse_archived_values():
    with pytest.raises(ValueError, match="历史端口"):
        probe.pick_port(18371, {18371, 8350})
    chosen = probe.pick_port(None, {18371, 8350})
    assert chosen not in (18371, 8350) and 1024 < chosen < 65536


def test_archive_targets_are_unique_and_controlled(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ARCHIVES", tmp_path / "archives")
    (tmp_path / "archives").mkdir()
    first = probe.allocate_archive(None)
    second = probe.allocate_archive(None)
    assert first != second and first.is_dir() and second.is_dir()
    with pytest.raises(FileExistsError):
        probe.allocate_archive(str(first))
    with pytest.raises(ValueError, match="受控子目录"):
        probe.allocate_archive(str(tmp_path / "outside" / "probe"))


class FakeProc:
    pid = 4242
    returncode = 0

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0


def install_fake_proc(monkeypatch, killed):
    monkeypatch.setattr(probe.collect.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(probe.collect.os, "killpg", lambda pid, sig: killed.append(sig))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_config(port):
    return {
        "gate": "VLLM_HUST_GDN_SPLIT_BA",
        "port": port,
        "server_command": ["true"],
        "environment": {},
    }


def test_health_interrupt_is_not_swallowed_by_startup_retry(tmp_path, monkeypatch):
    killed = []
    install_fake_proc(monkeypatch, killed)

    def boom(*args, **kwargs):
        raise probe.ProbeInterrupted("alarm during health wait")

    monkeypatch.setattr(probe.collect, "request", boom)
    with (
        pytest.raises(probe.ProbeInterrupted),
        probe.collect.server(server_config(free_port()), tmp_path / "native", "off"),
    ):
        pass
    assert signal.SIGTERM in killed, "打断后必须清理所属服务进程组"
    receipt = json.loads((tmp_path / "native" / "exit.json").read_text())
    assert receipt["shutdown_requested"] is True
    assert receipt["collection_completed"] is False
    assert receipt["returncode_before_shutdown"] is None


def test_owner_json_failure_still_terminates_server(tmp_path, monkeypatch):
    killed = []
    install_fake_proc(monkeypatch, killed)
    real_save = probe.collect.save

    def failing_save(path, value):
        if Path(path).name == "owner.json":
            raise OSError("disk full while writing owner.json")
        return real_save(path, value)

    monkeypatch.setattr(probe.collect, "save", failing_save)
    with (
        pytest.raises(OSError, match="owner.json"),
        probe.collect.server(server_config(free_port()), tmp_path / "native", "off"),
    ):
        pass
    assert signal.SIGTERM in killed, "owner 写入失败也必须清理已起的服务进程组"
    assert (tmp_path / "native" / "exit.json").is_file()


GOOD_EXIT = {
    "returncode": 0,
    "returncode_before_shutdown": None,
    "shutdown_requested": True,
    "forced_kill": False,
    "collection_completed": True,
}


@pytest.mark.parametrize("returncode", [0, -signal.SIGTERM])
def test_exit_receipt_accepts_clean_or_sigterm_shutdown(returncode):
    probe.exit_receipt_ok({**GOOD_EXIT, "returncode": returncode})


@pytest.mark.parametrize(
    "receipt, message",
    [
        ({k: v for k, v in GOOD_EXIT.items() if k != "forced_kill"}, "缺字段"),
        ({**GOOD_EXIT, "returncode": 37}, "不是整数 0"),
        ({**GOOD_EXIT, "returncode": None}, "不是整数 0"),
        ({**GOOD_EXIT, "returncode": True}, "不是整数 0"),
        ({**GOOD_EXIT, "returncode": 37.0}, "不是整数 0"),
        ({**GOOD_EXIT, "forced_kill": "false"}, "不是布尔"),
        ({**GOOD_EXIT, "collection_completed": 1}, "不是布尔"),
        ({**GOOD_EXIT, "returncode_before_shutdown": 0}, "提前退出"),
        ({**GOOD_EXIT, "shutdown_requested": False}, "提前退出"),
        ({**GOOD_EXIT, "forced_kill": True}, "硬杀"),
        ({**GOOD_EXIT, "collection_completed": False}, "硬杀"),
    ],
)
def test_exit_receipt_refuses_unknown_early_or_nonboolean(receipt, message):
    with pytest.raises(ValueError, match=message):
        probe.exit_receipt_ok(receipt)


def test_validate_cross_checks_reference_prompt_tokens(tmp_path):
    np = pytest.importorskip("numpy")
    reference = reference_row(0)
    reference["usage"]["prompt_tokens"] = 11
    shape = {
        **make_world(tmp_path)["config"]["collection_shape"],
        "profile_requests": 1,
    }
    with pytest.raises(ValueError, match="prompt_tokens 与参考档不一致"):
        probe.validate(
            [native_row(np, 0)], [reference], shape, expected_counts(), tmp_path
        )


def test_validate_refuses_empty_or_duplicated_request_ids(tmp_path):
    np = pytest.importorskip("numpy")
    rows = [native_row(np, 0), native_row(np, 1)]
    rows[1]["id"] = ""
    shape = {
        **make_world(tmp_path)["config"]["collection_shape"],
        "profile_requests": 2,
    }
    with pytest.raises(ValueError, match="为空或重复"):
        probe.validate(
            rows,
            [reference_row(0), reference_row(1)],
            shape,
            expected_counts(),
            tmp_path,
        )


def test_validate_refuses_multi_choice_rows(tmp_path):
    np = pytest.importorskip("numpy")
    row = native_row(np, 0)
    row["choices"].append({"text": "second choice"})
    shape = {
        **make_world(tmp_path)["config"]["collection_shape"],
        "profile_requests": 1,
    }
    with pytest.raises(ValueError, match="单 choice"):
        probe.validate([row], [reference_row(0)], shape, expected_counts(), tmp_path)


def test_validate_reports_npy_decode_failure_clearly(tmp_path):
    np = pytest.importorskip("numpy")
    row = native_row(np, 0)
    row["choices"][0]["routed_experts"] = base64.b64encode(b"not an npy file").decode()
    shape = {
        **make_world(tmp_path)["config"]["collection_shape"],
        "profile_requests": 1,
    }
    with pytest.raises(ValueError, match="NPY 解码失败"):
        probe.validate([row], [reference_row(0)], shape, expected_counts(), tmp_path)


@pytest.mark.parametrize(
    "tweak, message",
    [
        (lambda rows: rows[5].update(id=""), "为空或重复"),
        (lambda rows: rows[5]["choices"].append({"text": "second"}), "单 choice"),
    ],
)
def test_reference_rows_must_have_single_choice_and_nonempty_id(tweak, message):
    shape = {"profile_requests": 16, "profile_tokens": 16}
    rows = [reference_row(index) for index in range(16)]
    tweak(rows)
    with pytest.raises(ValueError, match=message):
        probe.reference_rows_ok(rows, shape)


def test_run_profile_requests_sink_keeps_finished_results(monkeypatch):
    saved = {}

    def sink(index, response):
        saved[index] = response

    def failing_completion(config, prompt, tokens):
        if prompt == "fail":
            raise ValueError("request 3 failed")
        return {"prompt": prompt}

    monkeypatch.setattr(probe.collect, "completion", failing_completion)
    with pytest.raises(ValueError, match="request 3 failed"):
        probe.collect.run_profile_requests({}, ["a", "b", "fail", "d"], 4, 1, sink=sink)
    assert sorted(saved) == [0, 1], "失败前已成功的请求必须已留档"
    rows = probe.collect.run_profile_requests({}, ["a", "b"], 4, 1)
    assert rows == [{"prompt": "a"}, {"prompt": "b"}], "缺省 sink 保持旧行为"


def test_concurrent_failure_keeps_finished_results_without_draining(monkeypatch):
    saved = {}

    def sink(index, response):
        saved[index] = response

    def failing_completion(config, prompt, tokens):
        if prompt == "fail":
            raise ValueError("boom")
        return {"prompt": prompt}

    monkeypatch.setattr(probe.collect, "completion", failing_completion)
    prompts = ["ok"] * 15 + ["fail"]
    with pytest.raises(ValueError, match="boom"):
        probe.collect.run_profile_requests({}, prompts, 4, 16, sink=sink)
    assert sorted(saved) == list(range(15)), "并发下先完成的请求不得因 1 个失败丢失"


@pytest.mark.parametrize("with_sink", [False, True])
def test_shared_callers_keep_the_original_drain_behavior(monkeypatch, with_sink):
    """无 sink（既有采集器）等池退出再交还；带 sink（本探针）快速返回。"""
    gate = threading.Event()
    started = threading.Event()
    failed = threading.Event()
    drained = threading.Event()

    def completion(config, prompt, tokens):
        if prompt == "block":
            started.set()
            assert gate.wait(timeout=10)
            drained.set()
            return {"prompt": prompt}
        # 只有等在途请求真正开跑后才失败，取消才碰不到它。
        assert started.wait(timeout=10)
        failed.set()
        raise ValueError("boom")

    monkeypatch.setattr(probe.collect, "completion", completion)
    saved = {}

    def sink(index, response):
        saved[index] = response

    errors = []

    def call():
        try:
            probe.collect.run_profile_requests(
                {}, ["fail", "block"], 4, 2, sink=sink if with_sink else None
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=call)
    worker.start()
    assert failed.wait(timeout=10)
    if with_sink:
        worker.join(timeout=10)
        assert not worker.is_alive(), "留档路径不得等在途请求跑完才返回"
        assert isinstance(errors[0], ValueError)
        assert not drained.is_set()
    else:
        worker.join(timeout=0.2)
        assert worker.is_alive() and errors == [], "既有调用方必须等池退出再交还异常"
    gate.set()
    assert drained.wait(timeout=10), "在途请求不被取消，须能按自身节奏完成"
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert isinstance(errors[0], ValueError)


def test_request_sink_writes_via_temp_rename(tmp_path):
    (tmp_path / "archive").mkdir()  # 真实流程中归档目录由 allocate_archive 先建好
    sink = probe.request_sink(tmp_path / "archive")
    sink(3, {"id": "cmpl-3"})
    directory = tmp_path / "archive" / "requests"
    assert json.loads((directory / "request-03.json").read_text()) == {"id": "cmpl-3"}
    assert not list(directory.glob("*.tmp")), "临时文件必须已 rename，不留半个命名件"
