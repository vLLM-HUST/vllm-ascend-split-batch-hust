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

"""配置面边界（CPU-only，离线，不占卡）。

背景（2026-09-26 核实的两侧事实）：

- **上游**：`vllm-hust-ext configure --file` 只校验"文件顶层是 object"
  （`extension-manager/src/vllm_hust_ext/cli.py:185-187` 的 `isinstance(..., dict)`），
  随后把 dict 原样存进 `ExtensionConfig.configuration`；**没有** per-bundle schema，
  未知键不会被拒绝、也不会被解释。
- **本仓**：没有任何载体读 `configuration`（`src/**` 无 `.configuration` 访问，
  也无 `VLLM_HUST_EXT_CONFIG`），全部门控走 **env**（35 处 `os.getenv`/`os.environ`）。

所以"配置 schema / 未知配置拒绝"这条缺口在本仓的实际形态是**边界声明**，不是可加的校验：
schema 属于上游框架、而本仓不消费该字段。这里把边界变成可执行判据，防止将来有人悄悄
引入一条未声明的配置契约（那才会真的出现"未知键改变行为"）：

1. 本仓载体的门控面只有 env（源码级扫描）；
2. 未知 `configuration` 键**不改变** `extension check` 的任何判定（对照实验）；
3. 管理器生命周期（configure → enable → disable → forget）只写元数据，
   **从不 import 本仓实现模块**（`sys.modules` 审计），未知键在往返中原样保留。

服务级"卸载后重启仍正常"需要起服务占卡，不在本文件范围（见 `docs/release.md` §8）。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "vllm_ascend_split_batch"
BUNDLE_ID = "org.vllm-hust.split-batch-full-graph"

#: 实现载体模块名：生命周期命令不应 import 它们（发现只读元数据）。
IMPLEMENTATION_MODULES = (
    "vllm_ascend_split_batch.cascade_plugin",
    "vllm_ascend_split_batch.cascade_graph_plugin",
    "vllm_ascend_split_batch.cascade_runner_patch",
    "vllm_ascend_split_batch.planner",
)

#: 在**一个**子进程里跑完对照实验 + 生命周期 + import 审计，避免多次启动 vLLM 的开销。
_PROBE = r"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

from vllm_hust_ext.cli import main

config_path = Path(os.environ["PROBE_CONFIG"])
input_path = Path(os.environ["PROBE_INPUT"])
bundle = os.environ["PROBE_BUNDLE"]


def read_bundle_state():
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload["extensions"].get(bundle)


def run(*args):
    main(["extension", *args])


def capture(*args):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        main(["extension", *args])
    return buffer.getvalue()


def capture_json(*args):
    # vLLM's logging can share stdout with the command's own print(), so locate
    # the report by its first line and decode from there (trailing noise ignored).
    text = capture(*args)
    start = text.index('{\n  "extension_id"')
    payload, _ = json.JSONDecoder().raw_decode(text, start)
    return payload


result = {}

run("enable", bundle)
result["after_enable_no_configuration"] = read_bundle_state()
result["check_without_configuration"] = capture_json("check", bundle)

run("configure", bundle, "--file", str(input_path))
result["after_configure"] = read_bundle_state()
result["check_with_unknown_configuration"] = capture_json("check", bundle)

run("disable", bundle)
result["after_disable"] = read_bundle_state()
result["check_after_disable"] = capture_json("check", bundle)

run("forget", bundle)
result["after_forget"] = read_bundle_state()

result["imported_implementation_modules"] = sorted(
    name for name in sys.modules if name.startswith("vllm_ascend_split_batch.")
)

print(json.dumps(result, sort_keys=True))
"""

UNKNOWN_CONFIGURATION = {
    "totally_unknown_key": {"nested": [1, 2, 3], "flag": True},
    "another-unknown": "value",
}


def _run_probe(tmp_path: Path) -> dict:
    config_path = tmp_path / "config.json"
    input_path = tmp_path / "unknown-configuration.json"
    input_path.write_text(json.dumps(UNKNOWN_CONFIGURATION), encoding="utf-8")
    env = dict(os.environ)
    env.update(
        {
            "PROBE_CONFIG": str(config_path),
            "PROBE_INPUT": str(input_path),
            "PROBE_BUNDLE": BUNDLE_ID,
            "VLLM_HUST_EXT_CONFIG": str(config_path),
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(SRC), env.get("PYTHONPATH", "")])
            ),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def probe_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """跑一次探针，两个测试共用（一次子进程 = 一次 vLLM 导入）。"""
    return _run_probe(tmp_path_factory.mktemp("config-boundary"))


def test_carriers_gate_on_env_not_on_extension_configuration() -> None:
    """本仓的门控面只有 env；引入 `configuration` 消费就必须先改这条测试。"""
    offenders = []
    getenv_hits = 0
    for path in sorted(SRC.rglob("*.py")):
        if "fi_sampling" in path.parts:
            continue  # vendored 字节由 hash 固定，不参与本判据
        text = path.read_text(encoding="utf-8")
        getenv_hits += len(re.findall(r"os\.(?:getenv|environ)", text))
        for lineno, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\.configuration\b", line):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}:{line.strip()}"
                )
            if "VLLM_HUST_EXT_CONFIG" in line:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}:{line.strip()}"
                )
    assert not offenders, (
        "a carrier started consuming the extension `configuration` field (or the "
        "manager's config path); this repository's gate surface is env-only "
        f"(35 os.getenv/environ hits expected): {offenders}"
    )
    assert getenv_hits >= 30, (
        f"expected the env gate surface to stay large, found {getenv_hits} hits"
    )


def _assert_version_dependent_states(states: list[str], *, enabled: bool) -> None:
    """只对**不依赖宿主版本**的 state 做绝对断言。

    `check` 的 states 里,`installed`/`discovered` 恒在;`enabled` 只反映**用户开关**,
    与宿主版本无关(2026-09-26 在隔离环境实测:未命中点钉时仍是
    `['installed','discovered','incompatible','enabled']`);而 `compatible`/`configured`
    取决于宿主版本是否命中 manifest 的 `host.version_range` 点钉 —— 未命中时是
    `incompatible` 且没有 `configured`。

    本文件测的是**配置面边界**,不是版本兼容性,所以按"是否 compatible"分两支;
    早期版本写死了 `configured`,换宿主即红,属测试自身的脆弱(2026-09-26 修)。
    """
    assert "installed" in states and "discovered" in states, states
    assert ("enabled" in states) is enabled, states
    if "compatible" in states:
        assert "incompatible" not in states, states
        assert "configured" in states, states
    else:
        assert "incompatible" in states, states
        assert "configured" not in states, states


def test_unknown_configuration_keys_do_not_change_any_verdict(
    probe_result: dict,
) -> None:
    """对照实验：同一命令，只改 `configuration` 内容 ⇒ 判定逐字相同。"""
    without = probe_result["check_without_configuration"]
    with_unknown = probe_result["check_with_unknown_configuration"]
    assert without == with_unknown, (
        "an unknown `configuration` key changed the extension check verdict; "
        "the manager stores it verbatim and this repository does not read it, "
        "so the two runs must be byte-identical:\n"
        f"without={json.dumps(without, sort_keys=True)}\n"
        f"with={json.dumps(with_unknown, sort_keys=True)}"
    )
    _assert_version_dependent_states(with_unknown["states"], enabled=True)


def test_lifecycle_writes_metadata_only_and_keeps_unknown_keys(
    probe_result: dict,
) -> None:
    """enable/disable/forget 只碰元数据：不 import 实现，未知键往返保留。"""
    result = probe_result

    assert result["after_enable_no_configuration"]["enabled"] is True
    assert result["after_enable_no_configuration"]["configuration"] == {}

    after_configure = result["after_configure"]
    assert after_configure["enabled"] is True
    assert after_configure["configuration"] == UNKNOWN_CONFIGURATION, (
        "the manager must store the configuration verbatim (it has no schema and "
        "no interpretation); if it started normalising keys, this test must change"
    )

    after_disable = result["after_disable"]
    assert after_disable["enabled"] is False
    assert after_disable["configuration"] == UNKNOWN_CONFIGURATION, (
        "disable must not drop the stored configuration"
    )
    _assert_version_dependent_states(
        result["check_after_disable"]["states"], enabled=False
    )

    assert result["after_forget"] is None, (
        "forget must remove the stored state entirely"
    )

    imported = [
        name
        for name in result["imported_implementation_modules"]
        if name in IMPLEMENTATION_MODULES
    ]
    assert not imported, (
        "the lifecycle commands imported implementation carriers; discovery and "
        f"enablement must stay metadata-only: {imported}"
    )
