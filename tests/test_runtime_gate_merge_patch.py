# SPDX-License-Identifier: Apache-2.0

"""`VLLM_HUST_MOE_GATE_MERGE` 锚点变换的守卫（CPU-only）。

两件事:
1. 变换本身在**合成源码**上钉住（CI 必跑）: 两个锚点各命中一次、注入 import 与
   安装调用、结果可 compile;
2. 宿主源码在位时（NPU 主机）核对锚点仍唯一命中 —— 失配即 AnchorError（loader
   会 fail-closed 回 stock）, 所以这里失败等于提醒我们上游改了形状。

env 门与回滚: 表只由 `ENV_GATE_MERGE` 触发; 默认关。
"""

import importlib.util
import os
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "runtime_source_patches",
    HERE.parent / "src" / "vllm_ascend_split_batch" / "runtime_source_patches.py",
)
rsp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rsp)

#: 宿主源码候选位置（NPU 主机上才有）
HOST_CANDIDATES = tuple(
    pathlib.Path(base) / "vllm/model_executor/models/qwen3_next.py"
    for base in (
        "/root/vllm-workspace/vllm-hust",
        "/vllm-workspace/vllm",
    )
)

_MOE_IMPORT = (
    "from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP"
)

SYNTHETIC = (
    _MOE_IMPORT
    + """
import torch


class Block:
    def forward(
        self,
        hidden_states: torch.Tensor,
        already_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        return hidden_states
"""
)


def test_table_is_registered_under_its_own_env():
    assert rsp.ENV_GATE_MERGE == "VLLM_HUST_MOE_GATE_MERGE"
    modules = [module for env, module, _ in rsp.TABLES if env == rsp.ENV_GATE_MERGE]
    assert modules == [rsp.QWEN3NEXT]
    assert rsp.QWEN3NEXT == "vllm.model_executor.models.qwen3_next"


def test_transforms_inject_import_and_installer_call():
    patched, log = rsp.apply_transforms(SYNTHETIC, rsp._GATE_MERGE_TRANSFORMS)
    assert len(log) == 2
    assert "install_gate_merge as _hust_install_gate_merge" in patched
    assert patched.count("_hust_install_gate_merge(self)") == 1
    # 安装调用必须在 shared expert 被调用之前（在 forward 开头）
    body = patched.split("hust_install_gate_merge(self)")[1]
    assert body.lstrip().startswith("# NOTE: hidden_states")
    compile(patched, "qwen3_next_synthetic.py", "exec")


def test_anchor_mismatch_raises_anchor_error():
    broken = SYNTHETIC.replace("already_sequence_parallel: bool = False", "x: int = 0")
    with pytest.raises(rsp.AnchorError):
        rsp.apply_transforms(broken, rsp._GATE_MERGE_TRANSFORMS)


def _host_source():
    for path in HOST_CANDIDATES:
        if path.is_file():
            return path
    return None


def test_host_source_anchors_still_unique():
    path = _host_source()
    if path is None:
        pytest.skip("host qwen3_next.py not deployed")
    source = path.read_text(encoding="utf-8")
    patched, log = rsp.apply_transforms(source, rsp._GATE_MERGE_TRANSFORMS)
    assert len(log) == 2, log
    compile(patched, str(path), "exec")
    # 幂等性: 再施一次必须失配（说明只注入一次）
    with pytest.raises(rsp.AnchorError):
        rsp.apply_transforms(patched, rsp._GATE_MERGE_TRANSFORMS)


def test_install_is_default_off_and_env_armed(monkeypatch):
    monkeypatch.delenv(rsp.ENV_GATE_MERGE, raising=False)
    monkeypatch.setattr(rsp, "_INSTALLED", False)
    assert rsp.install() is False  # 默认关: 不注册钩子
    monkeypatch.setenv(rsp.ENV_GATE_MERGE, "1")
    monkeypatch.setattr(rsp, "_INSTALLED", False)
    monkeypatch.setattr(rsp.sys, "meta_path", list(rsp.sys.meta_path))
    try:
        assert rsp.install() is True
        assert isinstance(rsp.sys.meta_path[0], rsp._Finder)
    finally:
        # 清理: 摘掉本次插入的钩子, 避免污染其它用例
        rsp.sys.meta_path[:] = [
            entry for entry in rsp.sys.meta_path if not isinstance(entry, rsp._Finder)
        ]
        monkeypatch.setattr(rsp, "_INSTALLED", False)
    assert os.environ.get(rsp.ENV_GATE_MERGE) == "1"
