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


class Model:
    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class ExpertModel:
    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
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
    assert "install_gate_merge_all as _hust_install_gate_merge_all" in patched
    # 每个 load_weights 各注入一次（合成源码里有两个实现）
    assert patched.count("_hust_install_gate_merge_all(self)") == 2
    # 安装必须在 load_weights 返回之前完成（权重就绪 + 不在 compile 区内）
    assert "loaded = loader.load_weights(" in patched
    assert "return loaded" in patched
    assert "def forward(" not in patched.split("_hust_install_gate_merge_all")[0]
    compile(patched, "qwen3_next_synthetic.py", "exec")


def test_installer_never_lands_in_a_compiled_forward():
    """forward 里不得出现安装调用（否则 Python 副作用会进图捕获: 已实测崩溃）。"""
    patched, _ = rsp.apply_transforms(SYNTHETIC, rsp._GATE_MERGE_TRANSFORMS)
    for chunk in patched.split("def forward(")[1:]:
        assert "_hust_install_gate_merge" not in chunk.split("def ")[0]


def test_anchor_mismatch_raises_anchor_error():
    broken = SYNTHETIC.replace(
        "return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)",
        "return loader.load_weights(weights)",
    )
    with pytest.raises(rsp.AnchorError):
        rsp.apply_transforms(broken, rsp._GATE_MERGE_TRANSFORMS)


def _host_source():
    for path in HOST_CANDIDATES:
        if path.is_file():
            return path
    return None


def test_host_source_anchors_still_unique():  # noqa: D103
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
