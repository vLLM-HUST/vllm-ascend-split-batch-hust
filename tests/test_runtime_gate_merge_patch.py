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
from _deployment import deployed_file

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "runtime_source_patches",
    HERE.parent / "src" / "vllm_ascend_split_batch" / "runtime_source_patches.py",
)
rsp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rsp)

#: 宿主源码候选位置（NPU 主机上才有）
HOST_CANDIDATES = tuple(
    pathlib.Path(base) / "vllm/model_executor/models/utils.py"
    for base in (
        "/root/vllm-workspace/vllm-hust",
        "/vllm-workspace/vllm",
    )
)

_MOE_IMPORT = "from vllm.logger import init_logger"

SYNTHETIC = (
    _MOE_IMPORT
    + """
import torch


class AutoWeightsLoader:
    def load_weights(self, weights, *, mapper=None):
        autoloaded_weights = set()
        self._check_skipped_aliases(autoloaded_weights)
        return autoloaded_weights
"""
)


def test_table_is_registered_under_its_own_env():
    assert rsp.ENV_GATE_MERGE == "VLLM_HUST_MOE_GATE_MERGE"
    modules = [module for env, module, _ in rsp.TABLES if env == rsp.ENV_GATE_MERGE]
    assert modules == [rsp.MODELS_UTILS]
    assert rsp.MODELS_UTILS == "vllm.model_executor.models.utils"


def test_transforms_inject_import_and_installer_call():
    patched, log = rsp.apply_transforms(SYNTHETIC, rsp._GATE_MERGE_TRANSFORMS)
    assert len(log) == 2
    assert "install_gate_merge_all as _hust_install_gate_merge_all" in patched
    # 安装注入模型级 post-load 钩子之后、且只注入一次
    assert patched.count("_hust_install_gate_merge_all(self.module)") == 1
    compile(patched, "qwen3_next_synthetic.py", "exec")


def test_installer_lands_in_the_post_load_hook_only():
    """安装只能落在 post-load 钩子里（不得进 forward: 实测过 dynamo 崩溃）。"""
    patched, _ = rsp.apply_transforms(SYNTHETIC, rsp._GATE_MERGE_TRANSFORMS)
    hook = patched.index("def load_weights(")
    call = patched.index("_hust_install_gate_merge_all(self.module)")
    assert call > hook
    assert "def forward(" not in patched


def test_anchor_mismatch_raises_anchor_error():
    broken = SYNTHETIC.replace(
        "        self._check_skipped_aliases(autoloaded_weights)",
        "        self._check_skipped_aliases_typo(autoloaded_weights)",
    )
    with pytest.raises(rsp.AnchorError):
        rsp.apply_transforms(broken, rsp._GATE_MERGE_TRANSFORMS)


def _host_source():
    for path in HOST_CANDIDATES:
        if deployed_file(path):
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
    # 只注入一次（锚点文本保留 → 二次施加会变 2 次, 故这里钉住计数）
    assert patched.count("_hust_install_gate_merge_all(self.module)") == 1


def test_install_is_default_off_and_env_armed(monkeypatch):
    monkeypatch.delitem(rsp.sys.modules, rsp.MODELS_UTILS, raising=False)
    monkeypatch.delenv(rsp.ENV_GATE_MERGE, raising=False)
    monkeypatch.delitem(rsp.sys.modules, rsp.GDN, raising=False)
    monkeypatch.setattr(rsp, "_INSTALLED", False)
    monkeypatch.setattr(rsp.sys, "meta_path", list(rsp.sys.meta_path))
    assert rsp.install() is True
    assert rsp.GDN in rsp.sys.meta_path[0]._targets
    assert rsp.MODELS_UTILS not in rsp.sys.meta_path[0]._targets
    monkeypatch.setenv(rsp.ENV_GATE_MERGE, "1")
    monkeypatch.setattr(rsp, "_INSTALLED", False)
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


def test_install_rejects_already_imported_target(monkeypatch):
    monkeypatch.setenv(rsp.ENV_GATE_MERGE, "1")
    monkeypatch.setattr(rsp, "_INSTALLED", False)
    monkeypatch.setitem(rsp.sys.modules, rsp.MODELS_UTILS, object())
    before = list(rsp.sys.meta_path)
    assert rsp.install() is False
    assert rsp.sys.meta_path == before


def test_finder_does_not_pop_target_on_find_spec_only():
    """find_spec 探测不得卸掉目标钩子。"""
    import types

    finder = rsp._Finder({rsp.MODELS_UTILS: rsp._GATE_MERGE_TRANSFORMS})
    parent = types.ModuleType("vllm.model_executor.models")
    parent.__path__ = []  # 触发 PathFinder 失败分支, 只验证"不卸钩"
    saved = rsp.sys.modules.get("vllm.model_executor.models")
    rsp.sys.modules["vllm.model_executor.models"] = parent
    try:
        assert finder.find_spec(rsp.MODELS_UTILS) is None  # 找不到 spec 也不许丢目标
        assert rsp.MODELS_UTILS in finder._targets
    finally:
        if saved is not None:
            rsp.sys.modules["vllm.model_executor.models"] = saved
        else:
            rsp.sys.modules.pop("vllm.model_executor.models", None)
