# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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

"""Source-level drift guard for ``aot_cache_guard_plugin``.

The carrier wraps ``vllm_ascend.compilation.compiler_interface`` internals that
the host owns.  If upstream renames/removes them, or adds the guard to the
fusion-pass branch itself (making this carrier redundant), the plugin must not
silently stop working or start double-patching.  These tests read the host
**source** and assert the exact shape the carrier relies on, so drift surfaces
in CI instead of as a mysterious engine-startup failure.

Read-only: no import of torch / vllm / vllm_ascend, no device use.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest


def _host_source() -> str:
    """Locate ``vllm_ascend/compilation/compiler_interface.py`` from disk only."""
    override = os.getenv("VLLM_ASCEND_HUST_ROOT")
    if override:
        root = pathlib.Path(override)
        if (root / "vllm_ascend").is_dir():
            root /= "vllm_ascend"
    else:
        spec = importlib.util.find_spec("vllm_ascend")
        assert spec is not None and spec.origin, (
            "host source not found; set VLLM_ASCEND_HUST_ROOT"
        )
        root = pathlib.Path(spec.origin).parent
    host = root / "compilation" / "compiler_interface.py"
    assert host.is_file(), f"host compile interface not found: {host}"
    return host.read_text(encoding="utf-8")


@pytest.mark.host_tree
def test_host_still_exposes_the_guard_the_carrier_wraps() -> None:
    src = _host_source()
    assert "def _disable_pytorch_aot_cache_for_npugraph_ex(" in src, (
        "host guard `_disable_pytorch_aot_cache_for_npugraph_ex` disappeared "
        "or was renamed; update aot_cache_guard_plugin.py (or drop the "
        "carrier if upstream fixed the branch)"
    )
    assert "class AscendCompiler" in src, (
        "AscendCompiler class disappeared; carrier target is gone"
    )
    assert "def compile(" in src, (
        "AscendCompiler.compile disappeared; carrier target is gone"
    )


@pytest.mark.host_tree
def test_fusion_pass_branch_is_still_unguarded() -> None:
    """If upstream guards this branch itself, the carrier is redundant -> delete it."""
    src = _host_source()
    # The else-branch of AscendCompiler.compile must still call fusion_pass_compile.
    assert "return fusion_pass_compile(" in src, (
        "AscendCompiler.compile no longer calls fusion_pass_compile directly; "
        "re-derive which path needs the guard"
    )
    # And that call must NOT already sit inside the host guard.  Naive textual
    # check: the guard context is only entered in npugraph_ex_compile.
    occurrences = src.count("with _disable_pytorch_aot_cache_for_npugraph_ex():")
    assert occurrences == 1, (
        "expected exactly 1 host-side guard entry (npugraph_ex path), "
        f"found {occurrences}; if upstream added one for the fusion-pass "
        "branch, drop aot_cache_guard_plugin.py"
    )


@pytest.mark.parametrize("value", [None, "0", "1", "", "false", " 0"])
def test_load_is_default_on_with_exact_opt_out(monkeypatch, value) -> None:
    from vllm_ascend_split_batch import aot_cache_guard_plugin as plugin

    monkeypatch.delenv(plugin.ENV_GUARD, raising=False)
    if value is not None:
        monkeypatch.setenv(plugin.ENV_GUARD, value)
    calls = []
    monkeypatch.setattr(plugin, "install", lambda: calls.append("install") or True)
    enabled = value != "0"
    assert plugin.is_enabled() is enabled
    assert plugin.load() is enabled
    assert calls == (["install"] if enabled else [])
