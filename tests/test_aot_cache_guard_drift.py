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

import pathlib
import sys

import pytest

_PLUGIN = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "vllm_ascend_split_batch"
    / "aot_cache_guard_plugin.py"
)


def _host_source() -> str:
    """Locate ``vllm_ascend/compilation/compiler_interface.py`` from disk only."""
    import vllm_ascend  # noqa: PLC0415 -- cheap: package dir only

    host = (
        pathlib.Path(vllm_ascend.__file__).resolve().parent
        / "compilation"
        / "compiler_interface.py"
    )
    if not host.exists():
        pytest.skip(f"host compile interface not found: {host}")
    return host.read_text(encoding="utf-8")


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


def test_load_is_default_on_with_opt_out() -> None:
    sys.path.insert(0, str(_PLUGIN.parent.parent))
    from vllm_ascend_split_batch import (
        aot_cache_guard_plugin as plugin,  # noqa: PLC0415
    )

    assert plugin.is_enabled() is True, (
        "carrier must be default-on (frozen config has no env var to spare)"
    )
    import os  # noqa: PLC0415

    os.environ[plugin.ENV_GUARD] = "0"
    try:
        assert plugin.is_enabled() is False, (
            "opt-out token `0` must disable the carrier"
        )
        assert plugin.load() is False, "disabled carrier must not install anything"
    finally:
        os.environ.pop(plugin.ENV_GUARD, None)
    assert plugin.is_enabled() is True
