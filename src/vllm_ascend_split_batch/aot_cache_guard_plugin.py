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

"""AOTAutograd cache guard for the **fusion-pass** compile path (fidelity fix).

Defect (fork baseline ``vllm-ascend-hust`` main, torch 2.13.0 / vLLM 0.28.1):
``AscendCompiler.compile`` has two branches.  The ``enable_npugraph_ex`` branch
enters ``_disable_pytorch_aot_cache_for_npugraph_ex()``, whose own docstring says
torch 2.13's bundled AOTAutograd cache "requires every backend compiler to
return an ``OutputCode``".  The ``else`` branch -- ``fusion_pass_compile`` ->
``compile_fx`` -> ``aot_autograd(fw_compiler=inner_compile)`` -- returns a plain
``GraphModule`` and is **not** guarded, so engine initialization dies with::

    AssertionError: expected OutputCode, got
      <class 'torch.fx.graph_module.GraphModule.__new__.<locals>.GraphModuleImpl'>
      torch/_functorch/_aot_autograd/autograd_cache.py:1431 unwrap_output_code
      <- graph_compile.py:531 _cache_inference_info   (cache write path)
      <- vllm_ascend/compilation/compiler_interface.py:80 compile_fx
      <- vllm_ascend/compilation/compiler_interface.py:97 fusion_pass_compile

A1-contract impact (why this is not optional): the acceptance configuration
freezes ``compile_mode=vllm_compile`` / ``enforce_eager=false`` / ``cudagraph
mode=piecewise`` (V3.8 表附-2/3), i.e. it always takes this branch.  The usual
workaround ``VLLM_DISABLE_COMPILE_CACHE=1`` is **forbidden** as a measurement
configuration -- V3.8 表附-4 ends with "其他执行变量｜禁止；不得进入正式测量".
So the fix must live in the engine (B1) rather than in an env var, which is
exactly the one variable the contract lets B1 change (表附-1: aside from engine
code identity, B0/B1 are identical item by item).

Carrier mechanism: wrap ``AscendCompiler.compile`` (a class attribute on an
imported object -- no host source edit, no second ``register_oot``).  The
wrapper reproduces the host's own branch predicate and enters the host's own
guard context only for the unguarded branch, so a future host fix makes this
carrier a no-op rather than a conflict.

Enablement: **default-on, opt out with** ``VLLM_HUST_AOT_CACHE_GUARD=0``.
This deviates from the repo's default-off convention on purpose: a default-off
carrier would require the forbidden env var to make the frozen config boot.
With the opt-out set, behavior is bit-identical to stock (the wrapper is not
installed at all, so ``AscendCompiler.compile`` is the host function).

Fail-open: any import/attribute/predicate failure leaves the host untouched and
records a drift marker in ``<tmp>/vllm_hust_aot_cache_guard.json`` for the
source-level guard test (``tests/test_aot_cache_guard_drift.py``).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time

#: ``"0"`` disables the carrier; anything else (including unset) enables it.
ENV_GUARD = "VLLM_HUST_AOT_CACHE_GUARD"

_LOCAL = threading.local()
_MARKER = "_vllm_hust_aot_cache_guard_wrapped"


def is_enabled() -> bool:
    """Opt-out semantics: only the exact token ``"0"`` disables."""
    return os.getenv(ENV_GUARD, "1") != "0"


def _drift_path() -> str:
    return os.path.join(tempfile.gettempdir(), "vllm_hust_aot_cache_guard.json")


def _record(status: str, detail: str = "") -> None:
    try:
        with open(_drift_path(), "w", encoding="utf-8") as fh:
            json.dump(
                {"status": status, "detail": detail, "ts": time.time()},
                fh,
                ensure_ascii=False,
            )
    except Exception:
        pass


def _npugraph_ex_enabled_else_none(ci) -> bool | None:
    """Return the host's branch predicate, or None when it cannot be read."""
    try:
        from vllm_ascend.ascend_config import get_ascend_config

        return bool(get_ascend_config().ascend_compilation_config.enable_npugraph_ex)
    except Exception:
        return None


def install() -> bool:
    """Install the wrapper.  Idempotent; returns False on any drift/failure."""
    if getattr(_LOCAL, "installing", False):
        return False
    _LOCAL.installing = True
    try:
        from vllm_ascend.compilation import compiler_interface as ci

        guard = getattr(ci, "_disable_pytorch_aot_cache_for_npugraph_ex", None)
        if guard is None:
            _record(
                "drift",
                "compiler_interface._disable_pytorch_aot_cache_for_npugraph_ex missing",
            )
            return False

        compiler_cls = getattr(ci, "AscendCompiler", None)
        orig = (
            getattr(compiler_cls, "compile", None)
            if compiler_cls is not None
            else None
        )
        if not callable(orig):
            _record("drift", "AscendCompiler.compile missing")
            return False
        if getattr(orig, _MARKER, False):
            _record("already-installed")
            return True

        def compile_with_aot_cache_guard(self, *args, **kwargs):
            npugraph = _npugraph_ex_enabled_else_none(ci)
            if npugraph:  # host guards this branch itself (npugraph_ex_compile)
                return orig(self, *args, **kwargs)
            # npugraph False, or predicate unreadable: entering the guard is
            # harmless when it turns out to be unnecessary, so prefer coverage.
            with guard():
                return orig(self, *args, **kwargs)

        setattr(compile_with_aot_cache_guard, _MARKER, True)  # noqa: B010 -- 函数标记，供幂等判定
        compile_with_aot_cache_guard.__name__ = getattr(orig, "__name__", "compile")
        compile_with_aot_cache_guard.__doc__ = getattr(orig, "__doc__", None)
        compiler_cls.compile = compile_with_aot_cache_guard
        _record(
            "installed",
            "npugraph_ex_branch_untouched; wrapped="
            f"{ci.AscendCompiler.__name__}.compile",
        )
        return True
    except Exception as exc:  # noqa: BLE001 -- load() must never break the engine
        _record("failed", f"{type(exc).__name__}: {exc}")
        return False
    finally:
        _LOCAL.installing = False


def load() -> bool:
    """``vllm.general_plugins`` entry point (default-on, opt out via env)."""
    if not is_enabled():
        return False
    return install()
