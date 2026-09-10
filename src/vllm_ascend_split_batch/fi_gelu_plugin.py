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

"""fi_gelu integration into the vLLM ``GeluAndMul`` activation (default-off).

Mechanism (W3.1c wiring): the ``vllm.general_plugins`` entry point ``fi-gelu``
calls :func:`load`, which -- only when ``VLLM_HUST_FI_GELU == "1"`` -- class-
patches the *actually dispatched* forward method of the host GeGLU activation
``vllm.model_executor.layers.activation.GeluAndMul`` with a thin adapter onto
``vllm_ascend_split_batch.fi_gelu.api.gelu_and_mul`` (the fused triton-ascend
exact-erf kernel).

Dispatch finding (probed on NPU, 2026-09-10, card 7): ``CustomOp.dispatch_forward``
sees ``GeluAndMul.enabled() is True`` and, because the Ascend ``NPUPlatform``
declares ``_enum = PlatformEnum.OOT``, ``current_platform.is_out_of_tree()`` is
True -- so the bound method stored in ``_forward_method`` is the inherited
``CustomOp.forward_oot`` (which in turn calls ``GeluAndMul.forward_native``).
``GeluAndMul`` defines neither ``forward_oot`` nor ``forward_cuda`` itself, so
the patch target is the inherited ``forward_oot`` (``FORWARD_TARGET``); setting
``GeluAndMul.forward_oot`` shadows the base method for the GeGLU class only.
Full evidence (``_forward_method`` repr, live call-chain trace) is in
``tests/test_npu_w31_gelu_plugin.py``.

Default-off semantics: with the env unset ``load()`` returns before importing
``torch``/``triton``/``fi_gelu`` or touching the host class -- nothing is
patched and no kernel module is imported, so the process is bit-identical to
stock vllm/vllm-ascend (asserted by a subprocess ``sys.modules`` purity test).

The patched forward is graph-capture safe: the enable decision is made once at
``load()`` time (no per-call env read), and the body contains only tensor/attr
inspection plus either the kernel call or a delegation to the original method.
``bench/results/bench_w31_gelu_graph.json`` established that the kernel is
ACL-graph capturable and replay-bit-exact.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: master switch, default off (repo AGENTS.md: every capability env-gated).
ENV_ENABLE = "VLLM_HUST_FI_GELU"

#: host seam (see module docstring / HOST_CONTRACT.md).
HOST_MODULE = "vllm.model_executor.layers.activation"
HOST_CLASS_NAME = "GeluAndMul"
#: the forward method the host dispatch actually selects on NPU (W3.1c probe).
FORWARD_TARGET = "forward_oot"
#: marker set on the patched class so a second ``install()`` is a no-op.
_PATCH_MARKER = "_fi_gelu_plugin_patched"

# Loaded lazily on the first enable (default-off imports nothing).
_fi_api = None
_fi_import_failed = False
_warning_once = False
#: original (unbound) ``GeluAndMul.forward_oot`` captured at patch time; kept so
#: the fallback path -- and ``_unpatch`` / differential tests -- can reach the
#: untouched implementation.
_orig_forward = None
#: host class currently patched (None when off).
_installed_cls = None
#: True when the class already had its own ``forward_oot`` (it inherits it today,
#: so ``_unpatch`` must delete the attribute rather than restore one).
_owned_forward = False


def is_enabled() -> bool:
    """True only for the exact enable token (default-off, no truthiness fuzz)."""
    return os.getenv(ENV_ENABLE) == "1"


def _warn_once(message: str, *args) -> None:
    global _warning_once
    if _warning_once:
        return
    logger.warning(message, *args)
    _warning_once = True


def _reset_for_tests() -> None:
    """Clear once-only state (does not unpatch; call ``_unpatch`` for that)."""
    global _fi_api, _fi_import_failed, _warning_once
    _fi_api = None
    _fi_import_failed = False
    _warning_once = False


def _load_fi_api():
    """Import the vendored ``fi_gelu`` host API (lazy, fail-open).

    Returns the module or ``None``; the caller stays on the host chain.  The
    import is deferred so the default-off path never pays for ``triton``.
    """
    global _fi_api, _fi_import_failed
    if _fi_api is not None:
        return _fi_api
    if _fi_import_failed:
        return None
    try:
        from .fi_gelu import api as fi_api
    except Exception as exc:  # noqa: BLE001 -- fail-open, host chain must survive
        _fi_import_failed = True
        _warn_once(
            "VLLM_HUST_FI_GELU=1 but the vendored fi_gelu package could not be "
            "imported (%r); serving stays on the host GeluAndMul implementation "
            "for this process.",
            exc,
        )
        return None
    _fi_api = fi_api
    return _fi_api


def _kernel_eligible(x, bf16_dtype, device_type: str = "npu") -> bool:
    """Cheap, capture-safe eligibility test for the fused kernel.

    Mirrors the guards of ``fi_gelu.api.gelu_and_mul`` (bf16, >= 2D, even and
    non-empty ``2*inter`` last dim) and additionally requires an NPU tensor, so
    anything the kernel would reject is delegated to the host implementation
    instead of raising (fail-open).  Pure attribute inspection: no imports, no
    host sync.
    """
    if x.dtype != bf16_dtype:
        return False
    if x.dim() < 2:
        return False
    last = x.shape[-1]
    if last <= 0 or last % 2 != 0:
        return False
    return getattr(x.device, "type", None) == device_type


def _make_forward(orig_forward, kernel_fn, bf16_dtype, device_type: str = "npu"):
    """Build the ``forward_oot`` replacement bound to ``kernel_fn``.

    The enable decision is already baked in (this function is only called from
    ``install()``); the body is a static pair of branches, so the call is safe
    inside ``torch.npu.graph`` capture.
    """

    def forward_oot(self, x):
        # ``approximate="tanh"`` is a different function (tanh gelu); the fused
        # kernel is exact-erf only, so those instances stay on the host chain.
        if self.approximate != "none" or not _kernel_eligible(
            x, bf16_dtype, device_type
        ):
            return orig_forward(self, x)
        try:
            return kernel_fn(x)
        except Exception as exc:  # noqa: BLE001 -- fail-open to the host chain
            _warn_once(
                "fi_gelu kernel call failed (%r); falling back to the host "
                "GeluAndMul implementation for the rest of this process.",
                exc,
            )
            return orig_forward(self, x)

    forward_oot.__name__ = "forward_oot"
    forward_oot.__qualname__ = "GeluAndMul.forward_oot"
    return forward_oot


def install() -> bool:
    """Class-patch ``GeluAndMul.forward_oot`` onto the fused kernel.

    Returns True when the patched method is in place.  Idempotent; never raises
    into the plugin loader (a missing/unusable host or kernel fails open with
    one warning and leaves the stock behavior untouched).
    """
    global _orig_forward, _installed_cls, _owned_forward
    try:
        import torch
        import vllm.model_executor.layers.activation as act_mod
    except Exception as exc:  # noqa: BLE001 -- fail-open (the host module may be
        # renamed in a future fork; load() is an entry point and must never
        # raise, or the engine process dies at startup).
        _warn_once(
            "VLLM_HUST_FI_GELU=1 but the host activation module could not be "
            "imported (%r); serving stays on the host GeluAndMul implementation.",
            exc,
        )
        return False

    if _installed_cls is act_mod.GeluAndMul and getattr(
        _installed_cls, _PATCH_MARKER, False
    ):
        return True

    cls = getattr(act_mod, HOST_CLASS_NAME, None)
    if cls is None:
        _warn_once(
            "VLLM_HUST_FI_GELU=1 but %s.%s no longer exists; serving stays on "
            "the host activation chain.",
            HOST_MODULE,
            HOST_CLASS_NAME,
        )
        return False

    fi_api = _load_fi_api()
    if fi_api is None:
        return False
    kernel_fn = getattr(fi_api, "gelu_and_mul", None)
    if kernel_fn is None:
        _warn_once(
            "VLLM_HUST_FI_GELU=1 but fi_gelu.api.gelu_and_mul is missing; "
            "serving stays on the host GeluAndMul implementation.",
        )
        return False

    # Resolve the target/backup BEFORE mutating the class so a failure cannot
    # leave a half-installed host.
    try:
        orig = getattr(cls, FORWARD_TARGET)
        owned = FORWARD_TARGET in cls.__dict__
        patched = _make_forward(orig, kernel_fn, torch.bfloat16)
        setattr(cls, FORWARD_TARGET, patched)
        setattr(cls, _PATCH_MARKER, True)
    except Exception as exc:  # noqa: BLE001 -- fail-open
        _warn_once(
            "VLLM_HUST_FI_GELU=1 but patching GeluAndMul.%s failed (%r); "
            "serving stays on the host activation chain.",
            FORWARD_TARGET,
            exc,
        )
        return False

    _orig_forward = orig
    _installed_cls = cls
    _owned_forward = owned
    logger.warning(
        "fi_gelu activation path is ACTIVE (VLLM_HUST_FI_GELU=1): "
        "GeluAndMul.forward_oot (approximate='none') now runs the fused "
        "triton-ascend exact-erf kernel; non-bf16 / tanh / malformed inputs "
        "delegate to the host implementation.",
    )
    return True


def _unpatch() -> None:
    """Restore the host ``GeluAndMul.forward_oot`` (tests / unload)."""
    global _orig_forward, _installed_cls, _owned_forward
    cls = _installed_cls
    if cls is not None:
        if _owned_forward and _orig_forward is not None:
            setattr(cls, FORWARD_TARGET, _orig_forward)
        elif FORWARD_TARGET in cls.__dict__:
            delattr(cls, FORWARD_TARGET)
        if _PATCH_MARKER in cls.__dict__:
            delattr(cls, _PATCH_MARKER)
    _orig_forward = None
    _installed_cls = None
    _owned_forward = False


def load() -> bool:
    """``vllm.general_plugins`` entry point (default-off)."""
    if not is_enabled():
        # Default-off: no import of torch/vllm/triton/fi_gelu, no patch.
        return False
    return install()
