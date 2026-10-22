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

"""Default-off GeluAndMul OOT provider; register before model construction.

Only the provider subclass overrides forward_oot. The host class, native
implementation and compilation custom-op name remain unchanged. Uninstall
only affects future instances; serving rollback requires a worker restart.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV_ENABLE = "VLLM_HUST_FI_GELU"
HOST_MODULE = "vllm.model_executor.layers.activation"
HOST_CLASS_NAME = "GeluAndMul"
FORWARD_TARGET = "forward_oot"

_fi_api = None
_fi_import_failed = False
_warning_once = False
_registered_cls = None
_registry = None


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
    """Clear once-only state; call uninstall separately."""
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
    """Build the provider adapter with input guards and host fallback."""

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
                "GeluAndMul implementation for this call.",
                exc,
            )
            return orig_forward(self, x)

    forward_oot.__name__ = "forward_oot"
    forward_oot.__qualname__ = "FiGeluAndMul.forward_oot"
    return forward_oot


def install() -> bool:
    """Register an unoccupied OOT key without modifying the host class."""
    global _registered_cls, _registry
    if not is_enabled():
        return False
    try:
        import torch
        import vllm.model_executor.layers.activation as act_mod
        from vllm.model_executor.custom_op import CustomOp, op_registry_oot

        cls = getattr(act_mod, HOST_CLASS_NAME)
        if _registered_cls is not None:
            return (
                _registry is op_registry_oot
                and op_registry_oot.get(HOST_CLASS_NAME) is _registered_cls
            )
        if HOST_CLASS_NAME in op_registry_oot:
            _warn_once(
                "fi_gelu OOT key %s is occupied; keeping its provider.", HOST_CLASS_NAME
            )
            return False
        if not issubclass(cls, CustomOp):
            raise TypeError("GeluAndMul no longer inherits CustomOp")
        orig = getattr(cls, FORWARD_TARGET)
        op_name = cls.name
        fi_api = _load_fi_api()
        if fi_api is None:
            return False
        kernel_fn = getattr(fi_api, "gelu_and_mul", None)
        if not callable(kernel_fn):
            raise TypeError("fi_gelu.api.gelu_and_mul is unavailable")
        adapter = _make_forward(orig, kernel_fn, torch.bfloat16)

        class FiGeluAndMul(cls):
            forward_oot = adapter

        try:
            CustomOp.register_oot(name=HOST_CLASS_NAME)(FiGeluAndMul)
            FiGeluAndMul.name = op_name
        except Exception:
            if op_registry_oot.get(HOST_CLASS_NAME) is FiGeluAndMul:
                del op_registry_oot[HOST_CLASS_NAME]
            raise
    except Exception as exc:
        _warn_once("fi_gelu OOT registration unavailable (%r); keeping host path.", exc)
        return False

    _registered_cls = FiGeluAndMul
    _registry = op_registry_oot
    logger.warning("fi_gelu OOT provider registered for new GeluAndMul instances.")
    return True


def uninstall() -> None:
    """Remove only our registry entry; existing instances are not hot-swapped."""
    global _registered_cls, _registry
    if (
        _registered_cls is not None
        and _registry is not None
        and _registry.get(HOST_CLASS_NAME) is _registered_cls
    ):
        del _registry[HOST_CLASS_NAME]
    _registered_cls = None
    _registry = None


def load() -> bool:
    """``vllm.general_plugins`` entry point (default-off)."""
    if not is_enabled():
        return False
    return install()
