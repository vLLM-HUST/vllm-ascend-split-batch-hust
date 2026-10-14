# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Hoist MoE router/gate weights' fp32 copy out of the decode hot path.

The carrier is deliberately opt-in.  vLLM general plugins are loaded before the
worker constructs the model, so this module wraps the active model-runner
``load_model`` seam and fills the cache only after the loader has materialized
(and, where applicable, converted) the weights.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading

logger = logging.getLogger(__name__)

ENV_ENABLE = "VLLM_HUST_GATE_FP32"
_MARKER = "_vllm_hust_gate_fp32_wrapped"
_LOCAL = threading.local()
_SERVE_LOGGER = None


def is_enabled() -> bool:
    """Enable only with an explicit ``1``; unset/other values are stock mode."""
    return os.getenv(ENV_ENABLE) == "1"


def _serve_logger():
    """Logger on vLLM's configured handler tree (visible in ``vllm serve`` logs).

    ``logging.getLogger(__name__)`` propagates to an unconfigured root logger, so
    its records never reach the serve output (same F7 note as ``cascade_plugin``).
    The import is lazy and best-effort so importing this module stays cheap and
    device-stack-free when the carrier is disabled.
    """
    global _SERVE_LOGGER
    if _SERVE_LOGGER is None:
        try:
            from vllm.logger import init_logger

            _SERVE_LOGGER = init_logger("vllm.vllm_ascend_split_batch.gate_fp32_plugin")
        except Exception:  # noqa: BLE001 -- logging must never break startup
            _SERVE_LOGGER = logger
    return _SERVE_LOGGER


def _cache_gate_weights(model) -> int:
    """Attach one contiguous fp32 copy to every module whose leaf name is gate.

    The exact leaf-name check intentionally excludes ``shared_expert_gate``.
    It also avoids depending on a vLLM/Ascend private class name, while the
    post-load timing ensures quantized checkpoints are read after conversion.
    """
    import torch

    cached = 0
    for name, gate in model.named_modules():
        if name.rsplit(".", 1)[-1] != "gate":
            continue
        weight = getattr(gate, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if hasattr(gate, "weight_fp32"):
            continue
        # to(dtype) is the required conversion from the already materialized
        # weight; contiguous makes the ND layout requirement explicit.
        gate.weight_fp32 = weight.to(dtype=torch.float32).contiguous()
        cached += 1
    return cached


def _install_on_class(cls) -> bool:
    original = getattr(cls, "load_model", None)
    if not callable(original) or getattr(original, _MARKER, False):
        return False

    def load_model_with_gate_fp32(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if model is not None:
            count = _cache_gate_weights(model)
            _serve_logger().info("Gate fp32 cache populated: %d MoE gates", count)
        return result

    setattr(load_model_with_gate_fp32, _MARKER, True)
    load_model_with_gate_fp32.__name__ = getattr(original, "__name__", "load_model")
    load_model_with_gate_fp32.__doc__ = getattr(original, "__doc__", None)
    cls.load_model = load_model_with_gate_fp32
    return True


def install() -> bool:
    """Wrap model runners; missing host seams fail open."""
    if getattr(_LOCAL, "installing", False):
        return False
    _LOCAL.installing = True
    try:
        # The active runner lives in ``vllm_ascend.worker.model_runner_v1``.
        # Importing that module *directly* can trip a circular import
        # (``device_op`` <-> ``ops/__init__`` -> ``fused_moe`` -> ...) unless the
        # ``vllm_ascend.ops`` package has been materialized first.  The real
        # worker reaches it through another order, but this plugin runs in
        # ``WorkerBase.__init__`` (``load_general_plugins``) *before* the worker
        # module is resolved -- so without this warm-up the probe silently wraps
        # only the (unused) v2/GPU runners.  Best-effort: any host without the
        # package (or a future rework) simply falls through to the plain probe
        # below, and a warm-up failure must never break engine startup.
        with contextlib.suppress(Exception):  # a warm-up must never break startup
            import vllm_ascend.ops  # noqa: F401
        candidates = (
            ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
            ("vllm_ascend.worker.v2.model_runner", "NPUModelRunner"),
            ("vllm_ascend._310p.worker.v2.model_runner", "NPUModelRunner310V2"),
            ("vllm_ascend._310p.model_runner_310", "NPUModelRunner310"),
            ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
            ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
        )
        installed = 0
        for module_name, class_name in candidates:
            try:
                module = __import__(module_name, fromlist=[class_name])
                if _install_on_class(getattr(module, class_name)):
                    installed += 1
            except (ImportError, AttributeError):
                continue
        return installed > 0
    except Exception:  # noqa: BLE001 -- plugin must never break engine startup
        logger.exception("Gate fp32 plugin failed open; host remains usable")
        return False
    finally:
        _LOCAL.installing = False


def load() -> bool:
    """vLLM general-plugin entry point; default-off via ``VLLM_HUST_GATE_FP32``."""
    return install() if is_enabled() else False
