# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GDN ``in_proj`` merge carrier (``VLLM_HUST_GDN_MERGE_INPROJ``), default-off.

Migration of the accepted **F2** mechanism (load-time ``in_proj_ba`` ->
``in_proj_qkvz`` weight merge, one ``[6176, 2048]`` ``MatMulV2`` per GDN layer
instead of two) out of the vllm-ascend *runtime tree* patch and into this
plugin library.

Evidence / statistics scope
---------------------------
* Accepted verdict: L1 card ``f2_accepted_verdict_1016``
  (``opt-ops/kb-working``), caliber ``mmlu-pro-eval-traffic``, cohort
  ``028-qwen35-35b-a3b-tp2-fullgraph``: decode ``MatMulV2`` -30.1/step
  (exactly 30 layers x 1), device kernel time -15.4% per pattern, TPOT no
  regression (13.539 -> 13.390 ms), L3 bitwise identical, zero new
  ``ConcatD``/``CopyD``.
* Wiring lessons: ``f2_wiring_and_netloader_lessons_1016`` -- this stack's serve
  path goes through the Ascend netloader, whose ``load_model`` never calls the
  vllm model loader's ``process_weights_after_loading``, so the post-load hook
  must be called explicitly from the worker after ``get_model()``; and writes to
  module state *inside* a traced ``forward`` are dynamo-unsafe (that is what
  killed the v1 lazy-``cat`` variant).
* The numbers above are **the accepted runtime-tree measurement**.  Any claim for
  this carrier must be re-accepted with a fresh L4 window on the same caliber
  (``mmlu-pro-eval-traffic``) -- see ``docs/architecture.md`` §5.

What this carrier implements
----------------------------
1. the exact-token env gate (only ``"1"``) with a **zero-import** ``load()``;
2. a wrap of the active model runner's ``load_model`` with the same probe list,
   warm-up and fail-open contract as ``gate_fp32_plugin``, so the post-load hook
   runs after the loader materialized the weights and before
   compilation/graph capture;
3. the load-time weight conversion itself: ``plan_merge`` (strict structural
   guards, refuses silently-wrong rewrites) plus the pure
   ``merge_inproj_weight`` helper, both unit-tested on CPU.

Why the carrier is fail-closed (and what would unblock it)
----------------------------------------------------------
The merged weight is only worth anything if the **forward** consumes it as one
matmul.  The host computes the two projections inline and monolithically:
``AscendGatedDeltaNetAttention.forward`` (rebound onto the core
``QwenGatedDeltaNetAttention`` by ``patch_qwen3_5``) calls
``self.in_proj_qkvz(hidden_states)`` and ``self.in_proj_ba(hidden_states)``
back-to-back and then runs the custom op + norm + ``out_proj`` in the same
method body; no projection/custom-op seam exists.

A plugin-side interception therefore needs exactly one of the two things this
contract forbids:

* **per-call Python state** (a module attribute or thread-local written inside the
  traced forward) to thread the single fused output into the second projection
  call -- the dynamo-unsafe pattern the F2 v1 carrier was rejected for;
* **a copy of the host forward body** (host private source; the plugin must not
  carry it, and it would drift silently against the five inline branches).

Merging the weight *without* a consumer is not an option either: it changes no
kernel count and feeds a ``[*, 6176]`` tensor into the stock
``mixed_qkvz.split([qkv_size, z_size])`` (6144 columns) -- i.e. it would trade a
correct stock model for an unverified one.  So with the current host this
carrier **refuses to rewrite any weight** (one warning, host stays bit-identical
to stock) instead of shipping a carrier that cannot do what it claims.

Unblocking condition (any one of these, then the merge guard may lift):
``AscendGatedDeltaNetAttention.forward`` grows an env-gated merged-consumption
branch for the flat ``gqa_interleaved_layout=False`` layout, or the projection
step is split into its own method / custom op the plugin can bind without
copying the body.  ``FORWARD_CONSUMER_BLOCKER`` below is the machine-readable
statement of the same thing.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading

logger = logging.getLogger(__name__)

ENV_ENABLE = "VLLM_HUST_GDN_MERGE_INPROJ"

#: Marker on our ``load_model`` wrapper (idempotency across the chain of plugins
#: that wrap the same host method, e.g. ``gate-fp32``).
_MARKER = "_vllm_hust_gdn_merge_inproj_wrapped"
#: Marker set on a GDN layer whose weight was merged (one-shot, see F2 v2).
_LAYER_MARK = "_vllm_hust_gdn_inproj_merged"

_LOCAL = threading.local()
_SERVE_LOGGER = None

#: ``(cls, original_load_model)`` pairs, so ``uninstall()`` can put the host back.
_INSTALLED: list[tuple[type, object]] = []

#: The F2 shape fixed by the accepted evidence: ``in_proj_qkvz[6144, 2048]`` +
#: ``in_proj_ba[32, 2048]`` -> ``[6176, 2048]`` (bf16, TP=2, flat qkvz layout).
QKVZ_OUT_ROWS = 6144
BA_OUT_ROWS = 32
MERGED_OUT_ROWS = QKVZ_OUT_ROWS + BA_OUT_ROWS

#: The accepted applicability boundary of F2 (see the verdict card): flat qkvz
#: layout and the unquantized bf16 stack.
ACCEPTED_LAYOUT = "gqa_interleaved_layout=False"
ACCEPTED_DTYPE = "bfloat16"

#: Device types this carrier is willing to rewrite weights on.  ``meta`` (and
#: anything unknown) is refused: an unmaterialized weight cannot be merged.
_ALLOWED_DEVICE_TYPES = ("npu", "cuda", "cpu")

#: Host objects the carrier binds to (read-only; names, not source).
_HOST_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_HOST_CLASS = "QwenGatedDeltaNetAttention"

#: The active runner seam (same probe order as ``gate_fp32_plugin``); the first
#: entry is the class ``worker/worker.py`` actually instantiates.
_CANDIDATES = (
    ("vllm_ascend.worker.model_runner_v1", "NPUModelRunner"),
    ("vllm_ascend.worker.v2.model_runner", "NPUModelRunner"),
    ("vllm_ascend._310p.worker.v2.model_runner", "NPUModelRunner310V2"),
    ("vllm_ascend._310p.model_runner_310", "NPUModelRunner310"),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
)

#: Machine-readable fail-closed statement (see the module docstring).  Kept as a
#: constant so the warning, the tests and ``HOST_CONTRACT.md`` cannot drift.
FORWARD_CONSUMER_BLOCKER = (
    "no safe default-off seam consumes a merged in_proj_qkvz weight: "
    "AscendGatedDeltaNetAttention.forward computes in_proj_qkvz + in_proj_ba "
    "inline (rebound onto the core QwenGatedDeltaNetAttention), so a plugin-side "
    "single-matmul consumer would need per-call Python state inside the traced "
    "forward (dynamo-unsafe, the rejected F2 v1 shape) or a copy of the host "
    "forward body (host private source). Weight merge is therefore refused and "
    "the host model is left stock"
)

_WARNED_FAIL_CLOSED = False


# --------------------------------------------------------------------- env gate
def is_enabled() -> bool:
    """Only an explicit ``1`` opts in; unset/other values are stock mode."""
    return os.getenv(ENV_ENABLE) == "1"


def _serve_logger():
    """Logger on vLLM's configured handler tree (visible in ``vllm serve`` logs).

    ``logging.getLogger(__name__)`` propagates to an unconfigured root logger, so
    its records never reach the serve output (same F7 note as ``gate_fp32_plugin``
    / ``cascade_plugin``).  The import is lazy and best-effort so importing this
    module stays cheap and device-stack-free when the carrier is disabled.
    """
    global _SERVE_LOGGER
    if _SERVE_LOGGER is None:
        try:
            from vllm.logger import init_logger

            _SERVE_LOGGER = init_logger(
                "vllm.vllm_ascend_split_batch.gdn_merge_inproj_plugin"
            )
        except Exception:  # noqa: BLE001 -- logging must never break startup
            _SERVE_LOGGER = logger
    return _SERVE_LOGGER


# ------------------------------------------------------- forward consumer seam
def forward_consumer_available() -> bool:
    """Whether the host exposes a safe seam that consumes the merged weight.

    Always ``False`` on the hosts we know: see ``FORWARD_CONSUMER_BLOCKER`` and
    the module docstring for the two forbidden shapes it would take.  Kept as a
    function (not a constant) so the merge path below stays reachable and is
    exercised by ``tests/test_gdn_merge_inproj_plugin.py``.
    """
    return False


def _warn_fail_closed_once() -> None:
    global _WARNED_FAIL_CLOSED
    if _WARNED_FAIL_CLOSED:
        return
    _WARNED_FAIL_CLOSED = True
    _serve_logger().warning(
        "%s=1 but the GDN merge carrier is fail-closed: %s",
        ENV_ENABLE,
        FORWARD_CONSUMER_BLOCKER,
    )


# ---------------------------------------------------- load-time weight conversion
def merge_inproj_weight(qkvz_weight, ba_weight):
    """F2 layout: ``cat([in_proj_qkvz.weight, in_proj_ba.weight], dim=0)``.

    The row order is the accepted one -- the qkvz shard first, the ba shard
    appended after it -- so a single ``F.linear`` output slices as ``[..., :6144]``
    (qkvz) and ``[..., 6144:]`` (ba).  Both inputs must be 2-D, share dtype,
    device and trailing dimension.  The inputs are never mutated; the caller
    decides what to do with the result.

    Structural guards that depend on the *F2* shape (6144/32 rows, the flat
    layout, bf16) live in ``plan_merge``; this helper only enforces what makes
    the concatenation itself well-defined.
    """
    import torch

    for name, weight in (("in_proj_qkvz", qkvz_weight), ("in_proj_ba", ba_weight)):
        if not isinstance(weight, torch.Tensor):
            raise TypeError(f"{name}.weight is not a torch.Tensor")
        if weight.dim() != 2:
            raise ValueError(
                f"{name}.weight must be 2-D, got shape {tuple(weight.shape)}"
            )
    if qkvz_weight.shape[1] != ba_weight.shape[1]:
        raise ValueError(
            "in_proj_qkvz / in_proj_ba hidden dims differ: "
            f"{qkvz_weight.shape[1]} != {ba_weight.shape[1]}"
        )
    if qkvz_weight.dtype != ba_weight.dtype:
        raise ValueError(
            f"in_proj_qkvz / in_proj_ba dtypes differ: "
            f"{qkvz_weight.dtype} != {ba_weight.dtype}"
        )
    if qkvz_weight.device != ba_weight.device:
        raise ValueError(
            f"in_proj_qkvz / in_proj_ba devices differ: "
            f"{qkvz_weight.device} != {ba_weight.device}"
        )
    return torch.cat([qkvz_weight, ba_weight], dim=0)


def _layer_dims(module) -> tuple[int, int] | None:
    """The row counts the layer's *own* dims imply, or ``None`` if unavailable.

    ``qkvz = (2*key_dim + 2*value_dim) / tp_size`` and ``ba = 2*num_v_heads /
    tp_size`` -- the same arithmetic the host forward uses to slice the merged
    output (``qkv_size + z_size`` vs the ba tail).
    """
    values = []
    for attr in ("key_dim", "value_dim", "tp_size", "num_v_heads"):
        value = getattr(module, attr, None)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return None
        values.append(value)
    key_dim, value_dim, tp_size, num_v_heads = values
    if (2 * key_dim + 2 * value_dim) % tp_size or (2 * num_v_heads) % tp_size:
        return None
    return (2 * key_dim + 2 * value_dim) // tp_size, (2 * num_v_heads) // tp_size


def _host_gdn_classes() -> tuple[type, ...]:
    """The host GDN attention class, or ``()`` when the host is not importable.

    Instances in a loaded model are **core-class** objects: the Ascend methods are
    rebound onto ``QwenGatedDeltaNetAttention`` by ``patch_qwen3_5``, so the class
    check must be against that name (F2 wiring lesson 3).
    """
    try:
        import importlib

        module = importlib.import_module(_HOST_MODULE)
        cls = getattr(module, _HOST_CLASS, None)
    except Exception:  # noqa: BLE001 -- absence is a fail-open case, not a crash
        return ()
    return (cls,) if isinstance(cls, type) else ()


def is_gdn_layer(module) -> bool:
    """Whether ``module`` is the confirmed host GDN attention layer class."""
    classes = _host_gdn_classes()
    return bool(classes) and isinstance(module, classes)


def plan_merge(module) -> str | None:
    """``None`` when the layer is an F2 candidate, else the refusal reason.

    Every refusal is explicit and no rewrite happens on any of them: a
    same-named ``in_proj_qkvz``/``in_proj_ba`` pair on some other module, an
    already-merged layer, the LoRA split layout, the interleaved GQA layout, a
    non-tensor / non-materialized / non-bf16 weight, or a shape that does not
    match what the layer's own dims imply.
    """
    import torch

    if not is_gdn_layer(module):
        return "not the host GDN attention class"
    if getattr(module, _LAYER_MARK, False):
        return "already merged"
    if hasattr(module, "in_proj_qkv"):
        return "LoRA-split in_proj_qkv layout (no in_proj_qkvz contract)"
    if getattr(module, "gqa_interleaved_layout", None):
        return f"not the accepted layout ({ACCEPTED_LAYOUT})"

    qkvz = getattr(module, "in_proj_qkvz", None)
    ba = getattr(module, "in_proj_ba", None)
    if qkvz is None or ba is None:
        return "missing in_proj_qkvz / in_proj_ba"

    qkvz_weight = getattr(qkvz, "weight", None)
    ba_weight = getattr(ba, "weight", None)
    if not isinstance(qkvz_weight, torch.Tensor) or not isinstance(
        ba_weight, torch.Tensor
    ):
        return "in_proj weight is missing or is not a tensor"
    if qkvz_weight.dim() != 2 or ba_weight.dim() != 2:
        return (
            "in_proj weight is not 2-D: "
            f"{tuple(qkvz_weight.shape)} / {tuple(ba_weight.shape)}"
        )

    dims = _layer_dims(module)
    if dims is None:
        return "layer dims (key_dim/value_dim/tp_size/num_v_heads) unavailable"
    if dims != (QKVZ_OUT_ROWS, BA_OUT_ROWS):
        return f"layer dims imply {dims} rows, F2 is {(QKVZ_OUT_ROWS, BA_OUT_ROWS)}"
    if (qkvz_weight.shape[0], ba_weight.shape[0]) != (
        QKVZ_OUT_ROWS,
        BA_OUT_ROWS,
    ):
        return (
            "weight rows do not match the F2 shape: "
            f"{tuple(qkvz_weight.shape)} / {tuple(ba_weight.shape)}"
        )
    if qkvz_weight.shape[1] != ba_weight.shape[1]:
        return (
            "in_proj_qkvz / in_proj_ba hidden dims differ: "
            f"{qkvz_weight.shape[1]} != {ba_weight.shape[1]}"
        )
    if qkvz_weight.dtype != torch.bfloat16 or ba_weight.dtype != torch.bfloat16:
        return (
            f"weights are {qkvz_weight.dtype}/{ba_weight.dtype}, F2 covers the "
            f"{ACCEPTED_DTYPE} unquantized stack"
        )
    if qkvz_weight.device.type not in _ALLOWED_DEVICE_TYPES:
        return f"in_proj weights live on {qkvz_weight.device.type!r}"
    if qkvz_weight.device != ba_weight.device:
        return (
            "in_proj_qkvz / in_proj_ba live on different devices: "
            f"{qkvz_weight.device} != {ba_weight.device}"
        )
    return None


def _apply_merge(module) -> bool:
    """Replace ``in_proj_qkvz.weight`` with the merged ``[6176, 2048]`` weight."""
    import torch

    merged = merge_inproj_weight(
        module.in_proj_qkvz.weight.data, module.in_proj_ba.weight.data
    )
    module.in_proj_qkvz.weight = torch.nn.Parameter(merged, requires_grad=False)
    setattr(module, _LAYER_MARK, True)
    return True


def _process_model(model) -> tuple[int, int]:
    """Merge every F2 candidate in ``model``; returns ``(merged, skipped)``.

    Fail-closed: without a forward consumer the model is left untouched and one
    warning is logged (see ``FORWARD_CONSUMER_BLOCKER``).
    """
    if not _host_gdn_classes():
        _serve_logger().warning(
            "GDN merge carrier: host class %s.%s is not importable; skipping",
            _HOST_MODULE,
            _HOST_CLASS,
        )
        return (0, 0)
    if not forward_consumer_available():
        _warn_fail_closed_once()
        return (0, 0)

    merged = 0
    skipped = 0
    for module in model.modules():
        if not is_gdn_layer(module):
            continue
        reason = plan_merge(module)
        if reason is not None:
            skipped += 1
            logger.debug("GDN merge carrier: layer skipped (%s)", reason)
            continue
        if _apply_merge(module):
            merged += 1
    if merged or skipped:
        _serve_logger().info(
            "GDN in_proj merge: %d layer(s) merged, %d skipped", merged, skipped
        )
    return (merged, skipped)


# ----------------------------------------------------------------- install/uninstall
def _install_on_class(cls) -> bool:
    """Wrap ``cls.load_model``; idempotent, and it never unwraps another plugin."""
    original = getattr(cls, "load_model", None)
    if not callable(original) or getattr(original, _MARKER, False):
        return False

    def load_model_with_gdn_merge_inproj(self, *args, **kwargs):
        # Chain-safe: ``original`` may already be ``gate_fp32_plugin``'s wrapper.
        # Its post-load hook and ours are independent (gate fp32 caches MoE gate
        # copies; this one merges GDN in_proj weights), so either order works.
        result = original(self, *args, **kwargs)
        # Only after a *successful* load: on a raise the loader state is unknown
        # and the exception must reach the caller untouched.
        model = getattr(self, "model", None)
        if model is not None:
            try:
                _process_model(model)
            except Exception:  # noqa: BLE001 -- must never break engine startup
                _serve_logger().warning(
                    "GDN merge post-load hook failed open", exc_info=True
                )
        return result

    setattr(load_model_with_gdn_merge_inproj, _MARKER, True)
    load_model_with_gdn_merge_inproj.__name__ = getattr(
        original, "__name__", "load_model"
    )
    load_model_with_gdn_merge_inproj.__doc__ = getattr(original, "__doc__", None)
    cls.load_model = load_model_with_gdn_merge_inproj
    _INSTALLED.append((cls, original))
    return True


def install() -> bool:
    """Wrap the active model runner's ``load_model``; missing seams fail open."""
    if getattr(_LOCAL, "installing", False):
        return False
    _LOCAL.installing = True
    try:
        # Same warm-up as gate_fp32_plugin: importing the runner *directly* can
        # trip a circular import unless ``vllm_ascend.ops`` is materialized first,
        # and this plugin runs in ``WorkerBase.__init__`` (``load_general_plugins``)
        # before the worker module is resolved.  A warm-up failure must never
        # break startup.
        with contextlib.suppress(Exception):  # a warm-up must never break startup
            import vllm_ascend.ops  # noqa: F401
        installed = 0
        for module_name, class_name in _CANDIDATES:
            try:
                module = __import__(module_name, fromlist=[class_name])
                if _install_on_class(getattr(module, class_name)):
                    installed += 1
            except (ImportError, AttributeError):
                continue
        return installed > 0
    except Exception:  # noqa: BLE001 -- plugin must never break engine startup
        logger.exception("GDN merge plugin failed open; host remains usable")
        return False
    finally:
        _LOCAL.installing = False


def uninstall() -> int:
    """Undo every ``load_model`` wrap this module installed; returns the count.

    Used by the CPU tests (and by a future enable/disable flow): the host class is
    restored to the callable that was there before this carrier wrapped it, which
    is again the other plugins' wrapper when they installed first.
    """
    restored = 0
    while _INSTALLED:
        cls, original = _INSTALLED.pop()
        if getattr(getattr(cls, "load_model", None), _MARKER, False):
            cls.load_model = original
            restored += 1
    return restored


def load() -> bool:
    """``vllm.general_plugins`` entry point; default-off.

    Enabled only by ``VLLM_HUST_GDN_MERGE_INPROJ=1``.
    """
    return install() if is_enabled() else False
