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

"""Decode FIA de-mask: drop the redundant 2048x2048 int8 causal template.

Rationale (design note ``docs/design/fia-decode-demask.md``, evidence
``profiles/qwen14b-instruct-hotspot-20260910/probe-fia/``):

``vllm-ascend``'s FULL-graph decode path
(``AscendAttentionBackendImpl.full_graph_fia`` -> ``.update_graph_params``)
calls the v1 FIA op with ``input_layout="TND"``, ``sparse_mode=3``,
``pre_tokens=next_tokens=INT_MAX`` and a hard-coded 2048x2048 int8 causal
template.  Per the CANN op doc, ``sparse_mode`` / ``pre_tokens`` /
``next_tokens`` are **no-ops when Q_S == 1** (the operator itself dispatches to
its IncreFlashAttention branch exactly at ``Q_S == 1``), and the single decode
query attends every valid kv position, so the template is inert while costing
8-23% of the op's device time.

This plugin wraps the *operator surface* (not any host private method):

* ``torch_npu.npu_fused_infer_attention_score``   (and its ``.out`` overload)
* ``torch_npu._npu_fused_infer_attention_score_get_max_workspace``

and, for calls that are provably pure decode, drops ``atten_mask`` and flips
``sparse_mode`` to 0.  Both the graph-capture call (``full_graph_fia``) and the
per-replay re-parameterization (``update_graph_params``) go through that same
``torch_npu`` attribute, so capture and replay stay consistent.  The host
itself patches this very attribute in ``vllm_ascend/batch_invariant.py``.

Default-off: with both env vars unset ``load()`` returns before importing
``torch``/``torch_npu`` or touching any module attribute -- no patch, no log,
bit-identical to stock vllm/vllm-ascend.  ``VLLM_HUST_FIA_DEMASK_TRACE=1`` is an
observe-only mode (records, never rewrites) used to capture the OFF-arm "mask
present" evidence.

Guard policy: the *decode-domain* predicate is pure duck-typed inspection of the
call kwargs and never raises out of the patched op (fail-open: unrecognised
call -> original kwargs unchanged, one counter increment).  A missing/renamed op
surface fails closed: installation is refused and the process stays stock.
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

#: master switch, default off (repo AGENTS.md: every capability env-gated).
ENV_ENABLE = "VLLM_HUST_FIA_DEMASK"
#: observe-only diagnostics (also default off; never rewrites a call).
ENV_TRACE = "VLLM_HUST_FIA_DEMASK_TRACE"

#: ``SWA_INT_MAX`` in the host (vllm_ascend/attention/attention_v1.py:72).
INT_MAX = 2147483647
#: ``str(torch.int8)``; compared as a string so the pure planner needs no torch.
INT8_DTYPE_STR = "torch.int8"

#: operator-surface symbols this plugin owns (public ``torch_npu`` attributes,
#: i.e. the CANN op surface -- *not* host private methods).
OP_FIA = "npu_fused_infer_attention_score"
OP_FIA_WORKSPACE = "_npu_fused_infer_attention_score_get_max_workspace"

# ------------------------------------------------------------------ bookkeeping

_lock = threading.Lock()
_installed = False
#: ``{op_symbol: original_object}`` for ``uninstall()`` / tests.
_orig: dict = {}
#: once-per-message log guard (a plain logging.Logger has no *_once helpers).
_ONCE_SEEN: set = set()
#: counters; ``stats()`` returns a copy.
_stats: dict = {"calls": 0, "applied": 0, "errors": 0, "reasons": {}}
#: enable decision, frozen at ``load()`` time (graph-capture safe: no per-call
#: env read).  ``active`` -> rewrite; ``observe`` -> record only.
_mode: dict = {"active": False, "observe": False, "installed": False}


def is_enabled() -> bool:
    """True only for the exact enable token (default-off, no truthiness fuzz)."""
    return os.getenv(ENV_ENABLE) == "1"


def is_trace_enabled() -> bool:
    """True only for the exact observe-only token."""
    return os.getenv(ENV_TRACE) == "1"


def _warn_once(message: str, *args) -> None:
    key = "warn:" + message
    if key not in _ONCE_SEEN:
        _ONCE_SEEN.add(key)
        logger.warning(message, *args)


def _info_once(message: str, *args) -> None:
    key = "info:" + message
    if key not in _ONCE_SEEN:
        _ONCE_SEEN.add(key)
        logger.info(message, *args)


def stats() -> dict:
    """Snapshot of the dispatch counters (tests / observability)."""
    snapshot = dict(_stats)
    snapshot["reasons"] = dict(_stats["reasons"])
    return snapshot


def _reset_for_tests() -> None:
    """Clear counters / once-guards (does not unpatch)."""
    _ONCE_SEEN.clear()
    _stats.update({"calls": 0, "applied": 0, "errors": 0})
    _stats["reasons"].clear()
    _mode.update({"active": False, "observe": False})


# --------------------------------------------------------------- pure planner


def _as_int(value, default=None):
    """Best-effort int coercion (``None`` -> ``default``); raises on garbage."""
    if value is None:
        return default
    return int(value)


def _to_list(value):
    """Normalise ``actual_seq_lengths`` (list / tuple / tensor) to a list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    try:
        return list(value)
    except TypeError:
        return None


def _mask_reason(mask) -> str:
    """Fingerprint of the host's compressed causal template (reason string)."""
    if mask is None:
        return "no-mask"
    if str(getattr(mask, "dtype", None)) != INT8_DTYPE_STR:
        return "mask-dtype"
    shape = getattr(mask, "shape", None)
    if shape is None or len(shape) != 2:
        return "mask-ndim"
    if int(shape[0]) != int(shape[1]):
        return "mask-nonsquare"
    return "ok"


def _is_pure_decode(kwargs) -> bool:
    """True when every request contributes exactly one query token.

    ``actual_seq_lengths`` is the cumulative per-request query length, so its
    length equalling the flattened token count ``T`` forces every delta to 1.
    That is exactly the operator's own ``Q_S == 1`` (IncreFlashAttention)
    domain, and it excludes prefill / chunked-prefill / mixed batches / spec
    decode (all of which keep the mask).
    """
    query = kwargs.get("query")
    seq = _to_list(kwargs.get("actual_seq_lengths"))
    if query is None or not seq or len(seq) < 1:
        return False
    shape = getattr(query, "shape", None)
    if shape is None or len(shape) < 1:
        return False
    total = int(shape[0])
    if len(seq) != total:
        return False
    return int(seq[-1]) == total


def plan_transform(kwargs) -> tuple:
    """Decide whether a FIA call is the redundant decode-mask case.

    Pure duck-typed inspection (no torch import, no device access), so it is
    CPU-testable and capture-safe.  Returns ``(new_kwargs, reason)``:
    ``new_kwargs`` is a *new* dict (input never mutated) when the rewrite
    applies, else ``None``.  Never raises: an unexpected input shape yields
    ``(None, "inspection-error")``.
    """
    try:
        if kwargs.get("input_layout") != "TND":
            return None, "layout"
        if _as_int(kwargs.get("sparse_mode"), 0) != 3:
            return None, "sparse-mode"
        if _as_int(kwargs.get("pre_tokens"), INT_MAX) != INT_MAX:
            return None, "pre-tokens"
        if _as_int(kwargs.get("next_tokens"), INT_MAX) != INT_MAX:
            return None, "next-tokens"
        reason = _mask_reason(kwargs.get("atten_mask"))
        if reason != "ok":
            return None, reason
        if not _is_pure_decode(kwargs):
            return None, "not-decode"
    except Exception:  # noqa: BLE001 -- fail-open: leave the call untouched
        return None, "inspection-error"
    new_kwargs = dict(kwargs)
    new_kwargs["atten_mask"] = None
    new_kwargs["sparse_mode"] = 0
    return new_kwargs, "applied"


def _record(reason: str) -> None:
    reasons = _stats["reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1


def _dispatch(kwargs):
    """Apply the frozen mode policy to one op call's kwargs.

    Returns the kwargs to forward (``None`` meaning "use the caller's").
    """
    _stats["calls"] += 1
    try:
        new_kwargs, reason = plan_transform(kwargs)
    except Exception as exc:  # noqa: BLE001 -- planner is defensive; belt+braces
        _stats["errors"] += 1
        _warn_once(
            "FIA de-mask inspection failed (%r); the call is forwarded "
            "unchanged for the rest of this process.",
            exc,
        )
        return None
    _record(reason)
    if reason != "applied":
        if _mode["observe"]:
            _info_once(
                "FIA observer: non-matching call (reason=%s, layout=%s, "
                "sparse_mode=%s, mask=%s)",
                reason,
                kwargs.get("input_layout"),
                kwargs.get("sparse_mode"),
                getattr(kwargs.get("atten_mask"), "shape", None),
            )
        return None
    if not _mode["active"]:
        # observe-only: this IS the OFF-arm evidence (mask present, mode 3).
        # Logged at WARNING on purpose: vLLM pins the root logger to WARNING,
        # so an INFO line would never reach the serve log.
        _warn_once(
            "FIA observer: decode FIA call carries the redundant mask "
            "(atten_mask=%s int8, sparse_mode=3, pre/next=INT_MAX, TND, "
            "Q_S=1); VLLM_HUST_FIA_DEMASK is OFF so the call is unchanged.",
            getattr(kwargs.get("atten_mask"), "shape", None),
        )
        return None
    _stats["applied"] += 1
    _warn_once(
        "FIA decode de-mask applied: dropped %s int8 atten_mask and set "
        "sparse_mode 3->0 on a pure-decode FIA call (TND, Q_S=1 per request, "
        "pre/next=INT_MAX).",
        getattr(kwargs.get("atten_mask"), "shape", None),
    )
    return new_kwargs


# ------------------------------------------------------------- op surface proxy


class _OpProxy:
    """Transparent proxy for a ``torch_npu`` op symbol (packet or function).

    Attribute access is forwarded to the wrapped object, so callers that touch
    ``.op`` / ``.overloads`` / ``__doc__`` keep working; ``.out`` gets its own
    interception point for the out-overload used by the host.
    """

    def __init__(self, op) -> None:
        object.__setattr__(self, "_wrapped_op", op)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_wrapped_op"), name)

    def __call__(self, *args, **kwargs):
        return _forward_op(object.__getattribute__(self, "_wrapped_op"), args, kwargs)

    def out(self, *args, **kwargs):
        target = object.__getattribute__(self, "_wrapped_op").out
        return _forward_op(target, args, kwargs)


def _forward_op(target, args, kwargs):
    """Call ``target`` with the mode-transformed kwargs (fail-open)."""
    try:
        new_kwargs = _dispatch(kwargs)
    except Exception as exc:  # noqa: BLE001 -- must never break the op call
        _stats["errors"] += 1
        _warn_once(
            "FIA de-mask dispatch failed (%r); forwarding the call unchanged "
            "for the rest of this process.",
            exc,
        )
        new_kwargs = None
    if new_kwargs is None:
        return target(*args, **kwargs)
    return target(*args, **new_kwargs)


# ---------------------------------------------------------------- install/load


def install() -> bool:
    """Patch the FIA v1 op surface (idempotent, fail-closed).

    Returns True when the proxies are in place.  A missing / renamed op surface
    refuses installation with one warning and leaves the process stock; this
    method itself never raises into the plugin loader.
    """
    global _installed
    if _installed:
        return True
    with _lock:
        if _installed:
            return True
        try:
            import torch_npu
        except Exception as exc:  # noqa: BLE001 -- load() must never raise
            _warn_once(
                "FIA de-mask requested but torch_npu could not be imported "
                "(%r); serving stays on the stock FIA call surface.",
                exc,
            )
            return False

        op = getattr(torch_npu, OP_FIA, None)
        if op is None or not callable(op) or not hasattr(op, "out"):
            # Signature-drift fail-closed: the expected v1 op (packet with an
            # ``.out`` overload) is not on this build.  Never guess.
            _warn_once(
                "FIA de-mask refused: torch_npu.%s is missing or has no .out "
                "overload (found=%r); the process stays on the stock FIA call "
                "surface.",
                OP_FIA,
                type(op).__name__,
            )
            return False

        _orig[OP_FIA] = op
        setattr(torch_npu, OP_FIA, _OpProxy(op))
        patched = [OP_FIA]

        workspace = getattr(torch_npu, OP_FIA_WORKSPACE, None)
        if workspace is not None and callable(workspace):
            # Same guard, same rewrite: the workspace must be sized for the
            # parameters actually handed to the op.
            _orig[OP_FIA_WORKSPACE] = workspace
            setattr(torch_npu, OP_FIA_WORKSPACE, _OpProxy(workspace))
            patched.append(OP_FIA_WORKSPACE)
        else:
            _warn_once(
                "FIA de-mask: torch_npu.%s not found; the workspace query is "
                "left untouched (sizes are an upper bound, so the op still "
                "runs).",
                OP_FIA_WORKSPACE,
            )

        _installed = True
        _mode["installed"] = True
        if _mode["active"]:
            logger.warning(
                "FIA decode de-mask is ACTIVE (VLLM_HUST_FIA_DEMASK=1): pure "
                "decode FIA calls (%s) now run without the redundant "
                "2048x2048 int8 causal template (sparse_mode 3->0); prefill / "
                "mixed / sliding-window / non-TND calls are untouched.",
                ", ".join(patched),
            )
        else:
            logger.warning(
                "FIA decode de-mask observer is ACTIVE "
                "(VLLM_HUST_FIA_DEMASK_TRACE=1, VLLM_HUST_FIA_DEMASK not 1): "
                "FIA calls are recorded but never modified (%s).",
                ", ".join(patched),
            )
        return True


def uninstall() -> None:
    """Restore the original op surface (tests / diagnostics)."""
    global _installed
    with _lock:
        if not _orig:
            _installed = False
            return
        try:
            import torch_npu

            for symbol, original in _orig.items():
                setattr(torch_npu, symbol, original)
        except Exception:  # noqa: BLE001 -- best-effort restore
            logger.exception("FIA de-mask uninstall failed")
        _orig.clear()
        _installed = False
        _mode["installed"] = False


def load() -> bool:
    """``vllm.general_plugins`` entry point (default-off).

    With both env vars unset this returns before importing torch_npu or
    touching anything: no patch, no log line.
    """
    active = is_enabled()
    observe = is_trace_enabled()
    if not active and not observe:
        # Default-off: no torch_npu import, no patch, no log output.
        return False
    _mode["active"] = active
    _mode["observe"] = observe or active
    return install()
