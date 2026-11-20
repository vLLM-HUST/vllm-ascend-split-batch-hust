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

"""Shared state, counters and host-rewrite helpers for the zero-cost wiring.

Both wiring features (``zerocost_fi_out`` and ``zerocost_skip_cos_sin``) and the
loader entry point (``zerocost_wiring``) use the same registry, the same
rollback bookkeeping and the same counters, so they live here instead of being
imported from the entry point module (which would import them back).

Nothing here imports ``torch`` / ``torch_npu`` / ``vllm_ascend``: every host
import happens lazily inside the feature modules, which is what keeps
``load()`` inert while both gates are off.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import json
import logging
import os
import signal
import sys
import textwrap
import threading
import time

logger = logging.getLogger("vllm_ascend_split_batch.zerocost_wiring")

#: ① prefill FIA writes into the caller's output buffer (`.out` overload).
ENV_FI_OUT = "VLLM_HUST_FI_PREFILL_OUT"
#: ② defer the unread rope cos/sin materialisation.
ENV_SKIP_COS_SIN = "VLLM_HUST_SKIP_COS_SIN"
#: optional diagnostics: dump the counters as JSON on process exit.
ENV_STATS_FILE = "VLLM_HUST_ZC_STATS_FILE"
#: signal that triggers an on-demand dump to ``<ENV_STATS_FILE>.snap<N>.json``
#: (phase split).  Not configurable: a signal number that the stack does not
#: use is worth more than an extra knob.
SNAPSHOT_SIGNAL = signal.SIGUSR1

#: private kwarg the rewritten call site hands to the wrapped adaptor.
FI_OUT_KWARG = "_zc_fi_out"
#: module-global helper injected into the rewritten host module.
IDENTITY_HELPER = "_zc_identity_copy"

#: host seams (see the entry point docstring for the exact anchors).
ATTENTION_MODULE = "vllm_ascend.attention.attention_v1"
ATTENTION_CLASS = "AscendAttentionBackendImpl"
METHOD_FORWARD = "forward"
METHOD_FIA = "forward_fused_infer_attention"
FIA_SYMBOL = "npu_fused_infer_attention_score"
ADAPTOR_ATTR = "DeviceOperator"
ROTARY_MODULE = "vllm_ascend.ops.rotary_embedding"
ROTARY_UPDATE = "update_cos_sin"
ROTARY_READ = "get_cos_and_sin_slice"
#: ``vllm_ascend.device.utils.FIA_TND_LARGE_HEAD_FALLBACK_HEAD_SIZE``: the host
#: adaptor picks a different kernel there and allocates its own output.
LARGE_HEAD_FALLBACK = 512

_lock = threading.Lock()
#: re-entrancy guard: importing the host module can make vllm re-enter the
#: general-plugin loader on the *same* thread, which must not deadlock on
#: ``_lock`` nor start a second install pass.
_LOCAL = threading.local()
_installed: set = set()
_ONCE: set = set()
#: ``{key: (owner, attribute, original)}`` for :func:`uninstall`.
_orig: dict = {}
_stats: dict = {
    # ① rewrite bookkeeping
    "fi_out_calls": 0,
    "fi_out_applied": 0,
    "fi_out_writeback": 0,
    "fi_out_head_fallback": 0,
    "fi_out_fail_open": 0,
    "selfcopy_identity": 0,
    "selfcopy_copied": 0,
    # ② lazy rope bookkeeping
    "cos_sin_deferred": 0,
    "cos_sin_materialized": 0,
    "cos_sin_native": 0,
}
#: ``num_tokens -> calls`` histogram of the ① call sites (diagnostics only).
_token_hist: dict = {}
#: ``attn_state name -> calls`` histogram of the ① call sites (diagnostics only).
_state_hist: dict = {}
#: monotonic counter for :func:`snapshot` file names.
_snapshot_seq = 0
#: sentinel: "no SIGUSR1 handler was recorded yet".
_UNSET = object()
#: previous SIGUSR1 handler, recorded by :func:`_install_stats_snapshot`.
_prev_snapshot_handler = _UNSET
#: atexit hook registration guard.
_atexit_registered = False
#: ``{"Class.method": rewritten source}`` -- the ported host body, for reports
#: and tests (``inspect.getsource`` keeps showing the file original).
_rewritten: dict = {}


def rewritten_sources() -> dict:
    """The ported host bodies actually installed (diagnostics / tests)."""
    return dict(_rewritten)


def is_fi_out_enabled() -> bool:
    """True only for the exact enable token (default-off, no truthiness fuzz)."""
    return os.getenv(ENV_FI_OUT) == "1"


def is_skip_cos_sin_enabled() -> bool:
    """True only for the exact enable token (default-off, no truthiness fuzz)."""
    return os.getenv(ENV_SKIP_COS_SIN) == "1"


def stats() -> dict:
    """Snapshot of the engagement counters (report / tests)."""
    return dict(_stats)


def token_histogram() -> dict:
    """``{num_tokens: calls}`` of the ① call sites (diagnostics / report)."""
    return {str(k): _token_hist[k] for k in sorted(_token_hist)}


def state_histogram() -> dict:
    """``{attn_state: calls}`` of the ① call sites (diagnostics / report).

    ``attn_state`` is ``AscendAttentionState`` (DecodeOnly / ChunkedPrefill /
    PrefillNoCache / SpecDecoding): it separates a serving-side decode step that
    fell through to eager from a chunked-prefill step, which ``num_tokens``
    alone cannot (a small batch can be either).
    """
    return {k: _state_hist[k] for k in sorted(_state_hist)}


def _tally_tokens(query) -> None:
    """Count one ① call by its ``num_tokens`` (never breaks the host call).

    Diagnostics only: this is one dict increment per ① call, next to the
    ``fi_out_calls`` counter the wrapper already bumps, so it cannot move a
    measurement whose object is a device-side copy per layer.
    """
    try:
        key = int(query.shape[0])
    except Exception:  # noqa: BLE001 -- a non-tensor query must not raise here
        return
    _token_hist[key] = _token_hist.get(key, 0) + 1


def _tally_state(attn_metadata) -> None:
    """Count one ① call by its attention state (diagnostics, never raises)."""
    state = getattr(attn_metadata, "attn_state", None)
    name = getattr(state, "name", None) or ("none" if state is None else repr(state))
    _state_hist[name] = _state_hist.get(name, 0) + 1


def _reset_for_tests() -> None:
    """Clear counters and once-guards (does **not** unpatch)."""
    global _snapshot_seq
    _ONCE.clear()
    for key in _stats:
        _stats[key] = 0
    _token_hist.clear()
    _state_hist.clear()
    _snapshot_seq = 0


def _warn_once(message: str, *args) -> None:
    key = "warn:" + message
    if key not in _ONCE:
        _ONCE.add(key)
        logger.warning(message, *args)


def _info_once(message: str, *args) -> None:
    key = "info:" + message
    if key not in _ONCE:
        _ONCE.add(key)
        logger.info(message, *args)


def _payload() -> dict:
    """The JSON body of both the exit dump and the on-demand snapshots."""
    return {
        "pid": os.getpid(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stats": stats(),
        "fi_out_tokens": token_histogram(),
        "fi_out_states": state_histogram(),
        "installed": sorted(_installed),
    }


def _write_json(path: str, payload: dict) -> None:
    """Atomically replace ``path`` (a reader must never see a half file)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def snapshot() -> str | None:
    """Dump the counters NOW to ``<ENV_STATS_FILE>.snap<N>.json``.

    Returns the path written, or ``None`` when the env var is unset.  This is
    what splits a measurement into phases: the counters are cumulative, so a
    snapshot taken after startup and before traffic gives the startup share by
    subtraction (the exit dump is the last point).
    """
    global _snapshot_seq
    path = os.getenv(ENV_STATS_FILE)
    if not path:
        return None
    _snapshot_seq += 1
    target = f"{path}.snap{_snapshot_seq}.json"
    try:
        _write_json(target, _payload())
    except Exception:  # noqa: BLE001 -- diagnostics only
        logger.exception("zero-cost wiring: stats snapshot failed")
        return None
    logger.warning(
        "zero-cost wiring: stats snapshot #%d written to %s", _snapshot_seq, target
    )
    return target


def _snapshot_handler(_signum, _frame) -> None:
    """``SIGUSR1`` -> :func:`snapshot` (installed only when the file is set)."""
    snapshot()


def _install_stats_snapshot() -> None:
    """Wire ``SIGUSR1`` to :func:`snapshot` when ``ENV_STATS_FILE`` is set.

    Fail-open: a non-main-thread install (``signal.signal`` raises there) or a
    refused signal just logs once and leaves the process alone.
    """
    global _prev_snapshot_handler
    if not os.getenv(ENV_STATS_FILE):
        return
    if _prev_snapshot_handler is not _UNSET:
        return
    try:
        previous = signal.getsignal(SNAPSHOT_SIGNAL)
        if previous not in (signal.SIG_DFL, signal.SIG_IGN, None):
            _warn_once(
                "zero-cost wiring: replacing the existing %s handler (%r) with "
                "the stats snapshot hook",
                signal.Signals(SNAPSHOT_SIGNAL).name,
                previous,
            )
        signal.signal(SNAPSHOT_SIGNAL, _snapshot_handler)
    except (OSError, ValueError, TypeError) as exc:
        _warn_once(
            "zero-cost wiring: cannot install the %s stats snapshot hook (%s: %s); "
            "the exit dump is still available",
            signal.Signals(SNAPSHOT_SIGNAL).name,
            type(exc).__name__,
            exc,
        )
        return
    _prev_snapshot_handler = previous
    _info_once(
        "zero-cost wiring: %s now writes a stats snapshot to %s.snap<N>.json",
        signal.Signals(SNAPSHOT_SIGNAL).name,
        os.getenv(ENV_STATS_FILE),
    )


def _restore_stats_snapshot() -> None:
    """Put the previous ``SIGUSR1`` handler back (rollback / uninstall)."""
    global _prev_snapshot_handler
    if _prev_snapshot_handler is _UNSET:
        return
    with contextlib.suppress(OSError, ValueError, TypeError):
        signal.signal(SNAPSHOT_SIGNAL, _prev_snapshot_handler)
    _prev_snapshot_handler = _UNSET


def _dump_stats_on_exit() -> None:
    path = os.getenv(ENV_STATS_FILE)
    if not path:
        return
    payload = _payload()
    with contextlib.suppress(Exception):  # diagnostics only
        _write_json(path, payload)
    # One serve runs several processes that all inherit the env var, so the
    # shared path is "last writer wins" -- measured 2026-09-13: the API server's
    # all-zero dump overwrote the EngineCore's real one, and a reader could not
    # tell.  The pid-tagged copy keeps the documented path working while making
    # the writer unambiguous.
    with contextlib.suppress(Exception):  # diagnostics only
        _write_json(f"{path}.{os.getpid()}.json", payload)


# ------------------------------------------------------------ AST method port


def _unwrap_callable(obj):
    """Follow ``__func__`` / ``func`` down to the plain function object.

    The wrappers the plugins install are plain functions, but a seam could also
    be a bound method (``classmethod``) or a ``functools.partial``; normalise so
    the closure walk always reaches something with ``__code__`` /
    ``__closure__``.
    """
    seen: set = set()
    while callable(obj) and id(obj) not in seen:
        seen.add(id(obj))
        inner = getattr(obj, "__func__", None)
        if inner is None:
            inner = getattr(obj, "func", None)
        if not callable(inner) or id(inner) in seen:
            return obj
        obj = inner
    return obj


def _host_func(func, host_file: str):
    """Require exactly one host-defined function in the wrapper closure graph."""
    seen: set = set()
    stack = [_unwrap_callable(func)]
    matches: list = []
    while stack:
        current = stack.pop()
        code = getattr(current, "__code__", None)
        if code is None or id(current) in seen:
            continue
        seen.add(id(current))
        if code.co_filename == host_file:
            matches.append(current)
        for cell in getattr(current, "__closure__", None) or ():
            try:
                candidate = cell.cell_contents
            except ValueError:  # empty cell
                continue
            if callable(candidate):
                stack.append(_unwrap_callable(candidate))
    unique: list = []
    for match in matches:
        if not any(match is other for other in unique):
            unique.append(match)
    if len(unique) != 1:
        found = (
            ", ".join(sorted(getattr(m, "__qualname__", repr(m)) for m in unique))
            or "<none>"
        )
        raise RuntimeError(
            f"{getattr(func, '__qualname__', func)} does not resolve to exactly "
            f"one function defined in {host_file} (found {len(unique)}: {found})"
        )
    host = unique[0]
    if host is not _unwrap_callable(func):
        _info_once(
            "zero-cost wiring: %s is wrapped by another plugin (%s); patching "
            "the host body it delegates to.",
            getattr(func, "__qualname__", func),
            getattr(getattr(func, "__code__", None), "co_filename", "?").rsplit("/", 1)[
                -1
            ],
        )
    return host


def _rewrite_method(module, cls, name: str, transform) -> None:
    """Port ``cls.name`` through ``transform`` and install the rewritten body.

    The *host* source is re-compiled (never a copied body), so a host that
    reshapes this method fails the anchor check instead of silently running a
    stale duplicate.  The compiled code object is assigned onto the host
    function **in place** (``func.__code__ = ...``) rather than onto the class:
    every existing reference -- the class attribute, subclasses, and any other
    plugin's wrapper closure -- keeps working and sees the rewritten body.
    """
    func = _host_func(getattr(cls, name), module.__file__)
    raw = inspect.getsource(func)
    tree = ast.parse(textwrap.dedent(raw))
    fn = tree.body[0]
    if not isinstance(fn, ast.FunctionDef) or fn.name != name:
        raise RuntimeError(f"unexpected source shape for {cls.__name__}.{name}")
    transform(fn)
    ast.fix_missing_locations(tree)
    code = compile(tree, module.__file__, "exec")
    namespace: dict = {}
    exec(code, namespace, namespace)  # noqa: S102 -- host-source port
    new_code = namespace[name].__code__
    old_code = func.__code__
    if new_code.co_freevars != old_code.co_freevars:
        raise RuntimeError(
            f"{name}: rewritten body free variables {new_code.co_freevars} do "
            f"not match the host's {old_code.co_freevars}"
        )
    func.__code__ = new_code
    _orig[f"code:{cls.__name__}.{name}"] = (func, "__code__", old_code)
    _rewritten[f"{cls.__name__}.{name}"] = ast.unparse(tree)


def _import_host(module_name: str):
    """Import a host module the way the host itself does.

    ``vllm_ascend``'s submodules assume ``vllm_ascend.ops`` was imported first
    (``device_op`` -> ``vllm_ascend.ops.triton.fla.*`` -> ``vllm_ascend.ops``
    ``__init__`` -> fused MoE -> back into ``device_op``); importing
    ``attention_v1`` cold trips the host's own import cycle.  Mirror the order
    the worker uses, then import the target.
    """
    if module_name != "vllm_ascend":
        importlib.import_module("vllm_ascend.ops")
    return importlib.import_module(module_name)


def _iter_other_modules(module):
    """Yield every imported module except ``module`` itself and ``None`` slots."""
    for _name, mod in list(sys.modules.items()):
        if mod is None or mod is module:
            continue
        yield mod
