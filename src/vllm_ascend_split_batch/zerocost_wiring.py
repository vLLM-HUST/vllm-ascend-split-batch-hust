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

"""Zero-new-kernel *host wiring* items (default-off, per-item env gates).

Two wiring-only candidates from
``profiles/qwen14b-instruct-hotspot-20260910/probe-copy-materialize/REPORT.md``
(X1) that need **no new kernel and change no numerics** -- they only stop the
host from moving data it does not have to move.

① ``VLLM_HUST_FI_PREFILL_OUT`` -- prefill FIA writes into the caller's buffer.

   The decode FULL-graph leg has always called the *out* overload
   (``attention_v1.full_graph_fia`` -> ``torch_npu.npu_fused_infer_attention_score.out``
   with ``out=[output, softmax_lse]``; mandatory there because ACL-graph replay
   cannot re-run the copy that sits outside the captured task group), while the
   eager leg calls the *functional* form and then copies the result into
   ``output``:

   * ``AscendAttentionBackendImpl.forward_fused_infer_attention`` --
     ``attn_output, _ = DeviceOperator.npu_fused_infer_attention_score(...)``
     followed by ``output[:num_tokens] = attn_output[:num_tokens]``: one real
     device task per layer per prefill pass
     (``aclnnInplaceCopy_TensorMoveAiCore_TensorMove``, 2x/layer/pass measured).
   * ``AscendAttentionBackendImpl.forward`` -- the same statement again, but
     every branch of ``forward_impl`` returns the very ``output`` object it was
     handed, so this second copy is a **pure self-copy**
     (``output[:n] = output[:n]``), i.e. the other half of the 2x/layer.

   This carrier rewrites those two methods (AST rewrite of the *host* body, so
   the carrier cannot drift into a silently duplicated copy of the method) plus
   the resolved device adaptor's FIA entry point:

   * the ``DeviceOperator.npu_fused_infer_attention_score`` call in
     ``forward_fused_infer_attention`` gains a private ``_zc_fi_out=`` kwarg
     (the destination region); the wrapped adaptor routes it to the op's ``out``
     overload and falls back to the natively-allocated output whenever that is
     unsafe (head-size fallback, out-overload shape/resize refusal, any error);
   * both ``output[...] = attn_output[...]`` statements are guarded by
     ``_zc_identity_copy(...)``, an exact "src and dst are the same bytes"
     predicate, so a copy that cannot change memory is skipped and every other
     copy still happens.

② ``VLLM_HUST_SKIP_COS_SIN`` -- do not materialise the rope cos/sin buffers
   that nothing reads.

   ``vllm_ascend/ops/rotary_embedding.update_cos_sin`` (called every step from
   ``worker/model_runner_v1.py``) is pure materialisation work -- measured as
   2x ``aclnnIndexSelect_GatherV3`` + 2x ``aclnnRepeat_Tile`` + 2x
   ``aclnnInplaceCopy_Slice`` per step -- whose only consumers are
   ``vllm_ascend/_310p/ops/rotary_embedding.py`` (310P) and
   ``vllm_ascend/patch/worker/patch_minimax_m2.py`` (MiniMax-M2).  On
   A2 + Qwen2.5 + the v1 runner nothing reads the buffers.

   Rather than deleting the call the carrier makes it *lazy*: the wrapper only
   records the positions, and ``get_cos_and_sin_slice`` materialises on demand
   before handing the buffers out.  A reader therefore still sees exactly the
   buffers it would have seen; it just does not pay for them when nobody reads.
   The reader list (:data:`ROPE_READER_MODULES`) is documentation of *why* the
   deferral is sound, not a gate: the host imports every patch module
   unconditionally, so module presence says nothing about whether this
   chip/model reads the buffers.  See :func:`_install_skip_cos_sin`.

Default-off contract: with both env vars unset :func:`load` returns before
importing ``torch`` / ``torch_npu`` / ``vllm_ascend``, touches nothing and logs
nothing, so the process is bit-identical to stock vllm/vllm-ascend (asserted in
``tests/test_zerocost_wiring.py``).

Fail-open contract: every rewrite is refused (with ONE warning) when an anchor
is missing or ambiguous, and the wrapped adaptor delegates to the original
whenever the out-overload path is not provably safe.  Nothing is ever raised
into the plugin loader or into a worker; a half-installed feature is rolled
back.

Diagnostics (off unless ``VLLM_HUST_ZC_STATS_FILE`` is set, and only useful
with a feature enabled): the counters are dumped as JSON at process exit -- to
the configured path *and* to a ``<path>.<pid>.json`` copy, because every
process of one serve inherits that env var and the shared path is last-writer-
wins (the API server's all-zero dump was measured to overwrite the engine's) --
and :func:`snapshot` dumps them *on demand* to ``<path>.snap<N>.json`` so a
measurement can split the totals by phase (before traffic / after traffic /
final).  ``SIGUSR1`` is wired to :func:`snapshot` for that purpose -- no other
part of vllm / vllm-ascend / torch_npu installs a SIGUSR1 handler (checked
2026-09-13), and the previous handler is recorded and restored by
:func:`uninstall`.  Alongside the counters the payload carries
``fi_out_tokens`` / ``fi_out_states``: histograms of the ``num_tokens`` and the
``attn_state`` seen at every ① call, so a phase can be attributed to a shape (a
capture dummy of 32/64/128 vs a real chunked-prefill chunk of ~2048) and to a
phase of inference (chunked prefill vs decode) instead of being guessed.
"""

from __future__ import annotations

import ast
import atexit
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

logger = logging.getLogger(__name__)

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

#: host seams (see the module docstring for the exact anchors).
ATTENTION_MODULE = "vllm_ascend.attention.attention_v1"
ATTENTION_CLASS = "AscendAttentionBackendImpl"
METHOD_FORWARD = "forward"
METHOD_FIA = "forward_fused_infer_attention"
FIA_SYMBOL = "npu_fused_infer_attention_score"
ADAPTOR_ATTR = "DeviceOperator"
ROTARY_MODULE = "vllm_ascend.ops.rotary_embedding"
ROTARY_UPDATE = "update_cos_sin"
ROTARY_READ = "get_cos_and_sin_slice"
#: modules that *do* read the rope buffers; documented here because ② is only
#: sound while their read pattern is "update then read in the same step" (they
#: are not a gate -- see :func:`_install_skip_cos_sin`).
ROPE_READER_MODULES = (
    "vllm_ascend._310p.ops.rotary_embedding",
    "vllm_ascend.patch.worker.patch_minimax_m2",
)
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
#: last positions handed to the lazy rope wrapper (per process).
_cos_sin_positions = None
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


# ------------------------------------------------------------------ utilities


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


def _identity_copy(attn_output, output, num_tokens) -> bool:
    """True when ``output[:n] = attn_output[:n]`` cannot change any byte.

    Exact, dtype/stride/address-level comparison of the two regions -- the only
    case in which skipping the copy is provably a no-op.  Never raises: any
    surprise means "not identity", i.e. the copy runs as it always did.
    """
    try:
        if attn_output is output:
            return True
        count = int(num_tokens)
        if count <= 0:
            return True
        src = attn_output[:count]
        dst = output[:count]
        return bool(
            src.shape == dst.shape
            and src.stride() == dst.stride()
            and src.data_ptr() == dst.data_ptr()
            and src.dtype == dst.dtype
            and src.device == dst.device
        )
    except Exception:  # noqa: BLE001 -- fail-open: keep the copy
        return False


# ------------------------------------------------------------ AST method port


def _is_tail_copy(node: ast.stmt) -> bool:
    """``output[:num_tokens] = attn_output[:num_tokens]`` (the guarded seam)."""
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
        return False
    if target.value.id != "output":
        return False
    value = node.value
    if not (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)):
        return False
    return value.value.id == "attn_output"


def _guard_tail_copy(fn: ast.FunctionDef) -> None:
    """Wrap the trailing ``output[...] = attn_output[...]`` in the identity guard."""
    hits = [i for i, stmt in enumerate(fn.body) if _is_tail_copy(stmt)]
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one trailing `output[...] = attn_output[...]` "
            f"assignment in {fn.name}, found {len(hits)}"
        )
    stmt = fn.body[hits[0]]
    guard = ast.If(
        test=ast.UnaryOp(
            op=ast.Not(),
            operand=ast.Call(
                func=ast.Name(id=IDENTITY_HELPER, ctx=ast.Load()),
                args=[
                    ast.Name(id="attn_output", ctx=ast.Load()),
                    ast.Name(id="output", ctx=ast.Load()),
                    ast.Name(id="num_tokens", ctx=ast.Load()),
                ],
                keywords=[],
            ),
        ),
        body=[stmt],
        orelse=[],
    )
    ast.copy_location(guard, stmt)
    fn.body[hits[0]] = guard


def _inject_fi_out(fn: ast.FunctionDef) -> None:
    """Hand the destination region to the adaptor FIA call."""
    hits = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == FIA_SYMBOL
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == ADAPTOR_ATTR
        ):
            hits.append(node)
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one {ADAPTOR_ATTR}.{FIA_SYMBOL} call in {fn.name}, "
            f"found {len(hits)}"
        )
    call = hits[0]
    if any(kw.arg == FI_OUT_KWARG for kw in call.keywords):
        return
    call.keywords.append(
        ast.keyword(
            arg=FI_OUT_KWARG,
            value=ast.Subscript(
                value=ast.Name(id="output", ctx=ast.Load()),
                slice=ast.Slice(upper=ast.Name(id="num_tokens", ctx=ast.Load())),
                ctx=ast.Load(),
            ),
        )
    )


def _transform_fia_method(fn: ast.FunctionDef) -> None:
    """``forward_fused_infer_attention``: hand the destination over + guard the copy."""
    _inject_fi_out(fn)
    _guard_tail_copy(fn)


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
    """Resolve the *host-defined* function behind any number of plugins.

    Other plugins in this bundle wrap the very same host methods: the eager
    ``cascade_plugin._make_forward_wrapper`` puts a pass-through around the host
    method, and with ``VLLM_ASCEND_ENABLE_CASCADE_GRAPH=1``
    ``cascade_graph_plugin.install`` stacks a *second* wrapper on top of it, so
    the class attribute sits two plugin wrappers deep (the outer wrapper's
    closure points at the inner wrapper, not at the host file).
    ``inspect.getsource`` on the class attribute would return the outermost
    wrapper's source.

    The criterion is therefore "the code object's file is the host file", not
    "one hop": walk the whole ``__closure__`` graph and require **exactly one**
    reachable function defined in the host file.  Zero candidates (a moved
    anchor / a broken chain) and several candidates (an ambiguous chain) both
    raise, which the caller turns into the documented fail-open rollback -- the
    host body is never patched on a guess.
    """
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


# ------------------------------------------------------- adaptor FIA wrapper


def _fi_out_call(orig, query, key, value, attn_metadata, args, kwargs, dest):
    """Route one FIA call to the ``.out`` overload, writing into ``dest``.

    Falls back to the host's functional call (whose result the caller's identity
    guard then copies into ``dest`` as before) whenever the out path is not
    provably safe.  ``orig`` is the host's *bound* classmethod (cls already
    bound), so it is called without the class argument.
    """
    import torch  # noqa: PLC0415 -- only reachable with the feature on
    import torch_npu  # noqa: PLC0415

    call = dict(kwargs)
    if call.get("head_size") == LARGE_HEAD_FALLBACK:
        # The host opts into its large-head prefill kernel there and allocates
        # its own output; the caller's guard copies it back.
        _stats["fi_out_head_fallback"] += 1
        return orig(query, key, value, attn_metadata, *args, **kwargs)

    num_key_value_heads = call.pop("num_key_value_heads")
    num_heads = call.pop("num_heads")
    scale = call.pop("scale")
    for consumed in (
        "key_cache",
        "value_cache",
        "current_key",
        "current_value",
        "head_size",
        "is_prefill_no_cache",
    ):
        call.pop(consumed, None)

    lse = torch.empty(1, dtype=query.dtype, device=query.device)
    result = torch_npu.npu_fused_infer_attention_score.out(
        query=query,
        key=key.contiguous(),
        value=value.contiguous(),
        num_key_value_heads=num_key_value_heads,
        num_heads=num_heads,
        scale=scale,
        out=[dest, lse],
        **call,
    )
    first = result[0] if isinstance(result, (tuple, list)) and result else result
    if first is not dest and (
        first.data_ptr() != dest.data_ptr() or tuple(first.shape) != tuple(dest.shape)
    ):
        # The op did not land in the caller's region (e.g. it re-allocated):
        # keep native semantics by writing the result back explicitly.
        dest.copy_(first)
        _stats["fi_out_writeback"] += 1
    else:
        _stats["fi_out_applied"] += 1
        _info_once(
            "zero-cost wiring ①: prefill FIA now writes into the caller's "
            "output buffer through the .out overload (no TensorMove copy)."
        )
    return dest, lse


def _make_fia_wrapper(orig):
    """Signature-agnostic wrapper for the adaptor's FIA entry point."""

    def _npu_fused_infer_attention_score(
        cls, query, key, value, attn_metadata, *args, **kwargs
    ):
        dest = kwargs.pop(FI_OUT_KWARG, None)
        if dest is None:
            return orig(query, key, value, attn_metadata, *args, **kwargs)
        _stats["fi_out_calls"] += 1
        _tally_tokens(query)
        _tally_state(attn_metadata)
        try:
            return _fi_out_call(
                orig, query, key, value, attn_metadata, args, kwargs, dest
            )
        except Exception as exc:  # noqa: BLE001 -- never break the host call
            _stats["fi_out_fail_open"] += 1
            _warn_once(
                "zero-cost wiring ①: FIA .out forwarding unavailable (%s: %s); "
                "this process keeps the host's functional FIA call plus its "
                "copy (fail-open).",
                type(exc).__name__,
                exc,
            )
            return orig(query, key, value, attn_metadata, *args, **kwargs)

    return _npu_fused_infer_attention_score


# ---------------------------------------------------------------- installations


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


def _install_fi_out() -> None:
    module = _import_host(ATTENTION_MODULE)
    cls = getattr(module, ATTENTION_CLASS)
    adaptor = getattr(module, ADAPTOR_ATTR)
    if not hasattr(adaptor, FIA_SYMBOL):
        raise RuntimeError(
            f"{ADAPTOR_ATTR}.{FIA_SYMBOL} is missing on host adaptor "
            f"{adaptor.__name__!r}"
        )

    # the rewritten body calls this helper; inject it into the host module so
    # the ported source stays byte-equal to the host body plus the guard.
    if getattr(module, IDENTITY_HELPER, None) is None:
        module.__dict__[IDENTITY_HELPER] = _identity_helper
        _orig[f"global:{ATTENTION_MODULE}.{IDENTITY_HELPER}"] = (
            module,
            IDENTITY_HELPER,
            None,
        )

    original = getattr(adaptor, FIA_SYMBOL)
    setattr(adaptor, FIA_SYMBOL, classmethod(_make_fia_wrapper(original)))
    _orig[f"method:{adaptor.__name__}.{FIA_SYMBOL}"] = (
        adaptor,
        FIA_SYMBOL,
        adaptor.__dict__.get(FIA_SYMBOL),
    )

    _rewrite_method(module, cls, METHOD_FIA, _transform_fia_method)
    _rewrite_method(module, cls, METHOD_FORWARD, _guard_tail_copy)


def _identity_helper(attn_output, output, num_tokens) -> bool:  # pragma: no cover
    """Host-module shim: counts then defers to :func:`_identity_copy`."""
    same = _identity_copy(attn_output, output, num_tokens)
    if same:
        _stats["selfcopy_identity"] += 1
    else:
        _stats["selfcopy_copied"] += 1
    return same


def _install_skip_cos_sin() -> None:
    module = _import_host(ROTARY_MODULE)
    update = getattr(module, ROTARY_UPDATE)
    reader = getattr(module, ROTARY_READ)

    def _lazy_update_cos_sin(positions):
        """Record the positions; materialise only if something reads them.

        Deferral is *exactly* equivalent for the known readers
        (:data:`ROPE_READER_MODULES`): both call ``update_cos_sin(positions)``
        and read ``get_cos_and_sin_slice()`` in the same step, so materialising
        at read time produces the same buffers.  Presence of a reader *module*
        is deliberately NOT used as a gate -- the host imports every patch
        module unconditionally (``vllm_ascend/patch/worker/__init__.py``), so
        module presence says nothing about whether this chip/model reads the
        buffers, and gating on it silently disabled this feature on A2
        (measured: 0 engagement, kernel counts unchanged).
        """
        global _cos_sin_positions
        _cos_sin_positions = positions
        _stats["cos_sin_deferred"] += 1
        return None

    def _materialising_read():
        """Hand out the buffers, materialising them first when deferred."""
        global _cos_sin_positions
        if _cos_sin_positions is not None:
            positions, _cos_sin_positions = _cos_sin_positions, None
            _stats["cos_sin_materialized"] += 1
            update(positions)
        return reader()

    module.__dict__[ROTARY_UPDATE] = _lazy_update_cos_sin
    module.__dict__[ROTARY_READ] = _materialising_read
    _orig[f"global:{ROTARY_MODULE}.{ROTARY_UPDATE}"] = (module, ROTARY_UPDATE, update)
    _orig[f"global:{ROTARY_MODULE}.{ROTARY_READ}"] = (module, ROTARY_READ, reader)

    # rebind every module that already imported the symbols by name
    for name, mod in list(sys.modules.items()):
        if mod is None or mod is module:
            continue
        if getattr(mod, ROTARY_UPDATE, None) is update:
            setattr(mod, ROTARY_UPDATE, _lazy_update_cos_sin)
            _orig[f"global:{name}.{ROTARY_UPDATE}"] = (mod, ROTARY_UPDATE, update)
        if getattr(mod, ROTARY_READ, None) is reader:
            setattr(mod, ROTARY_READ, _materialising_read)
            _orig[f"global:{name}.{ROTARY_READ}"] = (mod, ROTARY_READ, reader)


def install() -> set:
    """Install every requested feature (idempotent, fail-open per feature).

    Returns the set of features that ended up installed.  A failing feature is
    rolled back to the host implementation and reported with ONE warning; the
    other feature is unaffected.
    """
    if getattr(_LOCAL, "installing", False):
        # Re-entered from the vllm general-plugin loader while the host module
        # import was still in flight: the outer pass owns the installation.
        return set(_installed)
    with _lock:
        wanted = set()
        if is_fi_out_enabled():
            wanted.add(ENV_FI_OUT)
        if is_skip_cos_sin_enabled():
            wanted.add(ENV_SKIP_COS_SIN)
        if not wanted:
            return set()
        if wanted <= _installed:
            return set(_installed)
        global _atexit_registered
        if not _atexit_registered:
            atexit.register(_dump_stats_on_exit)
            _atexit_registered = True
        # Optional diagnostics (no-op unless ENV_STATS_FILE is set): phase
        # snapshots on demand.  Installed here, next to the exit dump, so the
        # rollback below takes it down with the features it measures.
        _install_stats_snapshot()

        _LOCAL.installing = True
        try:
            if ENV_FI_OUT in wanted and ENV_FI_OUT not in _installed:
                try:
                    _install_fi_out()
                    _installed.add(ENV_FI_OUT)
                    logger.warning(
                        "zero-cost wiring ① is ACTIVE (%s=1): the eager prefill "
                        "FIA leg now writes into the caller's output buffer "
                        "through the op's .out overload and the two redundant "
                        "TensorMove copies per layer are guarded by an exact "
                        "identity predicate.",
                        ENV_FI_OUT,
                    )
                except Exception:
                    logger.exception(
                        "zero-cost wiring ① refused (%s=1): a host anchor moved; "
                        "the process stays on the stock FIA call + copy path",
                        ENV_FI_OUT,
                    )
                    _uninstall_locked()

            if ENV_SKIP_COS_SIN in wanted and ENV_SKIP_COS_SIN not in _installed:
                try:
                    _install_skip_cos_sin()
                    _installed.add(ENV_SKIP_COS_SIN)
                    logger.warning(
                        "zero-cost wiring ② is ACTIVE (%s=1): update_cos_sin now "
                        "records positions only and get_cos_and_sin_slice() "
                        "materialises the rope buffers on demand.",
                        ENV_SKIP_COS_SIN,
                    )
                except Exception:
                    logger.exception(
                        "zero-cost wiring ② refused (%s=1): a host anchor moved; "
                        "the process stays on the stock cos/sin materialisation",
                        ENV_SKIP_COS_SIN,
                    )
                    _uninstall_locked()
        finally:
            _LOCAL.installing = False
        return set(_installed)


def uninstall() -> None:
    """Restore every patched seam (tests / rollback)."""
    with _lock:
        _uninstall_locked()


def _uninstall_locked() -> None:
    """Rollback body; the caller holds ``_lock`` (``install`` may call it)."""
    if True:  # keep the body at one indentation level for the locking split
        # modules that imported the patched symbols *after* installation hold
        # our wrappers in their own namespace without a recorded original; undo
        # those first, while the originals are still in ``_orig``.
        _restore_late_imports()
        _restore_stats_snapshot()
        for key, (owner, attribute, original) in list(_orig.items()):
            try:
                if original is None:
                    owner.__dict__.pop(attribute, None)
                else:
                    setattr(owner, attribute, original)
            except Exception:  # noqa: BLE001 -- best-effort restore
                logger.exception("zero-cost wiring: could not restore %s", key)
            _orig.pop(key, None)
        _installed.clear()


def _restore_late_imports() -> None:
    """Undo rebinds made by modules imported after :func:`install`."""
    module = sys.modules.get(ROTARY_MODULE)
    if module is None:
        return
    update = _orig.get(f"global:{ROTARY_MODULE}.{ROTARY_UPDATE}")
    reader = _orig.get(f"global:{ROTARY_MODULE}.{ROTARY_READ}")
    lazy = module.__dict__.get(ROTARY_UPDATE)
    materialise = module.__dict__.get(ROTARY_READ)
    for _name, mod in list(sys.modules.items()):
        if mod is None or mod is module:
            continue
        if update is not None and getattr(mod, ROTARY_UPDATE, None) is lazy:
            setattr(mod, ROTARY_UPDATE, update[2])
        if reader is not None and getattr(mod, ROTARY_READ, None) is materialise:
            setattr(mod, ROTARY_READ, reader[2])


def load() -> bool:
    """``vllm.general_plugins`` entry point (default-off).

    With both env vars unset this returns before importing torch / torch_npu /
    vllm_ascend and touches nothing.
    """
    if not is_fi_out_enabled() and not is_skip_cos_sin_enabled():
        return False
    if getattr(_LOCAL, "installing", False):
        return False
    try:
        return bool(install())
    except Exception:  # noqa: BLE001 -- load() must never raise
        logger.exception("zero-cost wiring: installation failed (stays stock)")
        return False
