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
Implemented in :mod:`vllm_ascend_split_batch.zerocost_fi_out`.

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
   that nothing reads.  Implemented in
   :mod:`vllm_ascend_split_batch.zerocost_skip_cos_sin`, which carries the
   reader list and the deferral argument.

Default-off contract: with both env vars unset :func:`load` returns before
importing ``torch`` / ``torch_npu`` / ``vllm_ascend``, touches nothing and logs
nothing, so the process is bit-identical to stock vllm/vllm-ascend (asserted in
``tests/test_zerocost_wiring.py``).

Fail-open contract: every rewrite is refused (with ONE warning) when an anchor
is missing or ambiguous, and the wrapped adaptor delegates to the original
whenever the out-overload path is not provably safe.  Nothing is ever raised
into the plugin loader or into a worker.

Rollback granularity is **the whole carrier, not one feature**: the two features
share one recovery registry (``zerocost_state._orig``), so if either install
raises, :func:`_uninstall_locked` restores every recorded seam and clears all
installation state -- the feature that did install is taken down together with
the one that failed, and the process goes back to the stock behaviour of both.
There is no per-feature fail-open: it is fail-open for the carrier as a whole.

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

Layout: this module holds only the gate ladder, install/uninstall bookkeeping
and the loader entry point.  Shared counters and host-rewrite helpers live in
:mod:`vllm_ascend_split_batch.zerocost_state`; the two features live in their
own modules.
"""

from __future__ import annotations

import atexit
import json  # noqa: F401 -- re-exported: tests patch ``zerocost_wiring.json``

from . import zerocost_state as st
from .zerocost_fi_out import install_fi_out
from .zerocost_skip_cos_sin import install_skip_cos_sin, restore_late_imports
from .zerocost_state import (  # noqa: F401 -- re-exported public surface
    ENV_FI_OUT,
    ENV_SKIP_COS_SIN,
    ENV_STATS_FILE,
    _installed,
    _orig,
    _reset_for_tests,
    is_fi_out_enabled,
    is_skip_cos_sin_enabled,
    rewritten_sources,
    snapshot,
    state_histogram,
    stats,
    token_histogram,
)

logger = st.logger


def install() -> set:
    """Install every requested feature (idempotent, all-or-nothing rollback).

    Returns the set of features that ended up installed.  Any failure is
    reported with ONE warning and triggers :func:`_uninstall_locked`, which
    restores *every* seam recorded in the shared registry -- including the seams
    of a feature that had already installed successfully.  So ``install()`` is
    fail-open for the carrier as a whole, not per feature: the return value is
    either every requested feature or nothing.
    """
    if getattr(st._LOCAL, "installing", False):
        # Re-entered from the vllm general-plugin loader while the host module
        # import was still in flight: the outer pass owns the installation.
        return set(st._installed)
    with st._lock:
        wanted = set()
        if st.is_fi_out_enabled():
            wanted.add(st.ENV_FI_OUT)
        if st.is_skip_cos_sin_enabled():
            wanted.add(st.ENV_SKIP_COS_SIN)
        if not wanted:
            return set()
        if wanted <= st._installed:
            return set(st._installed)
        if not st._atexit_registered:
            atexit.register(st._dump_stats_on_exit)
            st._atexit_registered = True
        # Optional diagnostics (no-op unless ENV_STATS_FILE is set): phase
        # snapshots on demand.  Installed here, next to the exit dump, so the
        # rollback below takes it down with the features it measures.
        st._install_stats_snapshot()

        st._LOCAL.installing = True
        try:
            if st.ENV_FI_OUT in wanted and st.ENV_FI_OUT not in st._installed:
                try:
                    install_fi_out()
                    st._installed.add(st.ENV_FI_OUT)
                    logger.warning(
                        "zero-cost wiring ① is ACTIVE (%s=1): the eager prefill "
                        "FIA leg now writes into the caller's output buffer "
                        "through the op's .out overload and the two redundant "
                        "TensorMove copies per layer are guarded by an exact "
                        "identity predicate.",
                        st.ENV_FI_OUT,
                    )
                except Exception:
                    logger.exception(
                        "zero-cost wiring ① refused (%s=1): a host anchor moved; "
                        "the process stays on the stock FIA call + copy path",
                        st.ENV_FI_OUT,
                    )
                    _uninstall_locked()

            if (
                st.ENV_SKIP_COS_SIN in wanted
                and st.ENV_SKIP_COS_SIN not in st._installed
            ):
                try:
                    install_skip_cos_sin()
                    st._installed.add(st.ENV_SKIP_COS_SIN)
                    logger.warning(
                        "zero-cost wiring ② is ACTIVE (%s=1): update_cos_sin now "
                        "records positions only and get_cos_and_sin_slice() "
                        "materialises the rope buffers on demand.",
                        st.ENV_SKIP_COS_SIN,
                    )
                except Exception:
                    logger.exception(
                        "zero-cost wiring ② refused (%s=1): a host anchor moved; "
                        "the process stays on the stock cos/sin materialisation",
                        st.ENV_SKIP_COS_SIN,
                    )
                    _uninstall_locked()
        finally:
            st._LOCAL.installing = False
        return set(st._installed)


def uninstall() -> None:
    """Restore every patched seam (tests / rollback)."""
    with st._lock:
        _uninstall_locked()


def _uninstall_locked() -> None:
    """Rollback body; the caller holds ``st._lock`` (``install`` may call it)."""
    # modules that imported the patched symbols *after* installation hold our
    # wrappers in their own namespace without a recorded original; undo those
    # first, while the originals are still in ``_orig``.
    restore_late_imports()
    st._restore_stats_snapshot()
    for key, (owner, attribute, original) in list(st._orig.items()):
        try:
            if original is None:
                # Injected attribute: drop our override so the inherited (or
                # absent) original resolves again.  ``delattr`` is the only form
                # that works for classes too -- ``owner.__dict__`` is a
                # mappingproxy there, not a mutable dict.
                if attribute in getattr(owner, "__dict__", {}):
                    delattr(owner, attribute)
            else:
                setattr(owner, attribute, original)
        except Exception:  # noqa: BLE001 -- best-effort restore
            logger.exception("zero-cost wiring: could not restore %s", key)
        st._orig.pop(key, None)
    st._installed.clear()


def load() -> bool:
    """``vllm.general_plugins`` entry point (default-off).

    With both env vars unset this returns before importing torch / torch_npu /
    vllm_ascend and touches nothing.
    """
    if not st.is_fi_out_enabled() and not st.is_skip_cos_sin_enabled():
        return False
    if getattr(st._LOCAL, "installing", False):
        return False
    try:
        return bool(install())
    except Exception:  # noqa: BLE001 -- load() must never raise
        logger.exception("zero-cost wiring: installation failed (stays stock)")
        return False
