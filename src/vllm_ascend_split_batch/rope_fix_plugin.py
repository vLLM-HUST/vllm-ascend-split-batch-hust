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

"""RoPE variant defects ①/②/③ carried plugin-side (default-off).

Three *real* defects of the fork baseline ``vllm-ascend-hust
main@74f0c0a27`` (design note ``docs/design/rope-variant-defects.md``,
numeric evidence ``knowledge/surveys/contrast/norm-rope-act/``):

① ``vllm_ascend/utils.py``'s ``REGISTERED_ASCEND_OPS`` has no
   ``"Llama3RotaryEmbedding"`` key, so a llama3-scaling model keeps the host
   class and lands on ``CustomOp.forward_oot`` (the default = ``forward_native``,
   bf16 element-wise math): the Ascend triton rope kernel is unreachable.
② ``AscendMRotaryEmbedding.forward_triton`` calls ``triton_mrope`` with 8
   positional arguments while the host signature has 9 (trailing
   ``is_neox_style``, no default) -> ``TypeError`` on every interleaved mrope
   batch (Qwen2.5-VL shape).
③ ``AscendYaRNRotaryEmbedding.__init__`` defaults ``truncate=False`` while
   vLLM/HF default to ``True`` -> a different cos/sin cache than host/GPU for
   the Qwen2.5-style yarn configs that do not spell ``truncate`` out.

Carrier mechanism (no host source edit, no second ``register_oot``):

* ``CustomOp.register_oot`` asserts the name is still absent, so the plugin
  must NOT register anything itself; it replaces the *entries* of
  ``vllm.model_executor.custom_op.op_registry_oot``, which ``CustomOp.__new__``
  (and ``PluggableLayer.__new__``) reads at **instantiation** time -- i.e. after
  the host's registration pass.
* ``vllm_ascend.utils.register_ascend_customop`` is therefore *wrapped*: the
  wrapper calls the original first and only then points the three rope keys at
  the plugin subclasses, so the host's ``assert reg_name not in op_registry_oot``
  never sees a pre-planted name.  The wrapper is installed on every module that
  already holds a direct reference to the function (``vllm_ascend/utils.py``,
  ``vllm_ascend/worker/worker.py``, ...) plus on the ``vllm_ascend.utils``
  attribute itself, which covers imports that happen later.
* The three fixes are plugin-side subclasses of the *host* rope classes (defined
  lazily inside :func:`_make_override_classes`, so the default-off path never
  imports torch/torch_npu/vllm_ascend).  A per-key precondition check refuses an
  override it does not understand (e.g. the 310P compatibility variants) and
  keeps the host implementation instead.

Default-off contract: with ``VLLM_HUST_ROPE_FIX`` unset :func:`load` returns
before importing ``vllm_ascend`` / ``torch`` / ``torch_npu``, before touching
any registry and without logging a single line, so the process stays
bit-identical to stock vllm/vllm-ascend.

Fail-open contract: a missing/renamed host seam, an unexpected registry entry or
any internal error only produces ONE warning and leaves the stock rope classes in
place -- nothing is ever raised into the plugin loader or into a worker.

Drift guard: ``tests/test_rope_fix_drift.py`` asserts, source-level, that the
three *defects are still present* upstream (so that an upstream fix fails the
guard and tells us to drop the now-redundant override) and that every seam this
carrier leans on still matches (otherwise the "anchor drifted" failure asks for a
re-audit of ``HOST_CONTRACT.md``).
"""

from __future__ import annotations

import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

#: master switch, default off (repo AGENTS.md: every capability env-gated).
ENV_ENABLE = "VLLM_HUST_ROPE_FIX"

# ------------------------------------------------------------- host seam names

#: module holding the registration pass this carrier has to wrap.
HOST_UTILS_MODULE = "vllm_ascend.utils"
#: the registration function itself (wrapped, never replaced semantically).
HOST_REGISTRATION_FUNC = "register_ascend_customop"
#: module holding the fork rope classes the plugin subclasses.
HOST_ROPE_MODULE = "vllm_ascend.ops.rotary_embedding"
#: module + attribute of the out-of-tree registry (the override lever).
CUSTOM_OP_MODULE = "vllm.model_executor.custom_op"
OP_REGISTRY_ATTR = "op_registry_oot"

#: the three rope registry keys owned by this carrier.
KEY_LLAMA3 = "Llama3RotaryEmbedding"
KEY_MROPE = "MRotaryEmbedding"
KEY_YARN = "YaRNScalingRotaryEmbedding"
OVERRIDE_KEYS = (KEY_LLAMA3, KEY_MROPE, KEY_YARN)

#: marker set on the wrapper so a second ``install()`` cannot wrap itself.
_WRAP_MARKER = "_vllm_hust_rope_fix_wrapper"

#: ``triton_mrope``, bound inside :func:`_make_override_classes`.  It is a module
#: global on purpose: the mirrored ``forward_triton`` body has to reference the
#: very same bare name the host method does, which is what makes the body-mirror
#: drift guard (``tests/test_rope_fix_drift.py``) an AST comparison.
triton_mrope = None

# ----------------------------------------------------------------- bookkeeping

_lock = threading.Lock()
_installed = False
#: once-per-message log guard (a plain logging.Logger has no *_once helpers).
_ONCE_SEEN: set = set()
#: ``{registry_key: class_that_was_there_before}`` for ``uninstall()``.
_ORIG: dict = {}
#: the three plugin subclasses (empty until ``install()`` ran).
_OVERRIDE_CLASSES: dict = {}
#: observable state; ``stats()`` returns a copy.
_STATE: dict = {"applied": {}, "skipped": {}, "wrapper": None, "wrapped": []}


def is_enabled() -> bool:
    """True only for the exact enable token (default-off, no truthiness fuzz)."""
    return os.getenv(ENV_ENABLE) == "1"


def _warn_once(message: str, *args) -> None:
    key = "warn:" + message
    if key not in _ONCE_SEEN:
        _ONCE_SEEN.add(key)
        logger.warning(message, *args)


def stats() -> dict:
    """Snapshot of the carrier state (tests / service-log observability)."""
    return {
        "installed": _installed,
        "applied": dict(_STATE["applied"]),
        "skipped": dict(_STATE["skipped"]),
        "wrapped": list(_STATE["wrapped"]),
    }


def _reset_for_tests() -> None:
    """Clear once-guards and the observable state (does not unpatch)."""
    _ONCE_SEEN.clear()
    _STATE["applied"].clear()
    _STATE["skipped"].clear()
    _STATE["wrapped"] = []


# ------------------------------------------------------- the three overrides


def _make_override_classes() -> dict:
    """Build the plugin-side subclasses (lazy: needs torch / vllm_ascend).

    Only ever called from :func:`install`, i.e. only on the enabled path, so the
    default-off ``load()`` imports nothing.
    """
    global triton_mrope

    import torch
    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import Llama3RotaryEmbedding
    from vllm_ascend.ops import rotary_embedding as fork_rope

    AscendMRotaryEmbedding = fork_rope.AscendMRotaryEmbedding
    AscendRotaryEmbedding = fork_rope.AscendRotaryEmbedding
    AscendYaRNRotaryEmbedding = fork_rope.AscendYaRNRotaryEmbedding
    # Private helper of the fork module; absent -> only the MTP/MLA cache
    # bookkeeping is skipped (warn once below), the forward path is unaffected.
    record_cos_sin_cache = getattr(fork_rope, "_record_cos_sin_cache", None)
    if record_cos_sin_cache is None:
        _warn_once(
            "RoPE fix: %s._record_cos_sin_cache is missing; the llama3 override "
            "installs without the cos/sin cache bookkeeping (MTP/MLA paths may "
            "miss the cache record).",
            HOST_ROPE_MODULE,
        )
    # ``triton_mrope`` is imported by the fork module under ``if HAS_TRITON`` only;
    # without triton ``forward_triton`` is unreachable there as well.
    try:
        from vllm.model_executor.layers.rotary_embedding.mrope import (
            triton_mrope as host_triton_mrope,
        )
    except Exception:  # noqa: BLE001 -- no triton: the mirrored path stays cold
        host_triton_mrope = None
    triton_mrope = host_triton_mrope

    class AscendFixLlama3RotaryEmbedding(Llama3RotaryEmbedding):
        """llama3-scaling rope on the Ascend kernel (defect ①).

        Same 10-argument construction signature as the host class (so
        ``CustomOp.__new__`` can instantiate it with the host's own argument
        list), plus the two side effects the fork's rope classes carry
        (``use_mtp`` for the draft-model all-gather branch of
        ``AscendRotaryEmbedding.forward_oot``, and the module-level cos/sin cache
        record), plus ``forward_oot`` delegating to that same Ascend forward.
        Only the forward path changes: the cache generation of
        ``Llama3RotaryEmbedding`` is already fp32-exact and single-rounded.
        """

        def __init__(
            self,
            head_size: int,
            rotary_dim: int,
            max_position_embeddings: int,
            base: float,
            is_neox_style: bool,
            dtype,
            scaling_factor: float,
            low_freq_factor: float,
            high_freq_factor: float,
            orig_max_position: int,
        ) -> None:
            super().__init__(
                head_size,
                rotary_dim,
                max_position_embeddings,
                base,
                is_neox_style,
                dtype,
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                orig_max_position,
            )
            vllm_config = get_current_vllm_config()
            self.use_mtp = bool(
                vllm_config.speculative_config
                and vllm_config.speculative_config.method == "mtp"
            )
            if record_cos_sin_cache is not None:
                record_cos_sin_cache(self.cos_sin_cache)

        def forward_oot(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: torch.Tensor | None = None,
            is_neox_style_override: bool | None = None,
            out_dtype: torch.dtype | None = None,
        ):
            return AscendRotaryEmbedding.forward_oot(
                self,
                positions,
                query,
                key,
                offsets,
                is_neox_style_override,
                out_dtype,
            )

    class AscendFixMRotaryEmbedding(AscendMRotaryEmbedding):
        """Interleaved mrope with the missing ``is_neox_style`` (defect ②).

        ``forward_triton`` is mirrored statement by statement from
        ``vllm_ascend.ops.rotary_embedding.AscendMRotaryEmbedding.forward_triton``
        with exactly one delta: the trailing ``self.is_neox_style`` handed to
        ``triton_mrope``.  The fork hard-codes the 8-argument call in the middle
        of that method and offers no seam to inject the ninth one (``triton_mrope``
        is a module global, the call site takes no hook), so re-stating the body
        in a subclass is the smallest carrier that needs no host edit; the mirror
        is policed by the AST comparison in ``tests/test_rope_fix_drift.py``.

        No docstring on purpose: the drift guard compares the AST of this method
        with the host's, and a docstring would show up as an extra node.
        """

        def forward_triton(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
            offsets: torch.Tensor | None = None,
        ):
            assert positions.ndim == 2
            assert key is not None

            self._match_cos_sin_cache_dtype(query)
            self.cos = None
            self.sin = None
            if self.cos is None and self.sin is None:
                cos_sin = self.cos_sin_cache[positions]  # type: ignore
                cos, sin = cos_sin.chunk(2, dim=-1)
                self.cos = cos.contiguous()
                self.sin = sin.contiguous()
            query_shape = query.shape
            key_shape = key.shape

            assert self.mrope_section

            # When the grid becomes large, enable TRITON_ALL_BLOCKS_PARALLEL
            # to avoid scheduler/runtime failures.
            if (
                query_shape[0] > self._ASCEND_TRITON_GRID_LIMIT
                and os.environ.get("TRITON_ALL_BLOCKS_PARALLEL") != "1"
            ):
                os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

            q, k = triton_mrope(
                query,
                key,
                self.cos,
                self.sin,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
                self.mrope_interleaved,
                self.is_neox_style,  # defect ② fix: the fork dropped this one
            )

            return q.reshape(query_shape), k.reshape(key_shape)

    class AscendFixYaRNRotaryEmbedding(AscendYaRNRotaryEmbedding):
        """YaRN rope with the vLLM/HF ``truncate`` default (defect ③).

        Deliberately signature-agnostic: everything the host's ``get_rope`` hands
        the fork class is forwarded verbatim and only the *default* of the
        keyword-only ``truncate`` changes to ``True`` (HF/vLLM semantics: the
        low/high correction dims are floor/ceil'ed).  An explicit ``truncate`` in
        ``rope_parameters`` still wins -- the escape hatch is preserved.
        """

        def __init__(self, *args, truncate: bool = True, **kwargs) -> None:
            super().__init__(*args, truncate=truncate, **kwargs)

    return {
        KEY_LLAMA3: AscendFixLlama3RotaryEmbedding,
        KEY_MROPE: AscendFixMRotaryEmbedding,
        KEY_YARN: AscendFixYaRNRotaryEmbedding,
    }


def _host_registration_ran() -> bool:
    """True when the host's registration pass has already populated the registry.

    Used by ``install()`` to decide whether an *eager* override attempt is
    meaningful (registration ran before ``load()``) or premature (the wrapper
    will do the work later).
    """
    try:
        from vllm.model_executor.custom_op import op_registry_oot
    except Exception:  # noqa: BLE001 -- nothing to inspect yet
        return False
    return any(op_registry_oot.get(key) is not None for key in (KEY_MROPE, KEY_YARN))


def _apply_overrides(post_registration: bool) -> dict:
    """Point the three ``op_registry_oot`` keys at the plugin classes.

    Idempotent and fail-open per key: an entry this carrier does not recognise
    (e.g. the 310P compatibility variants, or an upstream fix that already wired
    something) is left alone with one warning instead of being clobbered.

    ``post_registration`` states whether the host's registration pass has already
    returned.  It is deliberately required -- and the answer matters: **nothing is
    ever written into the registry before that pass**, because
    ``CustomOp.register_oot`` asserts the name is still absent and a pre-planted
    key would kill engine init (and would also collide the day upstream fixes
    defect ① itself).  Before the pass, an absent ``MRotaryEmbedding`` /
    ``YaRNScalingRotaryEmbedding`` entry simply means "not registered yet" and is
    skipped silently.

    Returns ``{key: reason}`` for the keys actually (re)pointed.
    """
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm_ascend.ops import rotary_embedding as fork_rope

    before = dict(_STATE["applied"])
    applied = dict(_STATE["applied"])
    skipped = dict(_STATE["skipped"])

    cls = _OVERRIDE_CLASSES[KEY_LLAMA3]
    current = op_registry_oot.get(KEY_LLAMA3)
    if current is cls:
        pass
    elif current is None:
        if post_registration:
            # Defect ① is a coverage gap: there is nothing to replace, the key
            # has to be created.  The host pass never writes this name, so the
            # creation cannot trip its duplicate-name assert.
            op_registry_oot[KEY_LLAMA3] = cls
            applied[KEY_LLAMA3] = "created (defect ①: the fork never wired it)"
    else:
        skipped[KEY_LLAMA3] = f"a foreign entry is already registered: {current!r}"

    bases = {
        KEY_MROPE: fork_rope.AscendMRotaryEmbedding,
        KEY_YARN: fork_rope.AscendYaRNRotaryEmbedding,
    }
    for key, base in bases.items():
        cls = _OVERRIDE_CLASSES[key]
        current = op_registry_oot.get(key)
        if current is cls:
            continue
        if current is None and not post_registration:
            # The host registers these keys itself; absent means "not yet".
            continue
        if current is base:
            # Only the fork class this carrier subclasses is replaced; anything
            # else (310P variants, a future upstream rework) stays untouched.
            _ORIG.setdefault(key, current)
            op_registry_oot[key] = cls
            applied[key] = f"replaced {base.__name__}"
        else:
            name = getattr(current, "__name__", current)
            skipped[key] = f"expected the fork class {base.__name__}, found {name!r}"

    for key, reason in skipped.items():
        _warn_once(
            "RoPE fix: op_registry_oot[%r] was left on the host implementation "
            "(%s); this key keeps the fork behaviour (fail-open).",
            key,
            reason,
        )
    _STATE["applied"] = applied
    _STATE["skipped"] = skipped
    newly = sorted(key for key in applied if key not in before)
    if newly:
        announce = "warn:rope-fix-applied:" + ",".join(newly)
        if announce not in _ONCE_SEEN:
            _ONCE_SEEN.add(announce)
            logger.warning(
                "RoPE fix: rope overrides in place: %s.",
                "; ".join(f"{key} -> {why}" for key, why in sorted(applied.items())),
            )
    return applied


def _wrap_registration(host_utils):
    """Wrap ``register_ascend_customop`` so the overrides land after the pass.

    The wrapper is installed on every module that already holds a direct
    reference to the original (``from vllm_ascend.utils import
    register_ascend_customop`` in ``vllm_ascend/worker/worker.py`` is the live
    call site) and on the ``vllm_ascend.utils`` attribute itself, which is what
    imports happening later will pick up.  Returns the wrapper, or ``None`` when
    the seam is missing.
    """
    original = getattr(host_utils, HOST_REGISTRATION_FUNC, None)
    if not callable(original):
        _warn_once(
            "RoPE fix: %s.%s is missing (found %r); the rope overrides cannot be "
            "applied after the host registration pass, so the process stays on "
            "the fork's stock rope classes.",
            HOST_UTILS_MODULE,
            HOST_REGISTRATION_FUNC,
            type(original).__name__,
        )
        return None
    if getattr(original, _WRAP_MARKER, False):
        return original

    def _wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            _apply_overrides(post_registration=True)
        except Exception as exc:  # noqa: BLE001 -- never break the host init
            _warn_once(
                "RoPE fix: applying the rope overrides after host registration "
                "failed (%r); the process stays on the fork's stock rope classes.",
                exc,
            )
        return result

    setattr(_wrapped, _WRAP_MARKER, True)
    _wrapped.__wrapped__ = original
    _wrapped.__name__ = getattr(original, "__name__", HOST_REGISTRATION_FUNC)
    _wrapped.__doc__ = original.__doc__
    patched = []
    for mod_name, module in list(sys.modules.items()):
        if module is None:
            continue
        try:
            if getattr(module, HOST_REGISTRATION_FUNC, None) is original:
                setattr(module, HOST_REGISTRATION_FUNC, _wrapped)
                patched.append(mod_name)
        except Exception:  # noqa: BLE001 -- arbitrary objects must not break it
            continue
    setattr(host_utils, HOST_REGISTRATION_FUNC, _wrapped)
    if HOST_UTILS_MODULE not in patched:
        patched.append(HOST_UTILS_MODULE)
    _STATE["wrapped"] = sorted(patched)
    logger.warning(
        "RoPE fix: %s.%s is wrapped (call sites rebound: %s); the rope overrides "
        "are applied right after the host's own registration pass.",
        HOST_UTILS_MODULE,
        HOST_REGISTRATION_FUNC,
        ", ".join(sorted(patched)),
    )
    return _wrapped


# ---------------------------------------------------------------- install/load


def install() -> bool:
    """Install the carrier (idempotent, fail-open).

    Returns True when the wrapper + the override classes are in place.  A
    missing/renamed seam or any internal error only warns once and leaves the
    process on the stock rope classes; this method never raises into the plugin
    loader.
    """
    global _installed
    if _installed:
        return True
    with _lock:
        if _installed:
            return True
        try:
            import vllm_ascend.utils as host_utils
        except Exception as exc:  # noqa: BLE001 -- load() must never raise
            _warn_once(
                "RoPE fix requested (VLLM_HUST_ROPE_FIX=1) but %s could not be "
                "imported (%r); the process stays on the fork's stock rope "
                "classes.",
                HOST_UTILS_MODULE,
                exc,
            )
            return False
        try:
            if not _OVERRIDE_CLASSES:
                _OVERRIDE_CLASSES.update(_make_override_classes())
            wrapper = _wrap_registration(host_utils)
            if wrapper is None:
                # No seam means no way to prove that the overrides land after
                # the host's registration pass -> touch nothing at all.
                return False
            _STATE["wrapper"] = wrapper
            # Eager attempt: covers the case where the registration pass already
            # ran (then the wrapper never fires).  Nothing is written while the
            # pass is still pending -- see ``_apply_overrides``.
            _apply_overrides(post_registration=_host_registration_ran())
        except Exception as exc:  # noqa: BLE001 -- fail-open, stay stock
            _warn_once(
                "RoPE fix installation failed (%r); the process stays on the "
                "fork's stock rope classes.",
                exc,
            )
            return False
        _installed = True
        logger.warning(
            "RoPE variant defect carrier is ACTIVE (VLLM_HUST_ROPE_FIX=1): "
            "op_registry_oot[%s] will be pointed at the plugin subclasses right "
            "after vllm-ascend registers its custom ops -- ① llama3-scaling onto "
            "the Ascend rope kernel, ② the missing triton_mrope is_neox_style, "
            "③ the YaRN truncate=True default of vLLM/HF.",
            ", ".join(OVERRIDE_KEYS),
        )
        return True


def uninstall() -> None:
    """Restore the wrapped host reference and the overridden registry entries."""
    global _installed
    with _lock:
        wrapper = _STATE["wrapper"]
        if wrapper is not None:
            original = getattr(wrapper, "__wrapped__", None)
            for _mod_name, module in list(sys.modules.items()):
                if module is None:
                    continue
                try:
                    if getattr(module, HOST_REGISTRATION_FUNC, None) is wrapper:
                        setattr(module, HOST_REGISTRATION_FUNC, original)
                except Exception:  # noqa: BLE001 -- best-effort restore
                    continue
        try:
            from vllm.model_executor.custom_op import op_registry_oot

            for key, cls in _OVERRIDE_CLASSES.items():
                if op_registry_oot.get(key) is not cls:
                    continue
                if key in _ORIG:
                    op_registry_oot[key] = _ORIG[key]
                else:
                    del op_registry_oot[key]
        except Exception:  # noqa: BLE001 -- registry may not be importable here
            logger.exception("RoPE fix uninstall could not restore the registry")
        _ORIG.clear()
        _STATE["wrapper"] = None
        _STATE["applied"].clear()
        _STATE["skipped"].clear()
        _installed = False


def load() -> bool:
    """``vllm.general_plugins`` entry point (default-off).

    With ``VLLM_HUST_ROPE_FIX`` unset this returns before importing
    ``vllm_ascend`` / ``torch`` / ``torch_npu``, before touching any registry and
    without logging: no patch, no import side effect, bit-identical to stock.
    """
    if not is_enabled():
        return False
    return install()
