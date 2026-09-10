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

"""fi_sampling integration into the vllm-ascend sampling path (default-off).

Mechanism (W2b package): the ``vllm.general_plugins`` entry point
``fi-sampling`` calls :func:`load`, which -- only when
``VLLM_HUST_FI_SAMPLING=1`` -- replaces the fork's
``vllm_ascend.sample.sampler.AscendTopKTopPSampler`` module attribute with
:class:`FiSamplingTopKTopPSampler`, a subclass that overrides
``forward_native``.  ``AscendSampler.__init__`` resolves that name at
construction time, so no other host object has to be touched.  This mirrors
the fork's own ``vllm_ascend/_310p/sample/sampler.py`` variant-entry pattern;
the host source trees stay untouched.

Default-off semantics: with the env unset ``load()`` returns before importing
``vllm_ascend`` or the kernel package -- nothing is patched and no kernel
module is imported, so the process is bit-identical to stock vllm-ascend.

Routing (see ``fi_sampling_route.py`` for the bench-sourced rationale) and
the five mandatory fallbacks live in pure, CPU-testable logic.  This module
owns only the device glue: lazy kernel import, host seed advance, and the
single fail-open warning.
"""

from __future__ import annotations

import logging
import os

from . import fi_sampling_route as route

logger = logging.getLogger(__name__)

#: master switch, default off (repo AGENTS.md: every capability env-gated).
ENV_ENABLE = "VLLM_HUST_FI_SAMPLING"
#: optional fixed base seed; default derives from torch's initial seed (which
#: vLLM seeds from ``--seed``) so serving stays reproducible.
ENV_SEED = "VLLM_HUST_FI_SAMPLING_SEED"
#: routing tunables, exposed so the e2e A/B can move the thresholds without a
#: code change (defaults are the reviewed bench values).
ENV_JOINT_MIN_BATCH = "VLLM_HUST_FI_SAMPLING_JOINT_MIN_BATCH"
ENV_JOINT_MIN_K = "VLLM_HUST_FI_SAMPLING_JOINT_MIN_K"
#: k=1 rows take argmax (default, per the W2 judgment table).  Set to 0 to
#: keep them on the fork chain for bit-exact tie handling (see docs).
ENV_K1_ARGMAX = "VLLM_HUST_FI_SAMPLING_K1_ARGMAX"
#: per-route debug trace (default off).  Logs the first decisions and the
#: final route histogram at process exit -- evidence that the intended route
#: actually ran in a serving leg.
ENV_TRACE = "VLLM_HUST_FI_SAMPLING_TRACE"
_TRACE_LIMIT = 20
_TRACE_HISTOGRAM_EVERY = 25

_PATCH_MARKER = "_fi_sampling_plugin_patched"

# Loaded lazily on the first FI route (default-off imports nothing).
_fi_api = None
_fi_import_failed = False
_warning_once = False
# env vars already reported as unparseable (one warning each, not per step).
_bad_env_warned: set[str] = set()
# ascend-config fallback knobs already reported as absent on this host (host
# drift must stay visible in the serving log, not just silently tolerated).
_missing_knob_warned: set[str] = set()
#: sentinel for "the host does not define this knob at all".
_MISSING = object()
# Monotonic call counter mixed into the per-call seed; fi_sampling's kernel
# decorrelates rows itself (philox offset includes the row index), so one seed
# per call is enough -- but the seed MUST advance per call, otherwise every
# decode step would draw the same uniform stream.
_call_counter = 0
_seed_base = None
# Route histogram for the trace (evidence only; default off).
_route_counts: dict[str, int] = {}
_route_traces: list[str] = []
_max_batch_seen = 0
_batch_histogram: dict[int, int] = {}


def _warn_bad_env(name: str, raw: str, default: str) -> None:
    """One warning per variable (forward_native parses these every step)."""
    if name in _bad_env_warned:
        return
    _bad_env_warned.add(name)
    logger.warning(
        "fi_sampling: %s=%r is not a valid integer; falling back to the "
        "default %r for this process.",
        name,
        raw,
        default,
    )


def _env_int(name: str, default: int) -> int:
    """Tolerant integer env read (review F1 follow-up: a garbage value must
    never raise -- these are parsed per decode step and inside ``install()``)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        _warn_bad_env(name, raw, str(default))
        return default


def _env_flag(name: str, default: str = "0") -> bool:
    return bool(_env_int(name, int(default)))


def is_enabled() -> bool:
    """True when the FI sampling path is switched on for this process."""
    return _env_flag(ENV_ENABLE)


def _joint_min_batch() -> int:
    return _env_int(ENV_JOINT_MIN_BATCH, route.FI_JOINT_MIN_BATCH)


def _joint_min_k() -> int:
    return _env_int(ENV_JOINT_MIN_K, route.FI_JOINT_MIN_K)


def _k1_argmax() -> bool:
    return _env_flag(ENV_K1_ARGMAX, "1")


def _warn_once(message: str, *args) -> None:
    global _warning_once
    if _warning_once:
        return
    logger.warning(message, *args)
    _warning_once = True


def _reset_for_tests() -> None:
    """Clear once-only state and the patch marker (unit tests only)."""
    global _warning_once, _fi_import_failed, _call_counter, _seed_base
    global _max_batch_seen
    _warning_once = False
    _fi_import_failed = False
    _call_counter = 0
    _seed_base = None
    _max_batch_seen = 0
    _bad_env_warned.clear()
    _missing_knob_warned.clear()
    _route_counts.clear()
    _route_traces.clear()
    _batch_histogram.clear()


def _trace_enabled() -> bool:
    return _env_flag(ENV_TRACE)


def _record_route(decision: str, detail: str, batch_size: int) -> None:
    """Per-call route trace (no-op unless VLLM_HUST_FI_SAMPLING_TRACE=1)."""
    global _max_batch_seen
    if not _trace_enabled():
        return
    total = sum(_route_counts.values()) + 1
    _route_counts[decision] = _route_counts.get(decision, 0) + 1
    _max_batch_seen = max(_max_batch_seen, batch_size)
    _batch_histogram[batch_size] = _batch_histogram.get(batch_size, 0) + 1
    if len(_route_traces) < _TRACE_LIMIT:
        _route_traces.append(detail)
        logger.warning("[fi-sampling trace] %s", detail)
    # The engine core is SIGTERM'd at shutdown, so an atexit dump never runs;
    # flush the histogram to stderr periodically instead (serving logs capture
    # stderr), which is what makes the e2e evidence attributable to a route.
    if total % _TRACE_HISTOGRAM_EVERY == 0:
        import sys

        print(
            f"[fi-sampling histogram calls={total} max_B={_max_batch_seen}] "
            f"{route_histogram()}",
            file=sys.stderr,
            flush=True,
        )


def route_histogram() -> dict[str, int]:
    """Route counts observed in this process (evidence helper)."""
    return dict(_route_counts)


def _install_route_histogram_dump() -> None:
    """Dump the route histogram at interpreter exit (trace mode only)."""
    if not _trace_enabled():
        return
    import atexit

    def _dump() -> None:
        if _route_counts:
            print(f"[fi-sampling histogram] {route_histogram()}", flush=True)

    atexit.register(_dump)


def _load_fi_api():
    """Import the vendored fi_sampling host API (lazy, fail-open).

    Returns the module or ``None``; the caller falls back to the fork chain.
    Imported on first use so the default-off path never pays for ``triton``.
    """
    global _fi_api, _fi_import_failed
    if _fi_api is not None:
        return _fi_api
    if _fi_import_failed:
        return None
    try:
        from .fi_sampling import api as fi_api
    except Exception as exc:  # noqa: BLE001 -- fail-open, host chain must survive
        _fi_import_failed = True
        _warn_once(
            "VLLM_HUST_FI_SAMPLING=1 but the vendored fi_sampling package could "
            "not be imported (%r); falling back to the fork sampling chain for "
            "this process.",
            exc,
        )
        return None
    _fi_api = fi_api
    return _fi_api


def _next_seed() -> int:
    """Per-call seed: reproducible under ``--seed``, different every step."""
    global _call_counter, _seed_base
    if _seed_base is None:
        # -1 sentinel: unset OR unparseable env value falls back to torch's
        # initial seed (an explicitly configured seed of -1 is masked to the
        # same value anyway by the & below, so nothing observable changes).
        env_seed = _env_int(ENV_SEED, -1)
        if env_seed >= 0:
            _seed_base = env_seed & 0x7FFFFFFF
        else:
            import torch

            _seed_base = int(torch.initial_seed()) & 0x7FFFFFFF
    _call_counter += 1
    # Mix instead of add: consecutive counters must not give correlated streams.
    return (_seed_base * 1000003 + _call_counter * 2654435761) & 0x7FFFFFFF


def _host_top_k_extremes(k) -> tuple[int | None, int | None]:
    """(min, max) of the per-row top-k as host ints.

    Costs ONE small device->host copy (``[B]`` int32, B <= max_num_seqs) per
    call that actually carries a top-k tensor; the untruncated path and the
    top-p-only path never sync.  Measured context (W2 bench §4.7): the fork
    chain's fixed host cost is 0.19-0.31 ms/call and the joint kernel itself
    is 6.7-20.9 ms at B=256/1024, so this copy is noise where it matters.
    Any device-side surprise returns ``(None, None)``, which routes the call
    back to the fork chain.
    """
    if k is None:
        return None, None
    try:
        values = k.detach().to("cpu").tolist()
    except Exception:  # noqa: BLE001 -- fail-open to the fork chain
        return None, None
    if not values:
        return None, None
    return int(min(values)), int(max(values))


def _build_sampler_class(base_cls, torch):
    """Create the ``forward_native``-overriding subclass for ``base_cls``."""

    class FiSamplingTopKTopPSampler(base_cls):
        """AscendTopKTopPSampler with the fi_sampling fast paths.

        Only ``forward_native`` is overridden; ``forward`` is bound to it in
        ``__init__`` by the parent class on this platform, exactly as for the
        fork class itself.
        """

        def forward_native(self, logits, generators, k, p):
            if not is_enabled():
                # Second gate: never diverge if the env is cleared mid-process.
                return super().forward_native(logits, generators, k, p)

            batch_size = int(logits.shape[0])
            has_top_k = k is not None
            has_top_p = p is not None
            batch_invariant, reduce_sample, async_exponential = _fallback_flags()

            # (1) Cheap fallbacks first: these need no device sync, and a
            # fallback leg must never pay for a host read it will not use.
            reason = route.fallback_reason(
                logprobs_mode=self.logprobs_mode,
                has_generators=bool(generators),
                batch_invariant=batch_invariant,
                reduce_sample=reduce_sample,
                async_exponential=async_exponential,
            )
            if reason is not None:
                _record_route(
                    route.ROUTE_FORK,
                    f"route=fork reason={reason} B={batch_size}",
                    batch_size,
                )
                return super().forward_native(logits, generators, k, p)

            # (2) Untruncated / top-p-only cells need no host read either.
            k1_argmax = _k1_argmax()
            top_k_min = top_k_max = None
            if has_top_k and route.needs_top_k_read(
                batch_size=batch_size,
                joint_min_batch=_joint_min_batch(),
            ):
                # (3) Only large-batch top-k cells need the per-row extremes.
                # The read is a per-step host sync: the first api1 pair showed
                # +4.9% median TPOT purely from one D2H/step (REPORT.md §4), so
                # it is taken only where the joint/k=1 cells can fire.  Raw copy
                # cost 0.068/0.098 ms at B=64/512 (logs/micro_host_top_k.txt).
                top_k_min, top_k_max = _host_top_k_extremes(k)

            decision = route.decide_route(
                logprobs_mode=self.logprobs_mode,
                has_generators=False,
                batch_invariant=False,
                reduce_sample=False,
                async_exponential=False,
                batch_size=batch_size,
                has_top_k=has_top_k,
                has_top_p=has_top_p,
                top_k_min=top_k_min,
                top_k_max=top_k_max,
                joint_min_k=_joint_min_k(),
                joint_min_batch=_joint_min_batch(),
                k1_argmax=k1_argmax,
            )
            _record_route(
                decision,
                f"route={decision} B={batch_size} k={top_k_min}..{top_k_max} "
                f"top_k={has_top_k} top_p={has_top_p} "
                f"generators={len(generators)} logprobs_mode={self.logprobs_mode}",
                batch_size,
            )
            if decision == route.ROUTE_FORK:
                return super().forward_native(logits, generators, k, p)

            if decision == route.ROUTE_ARGMAX:
                # k == 1 on every row keeps only the maximum token(s); the fork
                # chain is the slowest option here (bench table C) and the
                # joint kernel is slower still.  Tie rule: the fork chain draws
                # among tied maxima, argmax takes the lowest index (identical
                # unless logits have exact ties); set
                # VLLM_HUST_FI_SAMPLING_K1_ARGMAX=0 for bit-exact behavior.
                return torch.argmax(logits, dim=-1).view(-1), None

            fi_api = _load_fi_api()
            if fi_api is None:
                return super().forward_native(logits, generators, k, p)

            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            seed = _next_seed()
            try:
                if decision == route.ROUTE_FI_API1:
                    tokens = fi_api.sampling_from_probs(
                        probs,
                        seed=seed,
                        block_v=route.FI_API1_BLOCK_V,
                    )
                else:
                    tokens = fi_api.top_k_top_p_sampling_from_probs(
                        probs,
                        top_k=k if has_top_k else None,
                        top_p=p if has_top_p else None,
                        seed=seed,
                        block_v=route.FI_JOINT_BLOCK_V,
                    )
            except Exception as exc:  # noqa: BLE001 -- fail-open to the fork chain
                _warn_once(
                    "fi_sampling kernel call failed (%r); falling back to the "
                    "fork sampling chain for the rest of this process.",
                    exc,
                )
                return super().forward_native(logits, generators, k, p)
            # int32 from the kernel is fine: Sampler.forward casts to long.
            return tokens.view(-1), None

    FiSamplingTopKTopPSampler.__name__ = "FiSamplingTopKTopPSampler"
    FiSamplingTopKTopPSampler.__qualname__ = "FiSamplingTopKTopPSampler"
    return FiSamplingTopKTopPSampler


def _batch_invariant() -> bool:
    import vllm.envs as envs

    return bool(envs.VLLM_BATCH_INVARIANT)


def _ascend_config():
    from vllm_ascend.ascend_config import get_ascend_config

    return get_ascend_config()


def _config_flag(config, name: str) -> bool:
    """Read one optional ascend-config fallback knob, tolerating host drift.

    A knob this host version does not define gates a feature that does not
    exist here, so it cannot be enabled -> ``False``; the absence is warned
    once per knob so the drift stays visible in the serving log.  Reading a
    knob that IS present but raises still propagates, so the caller keeps
    failing closed on a genuinely unreadable config.

    Rationale (2026-09-10, new baseline): ``AscendConfig`` dropped
    ``enable_async_exponential`` in vllm-ascend commit ``4f0a38a95``
    ("[Refactor] asnc exponential optimization unset", #12306).  Probing it by
    plain attribute access made ``_fallback_flags`` raise on
    vllm-ascend ``0.25.1rc2``, which failed CLOSED and silently pushed every
    call to the fork chain -- the e2e route histogram was 100% ``fork`` with
    the plugin reporting itself ACTIVE.
    """
    value = getattr(config, name, _MISSING)
    if value is _MISSING:
        if name not in _missing_knob_warned:
            _missing_knob_warned.add(name)
            logger.warning(
                "fi_sampling: this vllm-ascend host does not define the "
                "ascend-config knob '%s'; treating the feature it gates as "
                "disabled (feature removed upstream).",
                name,
            )
        return False
    return bool(value)


def _fallback_flags() -> tuple[bool, bool, bool]:
    """(batch_invariant, reduce_sample, async_exponential) for the route call.

    Any failure to resolve the vllm/vllm-ascend config (import failure, missing
    attribute, config not ready) is treated as "all five fallbacks active",
    i.e. the call goes to the fork chain.  Failing closed here is what keeps
    the plugin unable to change behavior it cannot inspect.
    """
    try:
        batch_invariant = _batch_invariant()
        config = _ascend_config()
        return (
            batch_invariant,
            _config_flag(config, "enable_reduce_sample"),
            _config_flag(config, "enable_async_exponential"),
        )
    except Exception as exc:  # noqa: BLE001 -- conservative: stay on fork chain
        _warn_once(
            "fi_sampling could not read the vllm/vllm-ascend config (%r); routing "
            "every request to the fork sampling chain.",
            exc,
        )
        return True, True, True


def install() -> bool:
    """Class-level replacement of ``AscendTopKTopPSampler``.

    Returns True when the patched subclass is in place.  Idempotent; never
    raises for a missing/unusable vllm-ascend (fail-open with one warning).
    """
    try:
        import torch
        import vllm_ascend.sample.sampler as sampler_mod
    except Exception as exc:  # noqa: BLE001 -- fail-open (review F1: the host
        # module itself may be missing/renamed in a future fork; load() is a
        # vllm.general_plugins entry point and must never raise into the
        # plugin loader, or the whole engine process dies at startup).
        _warn_once(
            "VLLM_HUST_FI_SAMPLING=1 but the host sampler module could not be "
            "imported (%r); serving stays on the fork sampling chain.",
            exc,
        )
        return False
    # Resolve every knob BEFORE touching the host module (review follow-up):
    # after the class replacement below, a raise would leave the host module
    # half-installed.  (These are tolerant reads and cannot raise today; this
    # ordering keeps that invariant structural.)
    joint_min_batch = _joint_min_batch()
    joint_min_k = _joint_min_k()
    if getattr(sampler_mod, _PATCH_MARKER, False):
        return True
    try:
        base_cls = sampler_mod.AscendTopKTopPSampler
        patched = _build_sampler_class(base_cls, torch)
        sampler_mod.AscendTopKTopPSampler = patched
        sampler_mod._FiSamplingTopKTopPSampler = patched
        sampler_mod._fi_sampling_plugin_patched = True
    except Exception as exc:  # noqa: BLE001 -- fail-open
        _warn_once(
            "VLLM_HUST_FI_SAMPLING=1 but the AscendTopKTopPSampler replacement "
            "failed (%r); serving stays on the fork sampling chain.",
            exc,
        )
        return False
    logger.warning(
        "fi_sampling sampling path is ACTIVE (VLLM_HUST_FI_SAMPLING=1): "
        "untruncated sampling uses the flashinfer-semantics triton-ascend "
        "kernel, joint top-k/top-p at B>=%d with k>=%d uses the joint kernel "
        "and k=1 rows use argmax; every other case falls back to the fork "
        "chain. Output distribution matches flashinfer semantics, not the "
        "fork's exponential-race chain (see docs/fi-sampling.md).",
        joint_min_batch,
        joint_min_k,
    )
    _install_route_histogram_dump()
    return True


def load() -> bool:
    """``vllm.general_plugins`` entry point (default-off)."""
    if not is_enabled():
        # Default-off: no import of vllm_ascend / fi_sampling, no patch.
        return False
    return install()
