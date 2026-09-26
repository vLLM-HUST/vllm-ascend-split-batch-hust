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

"""Operator dispatch (selection) skeleton -- flashinfer-style ``plan``/``select``.

The model-landing hotspot pass is expected to surface a set of operators that
carry several interchangeable implementations (host primitive / own kernel /
fused combination) whose ranking depends on shape and scene.  This module is
the *interface and extension point* for that future selection: it decides
nothing about real kernels, registers no candidate, and is wired into no
runtime path (see ``docs/design/operator-dispatch.md``).

Semantics (mirroring the two in-repo precedents):

- ``fi_sampling_route.decide_route`` -- rule-based selection ("regla" tier);
- ``cascade_gate.decision_for`` -- measured selection from a startup bench plus
  an env override ("measured" tier).

Both collapse here into one ``OpSelector`` with a fixed decision-source
ladder, highest priority first:

1. ``SOURCE_ENV``      ``VLLM_ASCEND_OP_SELECT_<OP>`` / global override;
2. ``SOURCE_OVERRIDE`` programmatic :meth:`OpSelector.override` (test / A-B);
3. ``SOURCE_BENCH``    runtime bench table (:meth:`OpSelector.set_bench_table`);
4. ``SOURCE_STATIC``   declared shape/scene table (:meth:`set_static_table`);
5. ``SOURCE_COST``     smallest ``cost_hint`` among applicable candidates;
6. ``SOURCE_FALLBACK`` fail-open to the host path (``default``).

Contract invariants:

- **default-off**: :meth:`OpSelector.enabled` is the master gate and reads a
  env var that defaults to unset; the future wiring -- not this module -- is
  responsible for checking it before asking ``select`` (same split as
  ``cascade_gate.enabled``).
- **fail-open**: anything that cannot resolve -- no applicable candidate, an
  override naming an unknown/inapplicable candidate, an applicability
  predicate that raises -- returns the host implementation name, never a
  speculative kernel.  An explicit override is authoritative: it either
  applies or falls back, it never silently reaches a *different* kernel.
- **auditable**: every ``select`` records an :class:`OpDecision`
  (chosen / source / reason / context) into a bounded ring buffer.

Pure logic: no torch / vllm / triton import, CPU-testable
(``tests/test_selector.py``).
"""

from __future__ import annotations

import os
import re
import threading
from collections import deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

#: Implementation name assumed to be the always-available host primitive.
DEFAULT_HOST = "host"

#: Bounded audit ring size (per selector).
AUDIT_CAPACITY = 256

# Decision source identifiers (stable strings, surfaced in the audit record).
SOURCE_ENV = "env"
SOURCE_OVERRIDE = "override"
SOURCE_BENCH = "bench"
SOURCE_STATIC = "static"
SOURCE_COST = "cost_hint"
SOURCE_FALLBACK = "fallback"

#: Global override form, comma-separated ``<op>=<candidate>`` pairs.
ENV_OVERRIDE = "VLLM_ASCEND_OP_SELECT"
#: Per-operator override form, a single ``<candidate>`` name.
ENV_OVERRIDE_TEMPLATE = "VLLM_ASCEND_OP_SELECT_{op}"
#: Master enable gate (default unset -> disabled).
ENV_ENABLE = "VLLM_ASCEND_OP_DISPATCH"


def _always_applicable(ctx: OpContext) -> bool:
    """Default applicability predicate: every context matches."""
    return True


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    return default if value is None else str(value)


def _env_suffix(op: str) -> str:
    """``fi_sampling`` -> ``FI_SAMPLING`` (per-operator env suffix)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", op).strip("_").upper()


@dataclass(frozen=True)
class OpContext:
    """Shape/scene payload a caller hands to ``select``.

    The fields are the intersection of the two precedents, not an invented
    vocabulary:

    - ``num_tokens``  -- flattened token count of the step, i.e. the batch
      bucket in ``cascade_gate`` and ``batch_size`` in ``fi_sampling_route``;
    - ``shared_len``  -- shared/prefix length of the step (``cascade_gate``
      keyed its measured verdict on ``(num_tokens, prefix bucket)``);
    - ``headdim`` / ``dtype`` -- steady kernel-shape discriminators;
    - ``scene``       -- free-form categorical tag (e.g. ``"greedy_k1"``,
      ``"untruncated"``, ``"joint_truncated"``) carrying whatever rule cell the
      caller recognized, exactly as ``fi_sampling_route`` labels its routes.

    Op-specific payloads are a convention on ``scene`` in this skeleton;
    widening the dataclass is the supported way to grow it later.
    """

    num_tokens: int = 0
    shared_len: int = 0
    headdim: int = 0
    dtype: str = ""
    scene: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> OpContext:
        """Build a context from a plain dict (unknown keys are ignored)."""
        return cls(
            num_tokens=_as_int(payload.get("num_tokens", 0)),
            shared_len=_as_int(payload.get("shared_len", 0)),
            headdim=_as_int(payload.get("headdim", 0)),
            dtype=_as_str(payload.get("dtype", "")),
            scene=_as_str(payload.get("scene", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_tokens": self.num_tokens,
            "shared_len": self.shared_len,
            "headdim": self.headdim,
            "dtype": self.dtype,
            "scene": self.scene,
        }

    def key(self) -> tuple[Any, ...]:
        """Default table key: the full payload, shape first after the scene."""
        return (
            self.scene,
            self.num_tokens,
            self.shared_len,
            self.headdim,
            self.dtype,
        )


@dataclass(frozen=True)
class OpCandidate:
    """One interchangeable implementation of an operator.

    ``cost_hint`` is a lower-is-better estimate used only by the weakest
    decision tier; ``None`` means "no opinion" (the candidate then wins only
    through an explicit table/override entry).  ``is_applicable`` must be a
    pure predicate over the context; if it raises, the candidate is treated as
    inapplicable (fail-open), it never propagates.
    """

    name: str
    is_applicable: Callable[[OpContext], bool] = _always_applicable
    cost_hint: float | None = None
    label: str = ""

    def applicable(self, ctx: OpContext) -> bool:
        try:
            return bool(self.is_applicable(ctx))
        except Exception:  # noqa: BLE001  (fail-open: a broken predicate loses)
            return False


@dataclass(frozen=True)
class OpDecision:
    """Audit record for one selection."""

    op: str
    chosen: str
    source: str
    reason: str
    context: OpContext


class OpSelector:
    """Registry + decision ladder for one operator (pure logic)."""

    def __init__(
        self,
        op: str,
        default: str = DEFAULT_HOST,
        *,
        key_fn: Callable[[OpContext], Hashable] | None = None,
        audit_capacity: int = AUDIT_CAPACITY,
        enabled_env: str = ENV_ENABLE,
        global_env: str = ENV_OVERRIDE,
    ) -> None:
        if not op:
            raise ValueError("op name must be a non-empty string")
        self._op = op
        self._default = default or DEFAULT_HOST
        self._key_fn: Callable[[OpContext], Hashable] = key_fn or (
            lambda ctx: ctx.key()
        )
        self._audit_capacity = max(1, _as_int(audit_capacity, AUDIT_CAPACITY))
        self._enabled_env = enabled_env
        self._global_env = global_env
        self._candidates: dict[str, OpCandidate] = {}
        self._static: dict[Hashable, str] = {}
        self._bench: dict[Hashable, str] = {}
        self._override: str | None = None
        self._audit: deque[OpDecision] = deque(maxlen=self._audit_capacity)
        self._lock = threading.RLock()

    # --------------------------------------------------------------- metadata

    @property
    def op(self) -> str:
        return self._op

    @property
    def default(self) -> str:
        return self._default

    def enabled(self) -> bool:
        """Master gate; default-off (unset env -> False).

        Not consulted by :meth:`select`: the runtime wiring owns the gate, the
        same split as ``cascade_gate.enabled`` / ``decision_for``.
        """
        return os.getenv(self._enabled_env) == "1"

    def override_env_var(self) -> str:
        """Per-operator override env name for this selector."""
        return ENV_OVERRIDE_TEMPLATE.format(op=_env_suffix(self._op))

    # --------------------------------------------------------------- registry

    def register(self, candidate: OpCandidate, *, replace: bool = False) -> OpCandidate:
        """Add a candidate; a duplicate name raises unless ``replace=True``."""
        if not isinstance(candidate, OpCandidate):
            raise TypeError("candidate must be an OpCandidate")
        if not candidate.name:
            raise ValueError("candidate name must be a non-empty string")
        with self._lock:
            if candidate.name in self._candidates and not replace:
                raise ValueError(f"candidate already registered: {candidate.name}")
            self._candidates[candidate.name] = candidate
        return candidate

    def unregister(self, name: str) -> None:
        with self._lock:
            self._candidates.pop(name, None)

    def get(self, name: str) -> OpCandidate | None:
        with self._lock:
            return self._candidates.get(name)

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._candidates)

    # ----------------------------------------------------------- decision data

    def set_static_table(self, table: Mapping[Hashable, str]) -> None:
        """Replace the declared shape/scene -> candidate table."""
        with self._lock:
            self._static.clear()
            self._static.update(table)

    def static_table(self) -> dict[Hashable, str]:
        with self._lock:
            return dict(self._static)

    def clear_static_table(self) -> None:
        with self._lock:
            self._static.clear()

    def set_bench_table(self, table: Mapping[Hashable, str]) -> None:
        """Replace the runtime-bench shape/scene -> candidate table."""
        with self._lock:
            self._bench.clear()
            self._bench.update(table)

    def bench_table(self) -> dict[Hashable, str]:
        with self._lock:
            return dict(self._bench)

    def clear_bench_table(self) -> None:
        with self._lock:
            self._bench.clear()

    def override(self, name: str) -> None:
        """Force ``name``; highest programmatic priority below the env knob."""
        with self._lock:
            self._override = name

    def clear_override(self) -> None:
        with self._lock:
            self._override = None

    def active_override(self) -> tuple[str | None, str]:
        """Return ``(name, source)`` of the winning override, else ``(None, "")``.

        The env knob beats the programmatic one: a deployment must be able to
        force a cell regardless of what the process set for itself.
        """
        env = self._env_override()
        if env is not None:
            return env, SOURCE_ENV
        with self._lock:
            if self._override is not None:
                return self._override, SOURCE_OVERRIDE
        return None, ""

    def _env_override(self) -> str | None:
        per_op = os.getenv(self.override_env_var())
        if per_op and per_op.strip():
            return per_op.strip()
        raw = os.getenv(self._global_env)
        if not raw:
            return None
        for part in raw.split(","):
            key, sep, value = part.partition("=")
            if sep and key.strip() == self._op and value.strip():
                return value.strip()
        return None

    # ------------------------------------------------------------------ select

    def select(self, ctx: OpContext | Mapping[str, Any]) -> str:
        """Chosen implementation name for ``ctx`` (records an audit entry)."""
        return self.explain(ctx).chosen

    def explain(self, ctx: OpContext | Mapping[str, Any]) -> OpDecision:
        """Full decision record for ``ctx`` (also appended to the audit ring)."""
        context = _coerce_ctx(ctx)
        decision = self._decide(context)
        with self._lock:
            self._audit.append(decision)
        return decision

    def _decide(self, ctx: OpContext) -> OpDecision:
        with self._lock:
            candidates = dict(self._candidates)
            static = dict(self._static)
            bench = dict(self._bench)
        key = self._key_fn(ctx)

        override, source = self.active_override()
        if override is not None:
            # Explicit override is authoritative: it applies or fails open,
            # it never silently reaches a *different* kernel.
            candidate = candidates.get(override)
            if candidate is None:
                return self._fallback(ctx, f"override_unknown:{override}")
            if not candidate.applicable(ctx):
                return self._fallback(ctx, f"override_inapplicable:{override}")
            return OpDecision(self._op, override, source, "override", ctx)

        candidate = candidates.get(bench.get(key, ""))
        if candidate is not None and candidate.applicable(ctx):
            return OpDecision(
                self._op, candidate.name, SOURCE_BENCH, "bench_table", ctx
            )

        candidate = candidates.get(static.get(key, ""))
        if candidate is not None and candidate.applicable(ctx):
            return OpDecision(
                self._op, candidate.name, SOURCE_STATIC, "static_table", ctx
            )

        hinted = self._min_cost_hint(candidates, ctx)
        if hinted is not None:
            return OpDecision(self._op, hinted, SOURCE_COST, "min_cost_hint", ctx)

        return self._fallback(ctx, "no_applicable_candidate")

    @staticmethod
    def _min_cost_hint(
        candidates: Mapping[str, OpCandidate], ctx: OpContext
    ) -> str | None:
        best: tuple[float, str] | None = None
        for name, candidate in candidates.items():
            if candidate.cost_hint is None or not candidate.applicable(ctx):
                continue
            ranked = (candidate.cost_hint, name)
            if best is None or ranked < best:
                best = ranked
        return None if best is None else best[1]

    def _fallback(self, ctx: OpContext, reason: str) -> OpDecision:
        return OpDecision(self._op, self._default, SOURCE_FALLBACK, reason, ctx)

    # ------------------------------------------------------------------- audit

    def audit(self) -> tuple[OpDecision, ...]:
        with self._lock:
            return tuple(self._audit)

    def last_decision(self) -> OpDecision | None:
        with self._lock:
            return self._audit[-1] if self._audit else None

    def audit_capacity(self) -> int:
        return self._audit_capacity

    def clear_audit(self) -> None:
        with self._lock:
            self._audit.clear()

    def reset_for_tests(self) -> None:
        """Clear registry, tables, override and audit (unit tests only)."""
        with self._lock:
            self._candidates.clear()
            self._static.clear()
            self._bench.clear()
            self._override = None
            self._audit.clear()


def _coerce_ctx(ctx: OpContext | Mapping[str, Any]) -> OpContext:
    if isinstance(ctx, OpContext):
        return ctx
    if isinstance(ctx, Mapping):
        return OpContext.from_mapping(ctx)
    raise TypeError(f"unsupported context type: {type(ctx).__name__}")


# ----------------------------------------------------- process-wide registry
# Extension point for the runtime wiring: one selector per operator name.

_registry: dict[str, OpSelector] = {}
_registry_lock = threading.Lock()


def get_selector(op: str, default: str = DEFAULT_HOST, **kwargs: Any) -> OpSelector:
    """Return the process-wide selector for ``op``, creating it on first use."""
    with _registry_lock:
        selector = _registry.get(op)
        if selector is None:
            selector = OpSelector(op, default, **kwargs)
            _registry[op] = selector
        return selector


def registered_selectors() -> tuple[str, ...]:
    with _registry_lock:
        return tuple(_registry)


def reset_registry_for_tests() -> None:
    """Drop every process-wide selector (unit tests only)."""
    with _registry_lock:
        _registry.clear()
