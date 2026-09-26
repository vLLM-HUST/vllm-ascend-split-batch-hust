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

"""CPU-only tests for the operator dispatch skeleton (pure, no torch).

Coverage contract:
- registry: register / duplicate / replace / unregister / names;
- ladder: cost_hint < static table < bench table < programmatic override < env
  override, with fail-open below all of them;
- fail-open: no candidate, inapplicable predicate, predicate that raises, and
  an override naming an unknown/inapplicable candidate all fall back to the
  host path;
- audit ring: bounded, ordered, carries source/reason/context;
- default-off: ``enabled()`` is False with the env unset;
- concurrency: coarse-grained registration from many threads loses nothing.
"""

import threading

import pytest

from vllm_ascend_split_batch import op_selector as os_mod

OP = "demo_op"
PER_OP_ENV = "VLLM_ASCEND_OP_SELECT_DEMO_OP"
CTX = os_mod.OpContext(
    num_tokens=64, shared_len=8192, headdim=128, dtype="bf16", scene="joint"
)


def _ctx(**overrides) -> os_mod.OpContext:
    payload = CTX.as_dict()
    payload.update(overrides)
    return os_mod.OpContext(**payload)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(os_mod.ENV_ENABLE, raising=False)
    monkeypatch.delenv(os_mod.ENV_OVERRIDE, raising=False)
    monkeypatch.delenv(PER_OP_ENV, raising=False)
    os_mod.reset_registry_for_tests()
    yield
    os_mod.reset_registry_for_tests()


@pytest.fixture
def selector() -> os_mod.OpSelector:
    return os_mod.OpSelector(OP, default=os_mod.DEFAULT_HOST, audit_capacity=8)


def _cand(name, cost=None, applicable=True, label=""):
    predicate = (lambda ctx: True) if applicable is True else (lambda ctx: applicable)
    return os_mod.OpCandidate(
        name=name, is_applicable=predicate, cost_hint=cost, label=label
    )


# --------------------------------------------------------------------- registry


def test_register_and_names(selector):
    selector.register(_cand("native", cost=2.0))
    selector.register(_cand("kernel", cost=1.0))
    assert set(selector.names()) == {"native", "kernel"}
    assert selector.get("kernel").cost_hint == 1.0


def test_duplicate_registration_raises_unless_replaced(selector):
    selector.register(_cand("kernel", cost=1.0))
    with pytest.raises(ValueError):
        selector.register(_cand("kernel", cost=0.5))
    selector.register(_cand("kernel", cost=0.5), replace=True)
    assert selector.get("kernel").cost_hint == 0.5


def test_unregister_is_idempotent(selector):
    selector.register(_cand("kernel", cost=1.0))
    selector.unregister("kernel")
    selector.unregister("kernel")
    assert selector.names() == ()


def test_register_rejects_non_candidate(selector):
    with pytest.raises(TypeError):
        selector.register("kernel")


# ------------------------------------------------------------------------ ladder


def test_lowest_cost_hint_wins_among_applicable(selector):
    selector.register(_cand("native", cost=3.0))
    selector.register(_cand("kernel", cost=1.0))
    decision = selector.explain(CTX)
    assert decision.chosen == "kernel"
    assert decision.source == os_mod.SOURCE_COST
    assert decision.reason == "min_cost_hint"


def test_cost_hint_skips_inapplicable_candidates(selector):
    selector.register(_cand("fast_but_wrong", cost=0.1, applicable=False))
    selector.register(_cand("kernel", cost=1.0))
    assert selector.select(CTX) == "kernel"


def test_cost_hint_tie_is_broken_by_name(selector):
    selector.register(_cand("bbb", cost=1.0))
    selector.register(_cand("aaa", cost=1.0))
    assert selector.select(CTX) == "aaa"


def test_candidates_without_hint_never_win_the_cost_tier(selector):
    selector.register(_cand("no_opinion", cost=None))
    assert selector.select(CTX) == os_mod.DEFAULT_HOST


def test_static_table_beats_cost_hint(selector):
    selector.register(_cand("native", cost=3.0))
    selector.register(_cand("kernel", cost=1.0))
    selector.set_static_table({CTX.key(): "native"})
    decision = selector.explain(CTX)
    assert decision.chosen == "native"
    assert decision.source == os_mod.SOURCE_STATIC


def test_bench_table_beats_static_table(selector):
    selector.register(_cand("native", cost=3.0))
    selector.register(_cand("kernel", cost=1.0))
    selector.set_static_table({CTX.key(): "native"})
    selector.set_bench_table({CTX.key(): "kernel"})
    decision = selector.explain(CTX)
    assert decision.chosen == "kernel"
    assert decision.source == os_mod.SOURCE_BENCH


def test_stale_table_entry_falls_through(selector):
    """A table entry naming a dropped/inapplicable candidate is ignored."""
    selector.register(_cand("native", cost=3.0))
    selector.set_bench_table({CTX.key(): "gone"})
    assert selector.select(CTX) == "native"


def test_tables_are_keyed_by_scene_and_shape(selector):
    selector.register(_cand("native", cost=2.0))
    selector.register(_cand("kernel", cost=1.0))
    selector.set_static_table({_ctx(scene="other").key(): "native"})
    # A differently-scened context misses the table and lands on cost_hint.
    assert selector.select(CTX) == "kernel"


def test_custom_key_fn_is_honoured():
    selector = os_mod.OpSelector(
        OP, default="host", key_fn=lambda ctx: (ctx.num_tokens, ctx.shared_len)
    )
    selector.register(_cand("native", cost=2.0))
    selector.register(_cand("kernel", cost=1.0))
    selector.set_bench_table({(64, 8192): "native"})
    assert selector.select(CTX) == "native"
    assert selector.select(_ctx(num_tokens=8)) == "kernel"


# ----------------------------------------------------------------- fail-open


def test_empty_registry_falls_open_to_host(selector):
    decision = selector.explain(CTX)
    assert decision.chosen == os_mod.DEFAULT_HOST
    assert decision.source == os_mod.SOURCE_FALLBACK
    assert decision.reason == "no_applicable_candidate"


def test_no_applicable_candidate_falls_open(selector):
    selector.register(_cand("kernel", cost=1.0, applicable=False))
    assert selector.select(CTX) == os_mod.DEFAULT_HOST


def test_raising_predicate_is_treated_as_inapplicable(selector):
    def boom(ctx):
        raise RuntimeError("predicate bug")

    selector.register(
        os_mod.OpCandidate(name="broken", is_applicable=boom, cost_hint=0.0)
    )
    selector.register(_cand("kernel", cost=1.0))
    assert selector.select(CTX) == "kernel"


def test_mapping_context_is_accepted(selector):
    selector.register(_cand("kernel", cost=1.0))
    assert selector.select(CTX.as_dict()) == "kernel"


def test_unknown_mapping_keys_and_coercion(selector):
    ctx = os_mod.OpContext.from_mapping(
        {"num_tokens": "128", "scene": None, "unused": 7}
    )
    assert ctx.num_tokens == 128
    assert ctx.scene == ""
    assert ctx.shared_len == 0


# -------------------------------------------------------------------- override


def test_programmatic_override_beats_both_tables(selector):
    selector.register(_cand("native", cost=3.0))
    selector.register(_cand("kernel", cost=1.0))
    selector.set_bench_table({CTX.key(): "native"})
    selector.override("kernel")
    decision = selector.explain(CTX)
    assert decision.chosen == "kernel"
    assert decision.source == os_mod.SOURCE_OVERRIDE


def test_env_override_beats_programmatic_override(monkeypatch, selector):
    selector.register(_cand("native", cost=3.0))
    selector.register(_cand("kernel", cost=1.0))
    selector.override("native")
    assert selector.override_env_var() == PER_OP_ENV
    monkeypatch.setenv(PER_OP_ENV, "kernel")
    decision = selector.explain(CTX)
    assert decision.chosen == "kernel"
    assert decision.source == os_mod.SOURCE_ENV


def test_global_env_override_selects_by_op(monkeypatch, selector):
    selector.register(_cand("native", cost=3.0))
    selector.register(_cand("kernel", cost=1.0))
    monkeypatch.setenv(os_mod.ENV_OVERRIDE, "other=zzz,demo_op=kernel")
    assert selector.select(CTX) == "kernel"


def test_global_env_override_ignores_other_ops(monkeypatch, selector):
    selector.register(_cand("native", cost=3.0))
    selector.register(_cand("kernel", cost=1.0))
    monkeypatch.setenv(os_mod.ENV_OVERRIDE, "another_op=zzz")
    # The entry does not match this op, so the normal cost ladder applies.
    decision = selector.explain(CTX)
    assert decision.chosen == "kernel"
    assert decision.source == os_mod.SOURCE_COST


def test_override_naming_unknown_candidate_falls_open(selector):
    selector.register(_cand("kernel", cost=1.0))
    selector.override("typo")
    decision = selector.explain(CTX)
    assert decision.chosen == os_mod.DEFAULT_HOST
    assert decision.source == os_mod.SOURCE_FALLBACK
    assert decision.reason == "override_unknown:typo"


def test_override_naming_inapplicable_candidate_falls_open(selector):
    selector.register(_cand("kernel", cost=1.0))
    selector.register(_cand("small_only", applicable=False))
    selector.override("small_only")
    decision = selector.explain(CTX)
    assert decision.chosen == os_mod.DEFAULT_HOST
    assert decision.reason == "override_inapplicable:small_only"


def test_clear_override_restores_the_ladder(selector):
    selector.register(_cand("kernel", cost=1.0))
    selector.override("host_forced")
    selector.clear_override()
    assert selector.select(CTX) == "kernel"
    assert selector.active_override() == (None, "")


# ----------------------------------------------------------------------- audit


def test_audit_records_every_decision(selector):
    selector.register(_cand("kernel", cost=1.0))
    selector.select(CTX)
    selector.select(_ctx(num_tokens=8))
    records = selector.audit()
    assert len(records) == 2
    assert records[0].op == OP
    assert records[0].context.num_tokens == 64
    assert records[1].context.num_tokens == 8


def test_audit_ring_is_bounded_and_keeps_the_newest(selector):
    selector.register(_cand("kernel", cost=1.0))
    for num_tokens in range(20):
        selector.select(_ctx(num_tokens=num_tokens))
    assert len(selector.audit()) == 8
    assert selector.audit_capacity() == 8
    assert selector.audit()[-1].context.num_tokens == 19
    assert selector.last_decision().context.num_tokens == 19


def test_clear_audit_keeps_the_registry(selector):
    selector.register(_cand("kernel", cost=1.0))
    selector.select(CTX)
    selector.clear_audit()
    assert selector.audit() == ()
    assert selector.select(CTX) == "kernel"


def test_reset_for_tests_clears_everything(selector):
    selector.register(_cand("kernel", cost=1.0))
    selector.set_static_table({CTX.key(): "kernel"})
    selector.set_bench_table({CTX.key(): "kernel"})
    selector.override("kernel")
    selector.select(CTX)
    selector.reset_for_tests()
    assert selector.names() == ()
    assert selector.static_table() == {}
    assert selector.bench_table() == {}
    assert selector.active_override() == (None, "")
    assert selector.audit() == ()


# ------------------------------------------------------------------ default-off


def test_enabled_is_default_off(selector):
    assert selector.enabled() is False


def test_enabled_reads_the_master_gate(monkeypatch, selector):
    monkeypatch.setenv(os_mod.ENV_ENABLE, "1")
    assert selector.enabled() is True
    monkeypatch.setenv(os_mod.ENV_ENABLE, "0")
    assert selector.enabled() is False


def test_select_does_not_consult_the_master_gate(monkeypatch, selector):
    """The runtime wiring owns the gate, exactly like cascade_gate."""
    monkeypatch.setenv(os_mod.ENV_ENABLE, "0")
    selector.register(_cand("kernel", cost=1.0))
    assert selector.select(CTX) == "kernel"


# ------------------------------------------------------------------ registry


def test_get_selector_is_a_singleton_per_op():
    first = os_mod.get_selector(OP)
    second = os_mod.get_selector(OP)
    assert first is second
    assert os_mod.registered_selectors() == (OP,)


# ---------------------------------------------------------------- concurrency


def test_concurrent_registration_loses_no_candidate(selector):
    total = 64
    barrier = threading.Barrier(8)

    def worker(base: int) -> None:
        barrier.wait()
        for i in range(base, total, 8):
            selector.register(_cand(f"cand_{i}", cost=float(i)))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(selector.names()) == total
    assert selector.select(CTX) == "cand_0"


def test_concurrent_select_keeps_the_ring_bounded(selector):
    selector.register(_cand("kernel", cost=1.0))

    def worker() -> None:
        for _ in range(50):
            selector.select(CTX)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(selector.audit()) == selector.audit_capacity()
