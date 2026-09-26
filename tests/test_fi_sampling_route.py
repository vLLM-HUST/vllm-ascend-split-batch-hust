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

"""CPU-only tests for the fi_sampling routing logic (pure, no torch).

Coverage contract (W2b package):
- one case per fallback class of the five-entry fallback matrix;
- one case per dispatch cell of the routing table (argmax / api1 / joint /
  fork), with the bench-sourced thresholds pinned;
- default-off: the route helper itself never enables anything, and
  ``fi_sampling_plugin.load()`` is a no-op with the env unset (asserted in
  ``test_fi_sampling_plugin.py``).
"""

import pytest

from vllm_ascend_split_batch import fi_sampling_route as route

FALLBACKS = {
    "generators": {"has_generators": True},
    "processed_logits": {"logprobs_mode": "processed_logits"},
    "processed_logprobs": {"logprobs_mode": "processed_logprobs"},
    "batch_invariant": {"batch_invariant": True},
    "reduce_sample": {"reduce_sample": True},
    "async_exponential": {"async_exponential": True},
}


def _decide(**overrides) -> str:
    kwargs = {
        "logprobs_mode": "raw_logprobs",
        "has_generators": False,
        "batch_invariant": False,
        "reduce_sample": False,
        "async_exponential": False,
        "batch_size": 1,
        "has_top_k": False,
        "has_top_p": False,
        "top_k_min": None,
        "top_k_max": None,
    }
    kwargs.update(overrides)
    return route.decide_route(**kwargs)


@pytest.mark.parametrize("name", sorted(FALLBACKS))
def test_each_fallback_class_forces_the_fork_chain(name: str) -> None:
    """Fallback matrix: every class must route to the fork chain."""
    overrides = FALLBACKS[name]
    # A configuration that would otherwise take the FI api1 fast path.
    decision = _decide(batch_size=1024, **overrides)
    assert decision == route.ROUTE_FORK


def test_fallback_reason_reports_the_first_hit() -> None:
    assert (
        route.fallback_reason(
            logprobs_mode="raw_logprobs",
            has_generators=True,
            batch_invariant=True,
            reduce_sample=True,
            async_exponential=True,
        )
        == route.FALLBACK_GENERATORS
    )
    assert (
        route.fallback_reason(
            logprobs_mode="processed_logprobs",
            has_generators=False,
            batch_invariant=False,
            reduce_sample=False,
            async_exponential=False,
        )
        == route.FALLBACK_LOGPROBS_MODE
    )
    assert (
        route.fallback_reason(
            logprobs_mode="raw_logprobs",
            has_generators=False,
            batch_invariant=False,
            reduce_sample=False,
            async_exponential=False,
        )
        is None
    )


def test_untruncated_any_batch_takes_the_api1_kernel() -> None:
    """Main win region: vLLM's default (no top-k/top-p) random sampling."""
    for batch_size in (1, 8, 64, 256, 1024):
        assert _decide(batch_size=batch_size) == route.ROUTE_FI_API1


def test_joint_cell_uses_bench_thresholds() -> None:
    """k=50/p=0.95 wins at B>=256 with block_v=4096 (bench table B)."""
    assert (
        _decide(
            batch_size=route.FI_JOINT_MIN_BATCH,
            has_top_k=True,
            has_top_p=True,
            top_k_min=50,
            top_k_max=50,
        )
        == route.ROUTE_FI_JOINT
    )
    # One row below the measured threshold stays on the fork chain.
    assert (
        _decide(
            batch_size=route.FI_JOINT_MIN_BATCH - 1,
            has_top_k=True,
            has_top_p=True,
            top_k_min=50,
            top_k_max=50,
        )
        == route.ROUTE_FORK
    )
    assert route.FI_JOINT_BLOCK_V == 4096
    assert route.FI_JOINT_MIN_K == 32


def test_joint_cell_requires_medium_k() -> None:
    """k=8 loses at every B (bench) -> fork chain even at large batch."""
    assert (
        _decide(
            batch_size=1024,
            has_top_k=True,
            has_top_p=True,
            top_k_min=8,
            top_k_max=8,
        )
        == route.ROUTE_FORK
    )
    assert (
        _decide(
            batch_size=1024,
            has_top_k=True,
            has_top_p=True,
            top_k_min=route.FI_JOINT_MIN_K,
            top_k_max=64,
        )
        == route.ROUTE_FI_JOINT
    )


def test_k1_rows_route_to_argmax() -> None:
    """k=1 (approximate greedy) is the worst cell for both chains."""
    for batch_size in (1, 64, 1024):
        assert (
            _decide(
                batch_size=batch_size,
                has_top_k=True,
                has_top_p=False,
                top_k_min=1,
                top_k_max=1,
            )
            == route.ROUTE_ARGMAX
        )


def test_mixed_top_k_tensor_is_not_treated_as_k1() -> None:
    """Only an all-rows k=1 batch is greedy; mixed k must not hit argmax."""
    assert (
        _decide(
            batch_size=1024,
            has_top_k=True,
            has_top_p=False,
            top_k_min=1,
            top_k_max=50,
        )
        == route.ROUTE_FORK
    )


def test_unmeasured_cells_stay_on_the_fork_chain() -> None:
    """k-only / p-only large-batch cells are outside the reviewed bench."""
    assert (
        _decide(
            batch_size=1024,
            has_top_k=True,
            has_top_p=False,
            top_k_min=50,
            top_k_max=50,
        )
        == route.ROUTE_FORK
    )
    assert _decide(batch_size=1024, has_top_p=True) == route.ROUTE_FORK


def test_thresholds_are_overridable_for_the_e2e_ab() -> None:
    assert (
        route.decide_route(
            logprobs_mode="raw_logprobs",
            has_generators=False,
            batch_invariant=False,
            reduce_sample=False,
            async_exponential=False,
            batch_size=64,
            has_top_k=True,
            has_top_p=True,
            top_k_min=50,
            top_k_max=50,
            joint_min_k=50,
            joint_min_batch=64,
        )
        == route.ROUTE_FI_JOINT
    )


def test_missing_top_k_values_do_not_take_fi_routes() -> None:
    """A top-k tensor whose extremes could not be read stays on the fork chain."""
    assert (
        _decide(
            batch_size=1024,
            has_top_k=True,
            has_top_p=True,
            top_k_min=None,
            top_k_max=None,
        )
        == route.ROUTE_FORK
    )


def test_needs_top_k_read_matches_the_reachable_cells() -> None:
    """The device->host read is taken only where an FI cell can fire.

    The read costs a per-step host sync (measured +4.9% median TPOT in the
    first api1 pair), so a cell that cannot reach the joint/k=1 branches must
    not pay for it.
    """
    assert route.needs_top_k_read(batch_size=256, joint_min_batch=256)
    assert route.needs_top_k_read(batch_size=512, joint_min_batch=256)
    assert not route.needs_top_k_read(batch_size=64, joint_min_batch=256)
    assert not route.needs_top_k_read(batch_size=1, joint_min_batch=256)
