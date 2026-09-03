# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""CPU-only tests for the W2 cascade gate decision logic (no device)."""

import pytest

gate = pytest.importorskip("vllm_ascend_split_batch.cascade_gate")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(gate.ENV_GATE, raising=False)
    monkeypatch.delenv(gate.ENV_OVERRIDE, raising=False)
    gate.reset_for_tests()
    yield
    gate.reset_for_tests()


def _fill(decisions, buckets):
    gate._decisions.update(decisions)
    gate._prefix_buckets[:] = sorted(buckets)


def test_disabled_env_defaults_to_cascade_on(monkeypatch):
    monkeypatch.setenv(gate.ENV_GATE, "1")
    assert gate.enabled() is True
    # empty decision table -> neutral (cascade-on)
    assert gate.decision_for(4096, 32) is True


def test_bucket_lookup_picks_largest_leq():
    gate._decisions.update({
        (32, 4096): False,   # measured loss corner
        (32, 8192): True,
        (64, 4096): True,
    })
    gate._prefix_buckets[:] = [4096, 8192]
    assert gate.decision_for(4300, 32) is False   # falls in the 4096 bucket
    assert gate.decision_for(8200, 32) is True    # 8192 bucket
    assert gate.decision_for(9500, 32) is True    # still the 8192 bucket
    assert gate.decision_for(4300, 64) is True    # benched on
    assert gate.decision_for(16384, 128) is True  # unbenched -> default on


def test_below_smallest_bucket_defaults_on():
    _fill({(32, 4096): False}, [4096])
    # below the smallest benched prefix the core cascade gate is not engaged
    # anyway; the decision defaults to cascade-on (conservative).
    assert gate.decision_for(1024, 32) is True


def test_zero_shared_len_is_neutral():
    _fill({(32, 4096): False}, [4096])
    assert gate.decision_for(0, 32) is True


def test_override_wins_over_table(monkeypatch):
    _fill({(32, 4096): False}, [4096])
    monkeypatch.setenv(gate.ENV_OVERRIDE, "on")
    assert gate.override() == "on"
    assert gate.decision_for(4300, 32) is True
    monkeypatch.setenv(gate.ENV_OVERRIDE, "off")
    assert gate.decision_for(8200, 64) is False
    # invalid override values are ignored
    monkeypatch.setenv(gate.ENV_OVERRIDE, "bogus")
    assert gate.override() is None
    assert gate.decision_for(4300, 32) is False


def test_prefix_grid():
    assert gate._prefix_grid(4096, 20480) == [4096, 8192, 16384]
    assert gate._prefix_grid(8192, 16384) == [8192]   # 2x capped by half len
    assert gate._prefix_grid(20480, 20480) == []      # min prefix too large
