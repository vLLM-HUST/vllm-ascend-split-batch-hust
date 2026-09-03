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


def test_bucket_lookup_ceils_to_smallest_geq():
    gate._decisions.update({
        (32, 4096): False,   # measured loss corner
        (32, 8192): True,
        (64, 4096): True,
    })
    gate._prefix_buckets[:] = [4096, 8192]
    # block-aligned runtime shared lengths sit just below the nominal bucket
    # (e.g. 7936 = 62 blocks for a ~8.2k shared prefix): ceil keeps them in
    # the larger bucket whose measured verdict reflects their true regime.
    assert gate.decision_for(4096, 32) is False   # exactly the 4096 bucket
    assert gate.decision_for(4300, 32) is True    # ceils past 4096 -> 8192
    assert gate.decision_for(7936, 32) is True    # ceils to the 8192 bucket
    assert gate.decision_for(9500, 32) is True    # 8192 bucket
    assert gate.decision_for(4300, 64) is True    # benched on
    assert gate.decision_for(16384, 128) is True  # unbenched -> default on


def test_below_smallest_bucket_follows_that_bucket():
    _fill({(32, 4096): False}, [4096])
    # cascade steps only trigger at shared >= MIN_PREFIX (= the smallest
    # bucket), so sub-bucket values never occur at a real decision point;
    # the ceil mapping simply attributes them to the smallest bucket.
    assert gate.decision_for(1024, 32) is False


def test_zero_shared_len_is_neutral():
    _fill({(32, 4096): False}, [4096])
    assert gate.decision_for(0, 32) is True


def test_above_largest_bucket_defaults_on():
    _fill({(32, 8192): False}, [4096, 8192])
    # above the largest benched prefix cascade keeps winning -> default on
    assert gate.decision_for(20000, 32) is True


def test_override_wins_over_table(monkeypatch):
    _fill({(32, 4096): False, (64, 4096): True}, [4096])
    monkeypatch.setenv(gate.ENV_OVERRIDE, "on")
    assert gate.override() == "on"
    assert gate.decision_for(4300, 32) is True
    monkeypatch.setenv(gate.ENV_OVERRIDE, "off")
    assert gate.decision_for(4300, 64) is False
    # invalid override values are ignored
    monkeypatch.setenv(gate.ENV_OVERRIDE, "bogus")
    assert gate.override() is None
    assert gate.decision_for(4096, 32) is False


def test_prefix_grid():
    assert gate._prefix_grid(4096, 20480) == [4096, 8192, 16384]
    assert gate._prefix_grid(8192, 16384) == [8192]   # 2x capped by half len
    assert gate._prefix_grid(20480, 20480) == []      # min prefix too large
