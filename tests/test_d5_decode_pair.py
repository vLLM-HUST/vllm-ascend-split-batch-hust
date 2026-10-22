# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest

from vllm_ascend_split_batch.d5_decode_pair import (
    eligible,
    min_step,
    min_tokens,
    require_disjoint,
    split_decode,
)

torch = pytest.importorskip("torch")


@dataclass
class Conv:
    query_start_loc: object
    cache_indices: object
    initial_state_mode: object = None


@dataclass
class Decode:
    causal_conv1d: Conv
    actual_seq_lengths: object


def metadata():
    offsets = torch.arange(5, dtype=torch.int32)
    slots = torch.tensor([1, 2, 4, 5], dtype=torch.int32)
    return NS(
        num_prefills=0,
        num_spec_decodes=0,
        num_decodes=4,
        num_decode_tokens=4,
        num_actual_tokens=4,
        spec_sequence_masks=None,
        non_spec_query_start_loc=offsets,
        non_spec_state_indices_tensor=slots,
        non_spec_decode_metadata=Decode(
            Conv(offsets, slots), torch.tensor([0, 1, 1, 1, 1])
        ),
    )


def test_split_rebases_and_owns_metadata():
    original = metadata()
    a, b = split_decode(original, 0, 2), split_decode(original, 2, 4)
    require_disjoint(a, b)
    assert b.non_spec_query_start_loc.tolist() == [0, 1, 2]
    assert b.non_spec_state_indices_tensor.tolist() == [4, 5]
    assert b.non_spec_decode_metadata.actual_seq_lengths.tolist() == [0, 1, 1]
    b.non_spec_query_start_loc[0] = 9
    b.non_spec_state_indices_tensor[0] = 9
    assert original.non_spec_query_start_loc.tolist() == [0, 1, 2, 3, 4]
    assert original.non_spec_state_indices_tensor.tolist() == [1, 2, 4, 5]


@pytest.mark.parametrize("field", ["num_prefills", "num_spec_decodes"])
def test_non_decode_is_not_eligible(field):
    m = metadata()
    setattr(m, field, 1)
    assert not eligible(m, 4)
    with pytest.raises(ValueError, match="pure"):
        split_decode(m, 0, 2)


def test_shared_state_rejected():
    m = metadata()
    with pytest.raises(ValueError, match="shared"):
        require_disjoint(split_decode(m, 0, 3), split_decode(m, 2, 4))


def test_reserved_state_rejected():
    m = metadata()
    m.non_spec_state_indices_tensor[0] = 0
    with pytest.raises(ValueError, match="reserved"):
        require_disjoint(split_decode(m, 0, 2), split_decode(m, 2, 4))


def test_invalid_slice_rejected():
    with pytest.raises(ValueError, match="invalid"):
        split_decode(metadata(), 2, 2)


def test_two_dim_conv_cache_indices_disjoint():
    m = metadata()
    m.non_spec_decode_metadata.causal_conv1d.cache_indices = torch.tensor(
        [[10, 20], [11, 21], [12, 22], [13, 23]], dtype=torch.int32
    )
    a, b = split_decode(m, 0, 2), split_decode(m, 2, 4)
    require_disjoint(a, b)
    assert b.non_spec_decode_metadata.causal_conv1d.cache_indices.tolist() == [
        [12, 22],
        [13, 23],
    ]


def test_two_dim_conv_cache_indices_shared_rejected():
    m = metadata()
    m.non_spec_decode_metadata.causal_conv1d.cache_indices = torch.tensor(
        [[10, 20], [11, 21], [11, 21], [13, 23]], dtype=torch.int32
    )
    with pytest.raises(ValueError, match="shared"):
        require_disjoint(split_decode(m, 0, 2), split_decode(m, 2, 4))


def test_min_tokens_env_threshold(monkeypatch):
    monkeypatch.setenv("D5_PAIR_MIN_TOKENS", "32")
    assert min_tokens() == 32


def test_min_tokens_floors_and_ignores_invalid(monkeypatch):
    monkeypatch.delenv("D5_PAIR_MIN_TOKENS", raising=False)
    assert min_tokens() == 2
    monkeypatch.setenv("D5_PAIR_MIN_TOKENS", "bogus")
    assert min_tokens() == 2
    monkeypatch.setenv("D5_PAIR_MIN_TOKENS", "1")
    assert min_tokens() == 2


def test_min_step_env_threshold(monkeypatch):
    monkeypatch.setenv("D5_PAIR_MIN_STEP", "8")
    assert min_step() == 8


def test_min_step_floors_and_ignores_invalid(monkeypatch):
    monkeypatch.delenv("D5_PAIR_MIN_STEP", raising=False)
    assert min_step() == 2
    monkeypatch.setenv("D5_PAIR_MIN_STEP", "zero")
    assert min_step() == 2
    monkeypatch.setenv("D5_PAIR_MIN_STEP", "-3")
    assert min_step() == 2
