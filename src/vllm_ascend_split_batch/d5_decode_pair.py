# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from copy import copy


def eligible(metadata, tokens):
    return (
        metadata is not None
        and metadata.num_prefills == 0
        and metadata.num_spec_decodes == 0
        and metadata.num_decodes == tokens
        and metadata.num_decode_tokens == tokens
        and metadata.num_actual_tokens == tokens
        and tokens >= 2
        and metadata.spec_sequence_masks is None
    )


def split_decode(metadata, start, stop):
    from dataclasses import replace

    n = stop - start
    if not eligible(metadata, metadata.num_actual_tokens):
        raise ValueError("requires pure single-token non-spec decode")
    if not 0 <= start < stop <= metadata.num_actual_tokens:
        raise ValueError("invalid request slice")
    result = copy(metadata)
    result.num_decodes = result.num_decode_tokens = result.num_actual_tokens = n
    offsets = metadata.non_spec_query_start_loc[start : stop + 1].clone()
    result.non_spec_query_start_loc = offsets - offsets[0].clone()
    result.non_spec_state_indices_tensor = metadata.non_spec_state_indices_tensor[
        start:stop
    ].clone()
    decode = metadata.non_spec_decode_metadata
    conv = replace(
        decode.causal_conv1d,
        query_start_loc=result.non_spec_query_start_loc,
        cache_indices=decode.causal_conv1d.cache_indices[start:stop].clone(),
    )
    lengths = decode.actual_seq_lengths[: n + 1].clone()
    lengths.zero_()
    lengths[1:] = 1
    result.non_spec_decode_metadata = replace(
        decode, causal_conv1d=conv, actual_seq_lengths=lengths
    )
    return result


def require_disjoint(a, b):
    for field in ("non_spec_state_indices_tensor",):
        left = set(getattr(a, field).reshape(-1).tolist())
        right = set(getattr(b, field).reshape(-1).tolist())
        if left & right or 0 in left | right or min(left | right) < 0:
            raise ValueError("shared or reserved GDN state slot")
    left = set(a.non_spec_decode_metadata.causal_conv1d.cache_indices.tolist())
    right = set(b.non_spec_decode_metadata.causal_conv1d.cache_indices.tolist())
    if left & right or 0 in left | right or min(left | right) < 0:
        raise ValueError("shared or reserved conv state slot")
