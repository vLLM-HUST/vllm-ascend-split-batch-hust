# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from .base import OpCase
from .gdn_input_norm import GdnInputNormDecodeCase
from .gdn_norm_gate import GdnNormGateCase
from .gdn_output_norm_fusion import GdnOutputNormFusionCase
from .gdn_qkv_norm import GdnQkvNormCase
from .gdn_split_ba import GdnSplitBaCase
from .moe_shared_epilogue import MoeSharedEpilogueDecodeCase

CASE_REGISTRY: dict[str, type[OpCase]] = {
    MoeSharedEpilogueDecodeCase.name: MoeSharedEpilogueDecodeCase,
    GdnInputNormDecodeCase.name: GdnInputNormDecodeCase,
    GdnNormGateCase.name: GdnNormGateCase,
    GdnOutputNormFusionCase.name: GdnOutputNormFusionCase,
    GdnSplitBaCase.name: GdnSplitBaCase,
    GdnQkvNormCase.name: GdnQkvNormCase,
}


def get_case(case_name: str) -> OpCase:
    if case_name not in CASE_REGISTRY:
        raise KeyError(
            f"Unknown case {case_name}. Available cases: {list(CASE_REGISTRY.keys())}"
        )
    return CASE_REGISTRY[case_name]()


__all__ = [
    "CASE_REGISTRY",
    "OpCase",
    "get_case",
]
