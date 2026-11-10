# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

from .base import OpCase
from .dynamic_chain import DynamicVectorChainCase
from .gdn_input_norm import GdnInputNormDecodeCase
from .gdn_norm_gate import GdnNormGateCase
from .gdn_output_norm_fusion import GdnOutputNormFusionCase
from .gdn_qkv_norm import GdnQkvNormCase
from .gdn_split_ba import GdnSplitBaCase
from .moe_gate_merge import MoeGateMergeCase
from .moe_gating_routing import MoeGatingRoutingCase
from .moe_shared_epilogue import MoeSharedEpilogueDecodeCase

CASE_REGISTRY: dict[str, type[OpCase]] = {
    MoeSharedEpilogueDecodeCase.name: MoeSharedEpilogueDecodeCase,
    MoeGatingRoutingCase.name: MoeGatingRoutingCase,
    MoeGateMergeCase.name: MoeGateMergeCase,
    GdnInputNormDecodeCase.name: GdnInputNormDecodeCase,
    GdnNormGateCase.name: GdnNormGateCase,
    GdnOutputNormFusionCase.name: GdnOutputNormFusionCase,
    GdnSplitBaCase.name: GdnSplitBaCase,
    GdnQkvNormCase.name: GdnQkvNormCase,
}


def get_case(case_name: str) -> OpCase:
    if case_name in CASE_REGISTRY:
        return CASE_REGISTRY[case_name]()

    # Support dynamic vector chain cases synthesized by timeline insight
    if case_name.startswith("dynamic_chain_"):
        pattern_str = case_name.removeprefix("dynamic_chain_")
        # Strip trailing _r\d+ if present
        import re

        clean_pat = re.sub(r"_r\d+$", "", pattern_str).replace("_", " -> ")
        mock_card = {
            "pattern_signature": clean_pat,
            "anchor_before": {
                "type": "MatMul",
                "in_shape": "[32, 2048]",
                "out_shape": "[32, 4096]",
            },
            "anchor_after": {
                "type": "MatMul",
                "in_shape": "[32, 4096]",
                "out_shape": "[32, 2048]",
            },
            "operators": [
                {"in_shape": "[32, 4096]", "type": p.strip()}
                for p in clean_pat.split("->")
            ],
            "avg_chain_dur_us": 15.0,
            "rank": 1,
        }
        return DynamicVectorChainCase(mock_card)

    raise KeyError(
        f"Unknown case {case_name}. Available cases: {list(CASE_REGISTRY.keys())}"
    )


__all__ = [
    "CASE_REGISTRY",
    "DynamicVectorChainCase",
    "OpCase",
    "get_case",
]
