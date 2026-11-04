# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import os

os.environ["VLLM_HUST_GDN_NORM_GATE"] = "1"

from vllm_ascend_split_batch.gdn_norm_gate import gdn_norm_gate_fused


def optimized_op(x, z, weight):
    return gdn_norm_gate_fused(x, z, weight)
