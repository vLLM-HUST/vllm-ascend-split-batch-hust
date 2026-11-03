# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

from __future__ import annotations

import os

os.environ["VLLM_HUST_GDN_SPLIT_BA"] = "1"

from vllm_ascend_split_batch.gdn_split_ba import gdn_split_ba_fused


def optimized_op(ba):
    return gdn_split_ba_fused(ba)
