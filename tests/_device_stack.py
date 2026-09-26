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

"""Import guard for the tests that need the Ascend device stack.

The suite targets the NPU host, where ``torch`` / ``torch_npu`` / ``vllm-ascend``
are importable.  On a plain CPU runner (the org CI job installs only
``.[test]``) they are not -- and pytest >= 8.2 **re-raises** an ``ImportError``
raised *inside* an importable module instead of skipping it, so importing a
carrier module there is a collection ERROR, not a skip:

    ERROR tests/test_cascade_plugin.py - ImportError: ... torch ...

Importing this module first turns that into a clean, visible skip:

    import _device_stack  # noqa: F401  -- device stack present, or skip
    from vllm_ascend_split_batch import cascade_plugin

Objects needed by the importing test can be bound directly
(``from _device_stack import torch, torch_npu``).  On the NPU host all three
imports succeed, so this module changes nothing there.
"""

import pytest

torch = pytest.importorskip("torch", reason="device tests need torch")
torch_npu = pytest.importorskip(
    "torch_npu", reason="device tests need the torch_npu plugin"
)
vllm_ascend = pytest.importorskip(
    "vllm_ascend", reason="the carriers patch vllm-ascend internals"
)
F = torch.nn.functional
