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

"""Skip device-stack tests when required packages are unavailable."""

import pytest

torch = pytest.importorskip("torch", reason="device tests need torch")
torch_npu = pytest.importorskip(
    "torch_npu", reason="device tests need the torch_npu plugin"
)
vllm_ascend = pytest.importorskip(
    "vllm_ascend", reason="the carriers patch vllm-ascend internals"
)
F = torch.nn.functional
