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

import importlib

import pytest


def _importorskip(mod: str, reason: str):
    # importorskip only catches ImportError; device-less stacks can also fail
    # with SystemError/RuntimeError raised from C extension levels.
    try:
        return importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        pytest.skip(
            reason=f"{reason}: {type(e).__name__}: {e}", allow_module_level=True
        )
        raise  # unreachable; keeps type checkers happy


torch = _importorskip("torch", "device tests need torch")
torch_npu = _importorskip("torch_npu", "device tests need the torch_npu plugin")
vllm_ascend = _importorskip(
    "vllm_ascend", reason="the carriers patch vllm-ascend internals"
)
F = torch.nn.functional
