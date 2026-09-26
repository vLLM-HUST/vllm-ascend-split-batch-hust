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

"""Import guard for the tests that need the triton-ascend kernel stack.

``tests/_device_stack.py`` covers ``torch`` / ``torch_npu`` / ``vllm_ascend``;
the fi_gelu kernel tests additionally import ``numpy`` and ``triton`` (the
patched triton that ships with triton-ascend -- never installable from PyPI on
the CPU CI runner).  Importing this module skips the importing test module when
any piece is missing, with the missing name in the skip reason:

    from _kernel_stack import np, torch, torch_npu, triton  # device + kernels

On the NPU host every import succeeds, so this module changes nothing there.
"""

import pytest
from _device_stack import torch, torch_npu  # noqa: F401  -- re-exported

np = pytest.importorskip("numpy", reason="kernel probes need numpy")
triton = pytest.importorskip("triton", reason="kernel tests need triton-ascend")
