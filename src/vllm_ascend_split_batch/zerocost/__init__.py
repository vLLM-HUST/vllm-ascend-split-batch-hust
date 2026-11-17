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

"""Bundle marker module for org.vllm-hust.zerocost-wiring.

This package only exists so the extension manager can discover the bundle
manifest (``vllm-hust-extension-v0.2.json``).  It is not an implementation
entry: the runtime entry point is ``zerocost_wiring:load``
(``vllm_ascend_split_batch.zerocost_wiring``), and importing this package
enables no capability -- both wiring features stay off until their own env vars
are set.
"""
