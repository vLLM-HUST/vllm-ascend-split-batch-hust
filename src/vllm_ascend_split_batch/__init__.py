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

"""Split-batch planning contracts and inert runtime metadata."""


def __getattr__(name):
    if name in {
        "DualPadPlan",
        "SplitBatchConfig",
        "SplitSlice",
        "plan_dual_pad",
        "precheck_reason",
    }:
        from . import planner

        value = getattr(planner, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class VllmAscendSplitBatchContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


__all__ = [
    "DualPadPlan",
    "SplitBatchConfig",
    "SplitSlice",
    "VllmAscendSplitBatchContractProposal",
    "plan_dual_pad",
    "precheck_reason",
]
