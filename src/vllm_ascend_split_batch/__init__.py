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

"""Split-batch planning contracts and inert runtime metadata.

The public names are resolved lazily through :func:`__getattr__` (one lookup
source: ``__all__``), so importing this package never imports ``planner``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .planner import (
        DualPadPlan,
        SplitBatchConfig,
        SplitSlice,
        plan_dual_pad,
        precheck_reason,
    )

__all__ = [
    "DualPadPlan",
    "SplitBatchConfig",
    "SplitSlice",
    "plan_dual_pad",
    "precheck_reason",
]


def __getattr__(name: str):
    """Resolve one public name from ``planner`` on first access."""
    if name in __all__:
        from . import planner

        return getattr(planner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
