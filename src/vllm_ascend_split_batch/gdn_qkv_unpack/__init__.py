# Copyright (c) 2026 vLLM-HUST contributors.
#
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

"""``gdn_qkv_unpack`` -- single-pass packed qkv unpack for GDN (triton-ascend).

Host entry ``rearrange_mixed_qkv`` (``api``), kernel in ``kernels`` (triton,
imported on the first call), golden in ``reference``.  Not wired to any plugin
entry point or bundle manifest.

The exports are lazy: importing this package pulls neither ``torch`` nor
``triton``.
"""

from __future__ import annotations

import importlib

__all__ = ["rearrange_mixed_qkv", "rearrange_mixed_qkv_reference"]

_LAZY = {
    "rearrange_mixed_qkv": (".api", "rearrange_mixed_qkv"),
    "rearrange_mixed_qkv_reference": (".reference", "rearrange_mixed_qkv_reference"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(importlib.import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
