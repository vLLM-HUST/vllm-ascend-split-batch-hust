# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Zero-cost wiring ②: do not materialise rope cos/sin buffers nothing reads.

``vllm_ascend/ops/rotary_embedding.update_cos_sin`` (called every step) is pure
materialisation work.  This module makes it *lazy*: the wrapper only records the
positions, and ``get_cos_and_sin_slice`` materialises on demand before handing
the buffers out.  A reader therefore still sees exactly the buffers it would
have seen; it just does not pay for them when nobody reads.

Deferral is exactly equivalent for the known readers in
:data:`ROPE_READER_MODULES`: both call ``update_cos_sin(positions)`` and read
``get_cos_and_sin_slice()`` in the same step.  Presence of a reader *module* is
deliberately NOT used as a gate -- the host imports every patch module
unconditionally (``vllm_ascend/patch/worker/__init__.py``), so module presence
says nothing about whether this chip/model reads the buffers, and gating on it
silently disabled this feature on A2 (measured: 0 engagement, kernel counts
unchanged).
"""

from __future__ import annotations

import sys

from .zerocost_state import (
    ROTARY_MODULE,
    ROTARY_READ,
    ROTARY_UPDATE,
    _import_host,
    _iter_other_modules,
    _orig,
    _stats,
)

#: modules that *do* read the rope buffers; documented here because ② is only
#: sound while their read pattern is "update then read in the same step" (they
#: are not a gate -- see the module docstring).
ROPE_READER_MODULES = (
    "vllm_ascend._310p.ops.rotary_embedding",
    "vllm_ascend.patch.worker.patch_minimax_m2",
)

#: last positions handed to the lazy rope wrapper (per process).
_cos_sin_positions = None


def install_skip_cos_sin() -> None:
    """Install ②. Raises on a moved anchor; the caller rolls the feature back."""
    module = _import_host(ROTARY_MODULE)
    update = getattr(module, ROTARY_UPDATE)
    reader = getattr(module, ROTARY_READ)

    def _lazy_update_cos_sin(positions):
        """Record the positions; materialise only if something reads them."""
        global _cos_sin_positions
        _cos_sin_positions = positions
        _stats["cos_sin_deferred"] += 1
        return None

    def _materialising_read():
        """Hand out the buffers, materialising them first when deferred."""
        global _cos_sin_positions
        if _cos_sin_positions is not None:
            positions, _cos_sin_positions = _cos_sin_positions, None
            _stats["cos_sin_materialized"] += 1
            update(positions)
        return reader()

    module.__dict__[ROTARY_UPDATE] = _lazy_update_cos_sin
    module.__dict__[ROTARY_READ] = _materialising_read
    _orig[f"global:{ROTARY_MODULE}.{ROTARY_UPDATE}"] = (module, ROTARY_UPDATE, update)
    _orig[f"global:{ROTARY_MODULE}.{ROTARY_READ}"] = (module, ROTARY_READ, reader)

    # rebind every module that already imported the symbols by name
    for mod in _iter_other_modules(module):
        if getattr(mod, ROTARY_UPDATE, None) is update:
            setattr(mod, ROTARY_UPDATE, _lazy_update_cos_sin)
            _orig[f"global:{mod.__name__}.{ROTARY_UPDATE}"] = (
                mod,
                ROTARY_UPDATE,
                update,
            )
        if getattr(mod, ROTARY_READ, None) is reader:
            setattr(mod, ROTARY_READ, _materialising_read)
            _orig[f"global:{mod.__name__}.{ROTARY_READ}"] = (mod, ROTARY_READ, reader)


def restore_late_imports() -> None:
    """Undo rebinds made by modules imported after :func:`install_skip_cos_sin`."""
    module = sys.modules.get(ROTARY_MODULE)
    if module is None:
        return
    update = _orig.get(f"global:{ROTARY_MODULE}.{ROTARY_UPDATE}")
    reader = _orig.get(f"global:{ROTARY_MODULE}.{ROTARY_READ}")
    lazy = module.__dict__.get(ROTARY_UPDATE)
    materialise = module.__dict__.get(ROTARY_READ)
    for mod in _iter_other_modules(module):
        if update is not None and getattr(mod, ROTARY_UPDATE, None) is lazy:
            setattr(mod, ROTARY_UPDATE, update[2])
        if reader is not None and getattr(mod, ROTARY_READ, None) is materialise:
            setattr(mod, ROTARY_READ, reader[2])
