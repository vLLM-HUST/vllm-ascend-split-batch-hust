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

"""Guard for the tests that assert the legacy cascade kernel wheel is *usable*.

The cascade bundle belongs to the archived 0.25-line cohort.  Its native kernel
(``ascend_kernel``, the ``[kernels]`` extra) is a soft dependency and is not
installed on the 0.28 base, and it cannot simply be installed either: the
published wheels are built against glibc >= 2.38 while the 0.28 image base is
Ubuntu 22.04.5 / glibc 2.35, so ``import ascend_kernel`` dies inside
``torch.ops.load_library`` with::

    OSError: .../lib/libascend_kernel.so: version `GLIBC_2.38' not found

(measured 2026-10-13; same fact recorded in the kernel workspace
``ops/kernels/AGENTS.md`` section 6, ruling: keep cascade archived on the 0.28
stack and let the plugin fail open to full-KV.)

Two host states must therefore **skip** instead of failing:

* not installed -> ``ModuleNotFoundError``;
* installed but base-incompatible -> ``OSError`` from the native loader.

``pytest.importorskip`` only tolerates ``ImportError``, so the second case is
handled explicitly here.  Note that this guard is only for the handful of tests
that assert the wheel-present path; the fail-open contract tests in
``test_cascade_fail_open.py`` must keep running *without* the wheel and must not
call it.
"""

import pytest

# Mirrors ``cascade_plugin._probe_kernel_wheel()``'s success criterion.  Kept as
# a plain import check (rather than calling the plugin probe) so the guard has
# no side effect on the plugin's once-only probe/warning state.
_REGISTERED_OPS = ("fa_fp32_stage1", "lse_merge")


def require_cascade_kernel() -> None:
    """Skip the calling test unless the cascade kernel is importable *here*."""
    try:
        import ascend_kernel  # noqa: F401  -- registers the ops below
    except (ImportError, OSError) as exc:
        pytest.skip(f"legacy cascade kernel wheel unusable on this host: {exc}")

    import torch

    missing = [op for op in _REGISTERED_OPS if not hasattr(torch.ops.npu, op)]
    if missing:
        pytest.skip(
            "ascend_kernel imported but torch.ops.npu lacks: " + ", ".join(missing)
        )
