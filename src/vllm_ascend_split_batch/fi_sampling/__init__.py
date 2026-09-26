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

"""Vendored ``fi_sampling`` host API (flashinfer sampling port, triton-ascend).

Vendored from the W2 package ``knowledge/surveys/sampling/fi_sampling``
(worktree source of truth, reviewed and approved 2026-09-09; file hashes and
anchors in ``PROVENANCE.md``).  Only the *host* modules are vendored --
``api.py`` / ``kernels.py`` / ``npu_env.py`` / ``pure.py``; the reference
simulator and the test/bench harnesses stay in the source package, which
remains the authority for statistical-equivalence evidence.

Import path change vs the source package: relative imports instead of
top-level ``fi_sampling.*`` (so the vendored copy lives inside the plugin
package).  No algorithmic change.

Default-off: this package is imported only from
``fi_sampling_plugin._load_fi_api`` on the first FI-routed call; with
``VLLM_HUST_FI_SAMPLING`` unset nothing here is imported (and therefore
``triton`` is never pulled in).
"""

from .api import (
    sampling_from_probs,
    top_k_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
    top_p_sampling_from_probs,
)

__all__ = [
    "sampling_from_probs",
    "top_k_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
    "top_p_sampling_from_probs",
]
__version__ = "0.1.0"
