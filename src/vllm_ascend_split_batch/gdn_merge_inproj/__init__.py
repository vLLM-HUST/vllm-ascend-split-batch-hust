# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Bundle marker module for org.vllm-hust.gdn-merge-inproj.

The carrier itself lives in ``vllm_ascend_split_batch.gdn_merge_inproj_plugin``;
this package only carries the bundle manifest (``vllm-hust-extension-v0.2.json``).

The manifest keeps ``implementation[].status = import_only`` on purpose: the
carrier is default-off and currently **fail-closed** (the host has no safe seam
that consumes a merged ``in_proj_qkvz`` weight), so the extension manager must
refuse to enable it -- see ``HOST_CONTRACT.md`` (GDN section) and
``docs/support-matrix.md``.
"""
