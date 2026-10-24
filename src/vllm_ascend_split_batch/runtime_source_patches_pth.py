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

""".pth 入口: 解释器启动期注册源码变换钩子(早于一切业务 import)。

GDN 解包钩子默认注册，模型/形状在执行点门控；不在启动期导入 torch 或宿主。
其它变换需显式 env 开启。install() 幂等，site-packages 入口不改运行树。
"""

try:
    from .runtime_source_patches import install as _install

    _install()
except Exception:  # noqa: BLE001 - pth 阶段绝不阻塞解释器
    pass
