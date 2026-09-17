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

"""MLP token-chunked execution（default-off）——降低激活峰值，逐位等价。

**动机**（本机实测，见 `bench/runs/20260916-abcert-b1/A1-A3-CHUNK-FIX.md`）：
MLP 的 `gate_up`（2I·2B = 54 KiB/token）与 `npu_swiglu` 的输出（I·2B = 27 KiB/token）
**同时存活**——因为 `torch_npu.npu_swiglu` 无 `out=` 参数、总是新分配
（`AscendSiluAndMul.forward_oot` 调它）。于是 MLP 工作集 = 81 KiB/token。
按 token 维切块顺序执行，工作集降为 `(n/k)·81`，而每个 token 的数学与其它 token 无关。

**实测收益**（tokens=32768，fp16，H=5120，I=13824）：
| k | peak | vs k=1 | 逐位相等 | 中位耗时 |
|---|---|---|---|---|
| 1 | 3977 MiB | — | True | 53.26 ms |
| 2 | 2521 MiB | **−1.42 GiB** | True | 53.59 ms (**+0.6%**) |
| 4 | 1793 MiB | **−2.13 GiB** | True | 53.26 ms (−0.0%) |

⇒ 用于解 A1/A3 的 KV 缺口（−0.48 / −0.62 GiB），余量 2–3×。

**为什么不做成默认开**：它改变执行粒度（多若干次 kernel launch）。虽然实测无代价，
但属"改动数值路径之外的行为"，按仓库纪律保持 default-off，由使用者显式开启。

**为什么不改宿主**：`AGENTS.md` 硬性约束禁改 `vllm-hust/`；本载体用
`vllm.general_plugins` 入口包装 `Qwen2MLP.forward`，宿主可随时被替换而不影响本文件。

**合规**：不改任何冻结 CLI 参数/环境变量/`cudagraph_mode`/`splitting_ops`/`gmu`/
`max_model_len`/`max_num_batched_tokens` ⇒ 表附-8 的"生效值一致"不受影响。
"""

from __future__ import annotations

import os
import threading

#: 块数；``"1"`` 或未设 = 不启用。建议 2（收益/代价最优）或 4。
ENV_CHUNKS = "VLLM_HUST_MLP_CHUNKS"
#: 仅当本批 token 数 ≥ 该阈值才分块（decode 的 n 很小，分块无意义且徒增 launch）
ENV_MIN_TOKENS = "VLLM_HUST_MLP_CHUNK_MIN_TOKENS"

_LOCAL = threading.local()
_MARKER = "_vllm_hust_mlp_chunk_wrapped"
_DEFAULT_MIN_TOKENS = 256


def chunks() -> int:
    """返回块数（≥2 才生效；非法值按 1 处理）。"""
    try:
        k = int(os.getenv(ENV_CHUNKS, "1"))
    except ValueError:
        return 1
    return k if k >= 2 else 1


def min_tokens() -> int:
    try:
        return int(os.getenv(ENV_MIN_TOKENS, str(_DEFAULT_MIN_TOKENS)))
    except ValueError:
        return _DEFAULT_MIN_TOKENS


def install() -> bool:
    """包装 ``Qwen2MLP.forward``。幂等；任何失败都返回 False 且不改变宿主行为。"""
    if getattr(_LOCAL, "installing", False):
        return False
    _LOCAL.installing = True
    try:
        k = chunks()
        if k < 2:
            return False
        import torch

        from vllm.model_executor.models.qwen2 import Qwen2MLP

        orig = getattr(Qwen2MLP, "forward", None)
        if not callable(orig):
            return False
        if getattr(orig, _MARKER, False):
            return True
        thr = min_tokens()

        def forward_chunked(self, x):
            n = x.shape[0]
            if n < thr or self.act_fn.__class__.__name__ not in ("SiluAndMul", "AscendSiluAndMul"):
                return orig(self, x)
            out = torch.empty_like(x)
            c = (n + k - 1) // k
            for i in range(k):
                lo, hi = i * c, min((i + 1) * c, n)
                g, _ = self.gate_up_proj(x[lo:hi])
                s = self.act_fn(g)
                o, _ = self.down_proj(s)
                out[lo:hi] = o
                del g, s, o
            return out

        setattr(forward_chunked, _MARKER, True)
        forward_chunked.__name__ = getattr(orig, "__name__", "forward")
        forward_chunked.__doc__ = getattr(orig, "__doc__", None)
        Qwen2MLP.forward = forward_chunked
        return True
    except Exception:  # noqa: BLE001 -- load() 绝不能破坏引擎
        return False
    finally:
        _LOCAL.installing = False


def load() -> bool:
    """``vllm.general_plugins`` 入口（default-off，仅 ``VLLM_HUST_MLP_CHUNKS>=2`` 时启用）。"""
    return install()
