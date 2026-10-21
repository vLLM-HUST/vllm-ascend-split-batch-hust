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

"""运行树源码变换的持久化载波（C 轮: overlay 重灌免疫）。

背景: /vllm-workspace 运行树会被重灌（2026-10-21 17:04/17:20/17:26 三次实测）,
直接落在树里的补丁易失。本模块把三个已验收/在档补丁表达为**锚点变换表**, 在
宿主模块 import 前由 meta-path 钩子对源码文本施变换 —— 运行树保持 stock, 重灌
后下次起服自动重放。

fail-closed 语义:
* 每条变换 (anchor, replacement, count) 断言精确命中; 任一失配 -> 该模块整体
  回退 stock（不产出半变换源码）, 并打 warning（锚点漂移 = 上游改了该文件）。
* 目标模块已在 sys.modules（迟注册）-> 跳过并 warning（不打补丁的 stock 行为）。
* 全部载波 default-off, 由各自既有 env 门控制（与本树内版本同名同语义）。

表内容与 2026-10-21 运行树补丁一一对应:
* VLLM_HUST_D4_CONV_PACK   -> vllm_ascend.ops.gdn（conv 权重连续分配+消费, D4 site2）
* VLLM_HUST_GDN_UNPACK_VIEWS -> vllm_ascend.ops.gdn（decode T=1 零拷贝视图; 依赖上者）
* VLLM_HUST_MOE_INIT_TRITON  -> vllm_ascend.ops.fused_moe.token_dispatcher（decode T=1 triton init）
* VLLM_HUST_NZ_SKIP_GATE     -> vllm_ascend.ops.linear（NZ mode2 标量门豁免）
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

ENV_D4 = "VLLM_HUST_D4_CONV_PACK"
ENV_VIEWS = "VLLM_HUST_GDN_UNPACK_VIEWS"
ENV_INIT_TRITON = "VLLM_HUST_MOE_INIT_TRITON"
ENV_NZ_SKIP = "VLLM_HUST_NZ_SKIP_GATE"

GDN = "vllm_ascend.ops.gdn"
TD = "vllm_ascend.ops.fused_moe.token_dispatcher"
LIN = "vllm_ascend.ops.linear"

# ---------------------------------------------------------------- gdn: D4 pack
_D4_TRANSFORMS = [
    (
        'class AscendGatedDeltaNetAttention(GatedDeltaNetAttention, HpcModule):',
        '''_PACKED_CONV_WEIGHT_NAME = "ascend_conv1d_weight_packed"
_PACK_CONV_ENV = "VLLM_HUST_D4_CONV_PACK"


def _pack_conv_weights(conv1d) -> bool:
    """D4 site-2: materialise conv1d.weight [D,1,W] as contiguous [W,D] once."""
    source = conv1d.weight
    if source.is_meta:
        return False
    packed = (
        source.view(source.size(0), source.size(2))
        .transpose(0, 1)
        .contiguous()
    )
    if _PACKED_CONV_WEIGHT_NAME in conv1d._parameters:
        conv1d._parameters[_PACKED_CONV_WEIGHT_NAME].data = packed
    else:
        conv1d.register_parameter(
            _PACKED_CONV_WEIGHT_NAME,
            torch.nn.Parameter(packed, requires_grad=False),
        )
    return True


class AscendGatedDeltaNetAttention(GatedDeltaNetAttention, HpcModule):''',
        1,
    ),
    (
        '''        print(f"[W3F2v2] hook: env={_MERGE_GDN_INPROJ} gqa={getattr(self, 'gqa_interleaved_layout', None)} "''',
        '''        if os.environ.get(_PACK_CONV_ENV) == "1":
            try:
                _pack_conv_weights(self.conv1d)
                print("[D4site2] packed conv weight [W,D] ready", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[D4site2] pack skipped: {_e!r}", flush=True)
        print(f"[W3F2v2] hook: env={_MERGE_GDN_INPROJ} gqa={getattr(self, 'gqa_interleaved_layout', None)} "''',
        1,
    ),
    (
        '''        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:''',
        '''        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        _packed_conv_w = getattr(self.conv1d, _PACKED_CONV_WEIGHT_NAME, None)
        conv_weights_T = (
            _packed_conv_w
            if _packed_conv_w is not None
            else conv_weights.transpose(0, 1)
        )
        if spec_sequence_masks is not None:''',
        1,
    ),
    (
        '''        if spec_sequence_masks is not None:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0''',
        '''        if spec_sequence_masks is not None:
            activation_num = 1 if self.activation else 0''',
        1,
    ),
    (
        '''                if get_pcp_group().world_size > 1:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0''',
        '''                if get_pcp_group().world_size > 1:
                    activation_num = 1 if self.activation else 0''',
        1,
    ),
    (
        '''                else:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0''',
        '''                else:
                    activation_num = 1 if self.activation else 0''',
        1,
    ),
    (
        '''        elif attn_metadata.num_decodes > 0:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0''',
        '''        elif attn_metadata.num_decodes > 0:
            activation_num = 1 if self.activation else 0''',
        1,
    ),
]

# ---------------------------------------------------------------- gdn: views（依赖 D4 已施）
_VIEWS_TRANSFORMS = [
    (
        '_PACKED_CONV_WEIGHT_NAME = "ascend_conv1d_weight_packed"',
        '''_PACKED_CONV_WEIGHT_NAME = "ascend_conv1d_weight_packed"
_UNPACK_VIEWS_ENV = "VLLM_HUST_GDN_UNPACK_VIEWS"
_UNPACK_VIEWS_ON = os.environ.get(_UNPACK_VIEWS_ENV) == "1"''',
        1,
    ),
    (
        '''            output_non_spec = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                output_non_spec,
                mixed_qkv_non_spec,
                conv_weights_T,
                conv_state=self_kv_cache[0],
                bias_opt=self.conv1d.bias,
                query_start_loc_opt=non_spec_query_start_loc_device,''',
        '''            if _UNPACK_VIEWS_ON:
                output_non_spec = torch.empty(
                    mixed_qkv_non_spec.shape,
                    dtype=mixed_qkv_non_spec.dtype,
                    device=mixed_qkv_non_spec.device,
                )
            else:
                output_non_spec = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                output_non_spec,
                mixed_qkv_non_spec,
                conv_weights_T,
                conv_state=self_kv_cache[0],
                bias_opt=self.conv1d.bias,
                query_start_loc_opt=non_spec_query_start_loc_device,''',
        1,
    ),
    (
        '''        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)''',
        '''        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        if mixed_qkv_non_spec is None:
            query_non_spec, key_non_spec, value_non_spec = None, None, None
        elif (not _UNPACK_VIEWS_ON) or attn_metadata.num_prefills != 0:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)
        elif mixed_qkv_non_spec.is_contiguous() and mixed_qkv_non_spec.shape[0] == 1:
            query_non_spec, key_non_spec, value_non_spec = _unpack_mixed_qkv_views(
                mixed_qkv_non_spec, self
            )
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)''',
        1,
    ),
    (
        '''        # 2. Recurrent attention''',
        '''        # 2. Recurrent attention
        # (views helper 依赖 conv 输出为连续分配; 见 _UNPACK_VIEWS_ON 分支)''',
        1,
    ),
    (
        '''from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width''',
        '''from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width


def _unpack_mixed_qkv_views(mixed, layer):
    """ConcatD 消除: 行连续 mixed_qkv 的 q/k/v 直接切片视图（零拷贝）。"""
    qd = layer.key_dim // layer.tp_size
    vd = layer.value_dim // layer.tp_size
    s = mixed.shape[0]
    q = mixed[:, :qd].view(1, s, -1, layer.head_k_dim)
    k = mixed[:, qd:2 * qd].view(1, s, -1, layer.head_k_dim)
    v = mixed[:, 2 * qd:].view(1, s, -1, layer.head_v_dim)
    return q, k, v''',
        1,
    ),
]

# ---------------------------------------------------------------- dispatcher: triton init
_INIT_TRITON_TRANSFORMS = [
    (
        "from abc import ABC, abstractmethod",
        '''import os

import triton
import triton.language as tl

from abc import ABC, abstractmethod''',
        1,
    ),
    (
        "from abc import ABC, abstractmethod",
        '''from abc import ABC, abstractmethod


_MOE_INIT_TRITON_ON = os.environ.get("VLLM_HUST_MOE_INIT_TRITON") == "1"


@triton.jit
def _moe_init_routing_t1_kernel(
    x_ptr, ids_ptr,
    sorted_ptr, rowidx_ptr, counts_ptr,
    H: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
):
    kidx = tl.arange(0, K)
    ids = tl.load(ids_ptr + kidx).to(tl.int32)
    xrow = tl.load(x_ptr + tl.arange(0, H)).to(x_ptr.dtype.element_ty)
    sblock = tl.broadcast_to(xrow[None, :], (K, H))
    tl.store(sorted_ptr + kidx[:, None] * H + tl.arange(0, H)[None, :], sblock)
    iv = ids[:, None].to(tl.int32)
    less = tl.sum((ids[None, :] < iv).to(tl.int32), axis=1)
    sidx = kidx[None, :]
    ties_before = tl.sum(((ids[None, :] == iv) & (sidx < kidx[:, None])).to(tl.int32), axis=1)
    tl.store(rowidx_ptr + kidx, (less + ties_before).to(tl.int32))
    counts = tl.zeros((E,), tl.int32)
    for j in tl.static_range(K):
        e_j = tl.sum(tl.where(kidx == j, ids, 0), axis=0)
        counts += (tl.arange(0, E) == e_j.to(tl.int32)).to(tl.int32)
    tl.store(counts_ptr + tl.arange(0, E), counts.to(counts_ptr.dtype.element_ty))


def _moe_init_routing_t1(hidden_states, topk_ids, expert_num: int):
    k = topk_ids.shape[1]
    h = hidden_states.shape[1]
    sorted_t = torch.empty((k, h), dtype=hidden_states.dtype, device=hidden_states.device)
    rowidx = torch.empty((k,), dtype=torch.int32, device=hidden_states.device)
    counts = torch.zeros((expert_num,), dtype=torch.int64, device=hidden_states.device)
    _moe_init_routing_t1_kernel[(1,)](
        hidden_states, topk_ids, sorted_t, rowidx, counts,
        H=h, E=expert_num, K=k, num_warps=4,
    )
    return sorted_t, rowidx, counts, None''',
        1,
    ),
    (
        '''        sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,''',
        '''        if _MOE_INIT_TRITON_ON and hidden_states.shape[0] == 1 and hidden_states.is_contiguous():
            sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = _moe_init_routing_t1(
                hidden_states, topk_ids, global_num_experts)
        else:
            sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,''',
        1,
    ),
    (
        '''            quant_mode=quant_mode,
            act_quant_type=act_quant_type,
        )
        expert_tokens = expert_tokens.to(torch.int64)''',
        '''            quant_mode=quant_mode,
            act_quant_type=act_quant_type,
            )
        expert_tokens = expert_tokens.to(torch.int64)''',
        1,
    ),
]

# ---------------------------------------------------------------- linear: NZ skip
_NZ_SKIP_TRANSFORMS = [
    (
        "import torch\n",
        "import os\nimport torch\n",
        1,
    ),
    (
        '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        keep_nd_weight = _should_keep_nd_for_compatibility_weight(layer.weight.data)''',
        '''    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if os.environ.get("VLLM_HUST_NZ_SKIP_GATE") == "1" and str(
            getattr(layer, "prefix", "")
        ).endswith(".shared_expert_gate"):
            # NZ mode2 配置域 a2 例外: 标量门 [1,2048] 转 NZ 触发
            # aclnnMatmulWeightNz EZ1001(n=1/k=1); 官方逐层豁免属性
            layer.skip_weight_nz_conversion = True
        keep_nd_weight = _should_keep_nd_for_compatibility_weight(layer.weight.data)''',
        1,
    ),
]

# env 门 -> (模块名, 变换表); 顺序表: views 依赖 d4 的注入行
TABLES = [
    (ENV_D4, GDN, _D4_TRANSFORMS),
    (ENV_VIEWS, GDN, _VIEWS_TRANSFORMS),
    (ENV_INIT_TRITON, TD, _INIT_TRITON_TRANSFORMS),
    (ENV_NZ_SKIP, LIN, _NZ_SKIP_TRANSFORMS),
]


def apply_transforms(source: str, transforms) -> tuple[str, list[str]]:
    """施加变换表; 失配即抛 AnchorError（调用方回退 stock）。成功返回 (新源码, 命中日志)。"""
    log = []
    for anchor, replacement, count in transforms:
        found = source.count(anchor)
        if found != count:
            raise AnchorError(f"anchor x{found} (expect {count}): {anchor[:70]!r}")
        source = source.replace(anchor, replacement)
        log.append(f"ok x{count}: {anchor[:48]!r}")
    return source, log


class AnchorError(Exception):
    pass


class _PatchingLoader(importlib.machinery.SourceFileLoader):
    """对目标模块源码文本施锚点变换后编译; 失败即原样编译（stock）。"""

    def __init__(self, fullname, path, transforms):
        super().__init__(fullname, path)
        self._transforms = transforms

    def source_to_code(self, data, path, *, _path=None):
        source = data.decode("utf-8")
        try:
            patched, log = apply_transforms(source, self._transforms)
        except AnchorError as exc:
            logger.warning(
                "runtime_source_patches: %s anchor mismatch -> stock (%s)", path, exc)
            return super().source_to_code(data, path, _path=_path)
        for entry in log:
            logger.info("runtime_source_patches: %s %s", path, entry)
        return super().source_to_code(patched.encode("utf-8"), path, _path=_path)


class _Finder:
    def __init__(self, targets: dict[str, list]):
        self._targets = targets

    def find_spec(self, fullname, path=None, target=None):
        transforms = self._targets.get(fullname)
        if transforms is None:
            return None
        self._targets.pop(fullname, None)  # 一次性: 命中后即卸钩
        spec = importlib.util.find_spec(fullname)
        if spec is None or not getattr(spec, "origin", None):
            logger.warning("runtime_source_patches: %s 未定位到源文件 -> stock", fullname)
            return None
        spec.loader = _PatchingLoader(fullname, spec.origin, transforms)
        logger.info("runtime_source_patches: patched-loader installed for %s", fullname)
        return spec


def install() -> bool:
    """env 门控注册 meta-path 钩子; 无相关 env 时不注册（零开销）。"""
    targets: dict[str, list] = {}
    for env, module, transforms in TABLES:
        if os.environ.get(env) == "1":
            targets.setdefault(module, []).extend(transforms)
    if not targets:
        return False
    if any(m in sys.modules for m in targets):
        logger.warning(
            "runtime_source_patches: 目标模块 %s 已被 import, 迟注册不生效 -> stock",
            sorted(set(targets) & set(sys.modules)),
        )
        return False
    sys.meta_path.insert(0, _Finder(targets))
    logger.info(
        "runtime_source_patches: armed for %s (fail-closed anchors)", sorted(targets)
    )
    return True


def load() -> bool:
    """vllm general_plugins 兼容入口（default-off）。"""
    return install()
