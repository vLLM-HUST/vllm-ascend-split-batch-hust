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
* VLLM_HUST_MOE_GATE_MERGE   -> vllm.model_executor.models.qwen3_next（MoE router 与
  shared_expert_gate 合并成一次 GEMM; default-off, 消费者不全即 fail-closed 不安装）
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)

ENV_D4 = "VLLM_HUST_D4_CONV_PACK"
ENV_GATE_MERGE = "VLLM_HUST_MOE_GATE_MERGE"
ENV_VIEWS = "VLLM_HUST_GDN_UNPACK_VIEWS"
ENV_INIT_TRITON = "VLLM_HUST_MOE_INIT_TRITON"
ENV_NZ_SKIP = "VLLM_HUST_NZ_SKIP_GATE"

GDN = "vllm_ascend.ops.gdn"
TD = "vllm_ascend.ops.fused_moe.token_dispatcher"
LIN = "vllm_ascend.ops.linear"
QWEN3NEXT = "vllm.model_executor.models.qwen3_next"

# ---------------------------------------------------------------- gdn: D4 pack
_D4_TRANSFORMS = [
    (
        "class AscendGatedDeltaNetAttention(GatedDeltaNetAttention, HpcModule):",
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
        """        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if spec_sequence_masks is not None:""",
        """        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        _packed_conv_w = getattr(self.conv1d, _PACKED_CONV_WEIGHT_NAME, None)
        conv_weights_T = (
            _packed_conv_w
            if _packed_conv_w is not None
            else conv_weights.transpose(0, 1)
        )
        if spec_sequence_masks is not None:""",
        1,
    ),
    (
        """        if spec_sequence_masks is not None:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0""",
        """        if spec_sequence_masks is not None:
            activation_num = 1 if self.activation else 0""",
        1,
    ),
    (
        """                if get_pcp_group().world_size > 1:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0""",
        """                if get_pcp_group().world_size > 1:
                    activation_num = 1 if self.activation else 0""",
        1,
    ),
    (
        """                else:
                    conv_weights_T = conv_weights.transpose(0, 1)
                    activation_num = 1 if self.activation else 0""",
        """                else:
                    activation_num = 1 if self.activation else 0""",
        1,
    ),
    (
        """        elif attn_metadata.num_decodes > 0:
            conv_weights_T = conv_weights.transpose(0, 1)
            activation_num = 1 if self.activation else 0""",
        """        elif attn_metadata.num_decodes > 0:
            activation_num = 1 if self.activation else 0""",
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
        """            output_non_spec = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                output_non_spec,
                mixed_qkv_non_spec,
                conv_weights_T,
                conv_state=self_kv_cache[0],
                bias_opt=self.conv1d.bias,
                query_start_loc_opt=non_spec_query_start_loc_device,""",
        """            if _UNPACK_VIEWS_ON:
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
                query_start_loc_opt=non_spec_query_start_loc_device,""",
        1,
    ),
    (
        """        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)""",
        """        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        if mixed_qkv_non_spec is None:
            query_non_spec, key_non_spec, value_non_spec = None, None, None
        elif (not _UNPACK_VIEWS_ON) or attn_metadata.num_prefills != 0:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)
        elif mixed_qkv_non_spec.is_contiguous() and mixed_qkv_non_spec.shape[0] == 1:
            query_non_spec, key_non_spec, value_non_spec = _unpack_mixed_qkv_views(
                mixed_qkv_non_spec, self
            )
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(mixed_qkv_non_spec)""",
        1,
    ),
    (
        """        # 2. Recurrent attention""",
        """        # 2. Recurrent attention
        # (views helper 依赖 conv 输出为连续分配; 见 _UNPACK_VIEWS_ON 分支)""",
        1,
    ),
    (
        """from vllm_ascend.ops.triton.mamba.causal_conv1d import extract_last_width""",
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
        """        sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,""",
        """        if hidden_states.shape[0] == 1 and hidden_states.is_contiguous():
            from vllm_ascend_split_batch.moe_init_triton import (
                moe_init_routing_t1 as _moe_init_routing_t1,)

            sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = _moe_init_routing_t1(
                hidden_states, topk_ids, global_num_experts)
        else:
            sorted_hidden_states, expanded_row_idx, expert_tokens, dynamic_scale = DeviceOperator.npu_moe_init_routing(
            hidden_states,
            topk_ids,""",
        1,
    ),
    (
        """            quant_mode=quant_mode,
            act_quant_type=act_quant_type,
        )
        expert_tokens = expert_tokens.to(torch.int64)""",
        """            quant_mode=quant_mode,
            act_quant_type=act_quant_type,
            )
        expert_tokens = expert_tokens.to(torch.int64)""",
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
        """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        keep_nd_weight = _should_keep_nd_for_compatibility_weight(layer.weight.data)""",
        """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        if os.environ.get("VLLM_HUST_NZ_SKIP_GATE") == "1" and str(
            getattr(layer, "prefix", "")
        ).endswith(".shared_expert_gate"):
            # NZ mode2 配置域 a2 例外: 标量门 [1,2048] 转 NZ 触发
            # aclnnMatmulWeightNz EZ1001(n=1/k=1); 官方逐层豁免属性
            layer.skip_weight_nz_conversion = True
        keep_nd_weight = _should_keep_nd_for_compatibility_weight(layer.weight.data)""",
        1,
    ),
]

# env 门 -> (模块名, 变换表); 顺序表: views 依赖 d4 的注入行
ENV_DBO = "VLLM_HUST_DBO_ENABLE"
PLATFORM = "vllm_ascend.platform"
CORE_RUNNER = "vllm.v1.worker.gpu_model_runner"
NPU_RUNNER = "vllm_ascend.worker.model_runner_v1"

# DBO 解锁(2026-10-22 调研窗立项, 见 docs/comm-overlap-dbo-feasibility.md):
# 平台重置放行 + TP2/DP1 decode 触发 + NPU load_model 的 UBatchWrapper 包装。
# 核心侧无 os import, 用 __import__("os") 内联。
_DBO_PLATFORM_TRANSFORMS = [
    (
        """        if getattr(vllm_config.parallel_config, "enable_dbo", False):
            logger.warning(
                "Parameter is currently ignored on Ascend. parameter=enable_dbo, action: resetting to False. "
            )
            vllm_config.parallel_config.enable_dbo = False""",
        """        if getattr(vllm_config.parallel_config, "enable_dbo", False) and os.environ.get("VLLM_HUST_DBO_ENABLE") != "1":
            logger.warning(
                "Parameter is currently ignored on Ascend. parameter=enable_dbo, action: resetting to False. "
            )
            vllm_config.parallel_config.enable_dbo = False""",
        1,
    ),
]
_DBO_CORE_TRIGGER_TRANSFORMS = [
    (
        """        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:""",
        """        should_ubatch, num_tokens_across_dp = False, None
        if (self.vllm_config.parallel_config.enable_dbo and uniform_decode
                and __import__("os").environ.get("VLLM_HUST_DBO_ENABLE") == "1"):
            should_ubatch = True
        elif self.vllm_config.parallel_config.data_parallel_size > 1:""",
        1,
    ),
]
_DBO_NPU_WRAP_TRANSFORMS = [
    (
        """            elif cudagraph_mode.has_full_cudagraphs():
                self.model = ACLGraphWrapper(
                    self.model,
                    self.vllm_config,
                    runtime_mode=CUDAGraphMode.FULL,
                    use_eagle=self.use_eagle,
                    enable_enpu=self.enable_enpu,
                )""",
        """            elif cudagraph_mode.has_full_cudagraphs():
                if __import__("os").environ.get("VLLM_HUST_DBO_ENABLE") == "1":
                    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
                    self.model = UBatchWrapper(
                        self.model,
                        self.vllm_config,
                        CUDAGraphMode.FULL,
                        self.device,
                    )
                else:
                    self.model = ACLGraphWrapper(
                        self.model,
                        self.vllm_config,
                        runtime_mode=CUDAGraphMode.FULL,
                        use_eagle=self.use_eagle,
                        enable_enpu=self.enable_enpu,
                    )""",
        1,
    ),
]

# ------------------------------------------------- MoE 双门合并 (router + shared gate)
# 语义: MoE 输入 x 同时喂 self.gate(2048→256) 与 self.shared_expert_gate(2048→1),
# 故 cat([W_router, W_shared], 0) = 257×2048 一次 GEMM + 两路 contiguous 切片成立。
# 安装点在 load_weights 返回之后（control-plane, 不在被 torch.compile 的 forward 里）:
# 此时权重已加载、代理在首次 forward 前就位, 且不会把 Python 副作用带进图捕获。
# 载体实现见 vllm_ascend_split_batch.moe_gate_merge。
_GATE_MERGE_TRANSFORMS = [
    (
        "from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP",
        "from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP\n"
        "from vllm_ascend_split_batch.moe_gate_merge import (\n"
        "    install_gate_merge_all as _hust_install_gate_merge_all,\n"
        ")",
        1,
    ),
    (
        """        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)""",
        """        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        _hust_install_gate_merge_all(self)
        return loaded""",
        2,
    ),
]

TABLES = [
    (ENV_D4, GDN, _D4_TRANSFORMS),
    (ENV_VIEWS, GDN, _VIEWS_TRANSFORMS),
    (ENV_INIT_TRITON, TD, _INIT_TRITON_TRANSFORMS),
    (ENV_NZ_SKIP, LIN, _NZ_SKIP_TRANSFORMS),
    (ENV_DBO, PLATFORM, _DBO_PLATFORM_TRANSFORMS),
    (ENV_DBO, CORE_RUNNER, _DBO_CORE_TRIGGER_TRANSFORMS),
    (ENV_DBO, NPU_RUNNER, _DBO_NPU_WRAP_TRANSFORMS),
    (ENV_GATE_MERGE, QWEN3NEXT, _GATE_MERGE_TRANSFORMS),
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
    """对目标模块源码文本施锚点变换后编译; 失败即原样编译（stock）。

    重写 get_code（而非 source_to_code）: SourceFileLoader 的字节码缓存与
    stock 源 mtime/size 匹配时会短路, 变换永远不会发生——必须绕过缓存。
    """

    def __init__(self, fullname, path, transforms):
        super().__init__(fullname, path)
        self._transforms = transforms

    def get_code(self, fullname):
        src_path = self.get_filename(fullname)
        source = self.get_data(src_path).decode("utf-8")
        try:
            patched, log = apply_transforms(source, self._transforms)
        except AnchorError as exc:
            logger.warning(
                "runtime_source_patches: %s anchor mismatch -> stock (%s)",
                src_path,
                exc,
            )
            return super().get_code(fullname)
        for entry in log:
            logger.info("runtime_source_patches: %s %s", src_path, entry)
        return compile(patched, src_path, "exec", dont_inherit=True)


class _Finder:
    def __init__(self, targets: dict[str, list]):
        self._targets = targets

    def find_spec(self, fullname, path=None, target=None):
        transforms = self._targets.get(fullname)
        if transforms is None:
            return None
        self._targets.pop(fullname, None)  # 一次性: 命中后即卸钩
        parent_name, _, sub_name = fullname.rpartition(".")
        parent = sys.modules.get(parent_name)
        parent_path = list(getattr(parent, "__path__", []) or [])
        if not parent_path:
            logger.warning("runtime_source_patches: %s 父包未就绪 -> stock", fullname)
            return None
        spec = importlib.machinery.PathFinder.find_spec(sub_name, parent_path)
        if spec is None or not getattr(spec, "origin", None):
            logger.warning(
                "runtime_source_patches: %s 未定位到源文件 -> stock", fullname
            )
            return None
        return importlib.util.spec_from_file_location(
            fullname,
            spec.origin,
            loader=_PatchingLoader(fullname, spec.origin, transforms),
        )


_INSTALLED = False


def install() -> bool:
    """env 门控注册 meta-path 钩子; 无相关 env 时不注册（零开销）。幂等。"""
    global _INSTALLED
    if _INSTALLED:
        return True
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
    _INSTALLED = True
    logger.info(
        "runtime_source_patches: armed for %s (fail-closed anchors)", sorted(targets)
    )
    return True


def load() -> bool:
    """vllm general_plugins 兼容入口（default-off）。"""
    return install()
