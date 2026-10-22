# Decode 通信重叠可行性侦察（DBO/ubatch，2026-10-22）

目标：吃掉 decode FULL graph 的 12.41ms/step 暴露通信（overlap=0，见
1021-182739 opstat 与 1022 D1 窗回执）。

## 接缝盘点（本栈 vllm 0.28 + vllm_ascend，文件级）

机器**在栈内**：
- `vllm/v1/worker/ubatching.py`：UBatchContext 双微批机制
  （comm_stream + compute_stream、线程屏障、GPU 事件，_NUM_UBATCHES=2）
- `vllm/v1/worker/gpu_ubatch_wrapper.py`、`ubatch_utils.py`
- 开关：`--enable-dbo`（num_ubatches=2）与 `--ubatch-size`（parallel.py:222/586）
- NPU 侧已接线：`vllm_ascend/worker/model_runner_v1.py:2243-2269`
  （should_ubatch → maybe_create_ubatch_slices，FULL 图 padding 分支在）

## 三层闸门（全关）

1. `vllm_ascend/platform.py:838-842`：**enable_dbo 被平台显式重置 False**
   （"Parameter is currently ignored on Ascend"）。
2. core `gpu_model_runner.py:4111-4122`：should_ubatch 仅 `data_parallel_size > 1`
   分支赋值——本 cohort TP2/DP=1 恒 False。
3. 模型侧差异：DBO 下 cascade attention 禁用（model_runner_v1:2232）等。

## 不确定性（须实验裁决）

- `torch.cuda.Stream`（ubatching.py 直接使用）经 torch_npu 兼容层是否落到 NPU 流
- 双流 + HCCL 在 FULL cudagraph capture 的行为（当前单流图内 allReduce 已工作）
- 双微批对 GDN/husk 载波的语义：moe_init_triton 的 M 语义、views 的
  contiguity 守卫、gate-fp32 注入（批切分是否保持各载波触发条件）

## 裁决与立项

立 probing 候选 `dbo-ascend-tp2-feasibility`。调研窗后静态深挖修正了解锁序列——
**实为 4 步，且②③已实现**（husk 变换，default-off `VLLM_HUST_DBO_ENABLE`）：

1. 平台重置放行（`_DBO_PLATFORM_TRANSFORMS`，platform.py reset 块加 env 门）
2. TP2/DP1 decode 触发（`_DBO_CORE_TRIGGER_TRANSFORMS`，core should_ubatch
   的 enable_dbo+uniform_decode 分支；core 无 os import 用 `__import__("os")`）
3. **NPU load_model 包装缺口**（深挖发现）：core 的 UBatchWrapper 包装分支
   （gpu_model_runner.py:5560，use_ubatching 独占分支，FULL 模式由 wrapper
   自管内层图）对 NPU 不可达——NPUModelRunner.load_model 全程自持无 super 调用。
   `_DBO_NPU_WRAP_TRANSFORMS` 在 ACLGraphWrapper 分支植入独占替代
   （env 门控时 UBatchWrapper 取代 ACLGraphWrapper，镜像 core 语义）。
4. 实证：boot + FULL graph + L3 8/8 + L1（对照 1022-023x standing 基线）。
   死刑条件不变：torch_npu 双流/HCCL 在该路径不成立。

补充约束（深挖新事实）：
- DBO 钩子消费者（dbo_enabled/dbo_register_recv_hook/make_ubatch_contexts）
  仅存在于 DeepEP 系 EP-MoE 内核（deepep_ht/ll/nixl/modular_kernel）——
  TP allReduce 无钩子层，重叠编排完全依赖 UBatchWrapper 自身双流循环，
  该假设是第 4 步的裁决点之一。
- 预算形态：2-3 个标准窗；收益天花板 = 12.4ms/step 的重叠份额。
- 载波环境互斥注意：DBO 探针不开 D4/views 载波（单变量；两者树内惰性补丁
  与 husk 通道并存无害）。

## 终章（已判死，2026-10-22）

实证起服死于 config 校验层最前断言：`Microbatching currently only supports
deepep_low_latency / deepep_high_throughput /
nixl_ep all2all backends`。
结合静态事实（DBO 钩子消费者仅 DeepEP 系内核），本 shape 的 decode 通信
重叠在本栈不存在候选级接缝；
candidate `dbo-ascend-tp2-feasibility` 收口为 rejected。
reopen：上游泛化 microbatching 至标准 TP MoE，或顶层授权自研
TP-allReduce comm-hook（届时才进入本文第 4 步试验）。
