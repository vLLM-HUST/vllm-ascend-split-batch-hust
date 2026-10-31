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
- 双微批对 GDN 算子及张量连续性的影响：moe_init_triton 的 M 维度语义、views 的
  连续性检查门控、gate-fp32 注入（批切分是否保持各优化特性的触发条件）

## 裁决与立项

立调研候选 `dbo-ascend-tp2-feasibility`。调研后深挖修正了解锁序列（default-off 门控 `VLLM_HUST_DBO_ENABLE`）：

1. 平台重置放行（`_DBO_PLATFORM_TRANSFORMS`，platform.py reset 块加 env 门）
2. TP2/DP1 decode 触发（`_DBO_CORE_TRIGGER_TRANSFORMS`，core should_ubatch
   的 enable_dbo+uniform_decode 分支；core 无 os import 用 `__import__("os")`）
3. **NPU load_model 包装缺口**：core 的 UBatchWrapper 包装分支
   （gpu_model_runner.py:5560，use_ubatching 独占分支，FULL 模式由 wrapper
   自管内层图）对 NPU 不可达——NPUModelRunner.load_model 全程自持无 super 调用。
   `_DBO_NPU_WRAP_TRANSFORMS` 在 ACLGraphWrapper 分支植入独占替代
   （env 门控时 UBatchWrapper 取代 ACLGraphWrapper，镜像 core 语义）。
4. 实证：启动服务 + 图模式 + L3 数值与微基准验证。
   终止判定条件：torch_npu 双流/HCCL 在该路径不成立。

补充约束：
- DBO 钩子消费者（dbo_enabled/dbo_register_recv_hook/make_ubatch_contexts）
  仅存在于 DeepEP 系 EP-MoE 内核（deepep_ht/ll/nixl/modular_kernel）——
  TP allReduce 无钩子层，重叠编排完全依赖 UBatchWrapper 自身双流循环。
- 收益天花板：12.4ms/step 的重叠份额。
- 单变量控制：测试 DBO 时不开启 D4 卷积权重预打包等其他优化选项。

## 技术可行性结论：不可行（已结题，2026-10-22）

实证起服拦截于配置校验层断言：`Microbatching currently only supports
deepep_low_latency / deepep_high_throughput /
nixl_ep all2all backends`。
结合静态事实（DBO 钩子消费者仅 DeepEP 系内核），本规格的 decode 通信
重叠在当前栈不存在可行切入点；
候选技术项 `dbo-ascend-tp2-feasibility` 结论为：不可行（已关闭）。
后续重启条件：上游将 microbatching 泛化至标准 TP MoE，或宿主层支持
TP-allReduce comm-hook。
