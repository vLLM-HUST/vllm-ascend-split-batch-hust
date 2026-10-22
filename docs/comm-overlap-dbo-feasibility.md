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

立 probing 候选 `dbo-ascend-tp2-feasibility`：最小解锁实验序列
① platform 重置点 env 门控放行（default-off）→ ② TP2/DP1 decode 的
should_ubatch 触发补丁 → ③ boot + FULL graph + L3 8/8 + L1 实证（1 起服/步）。
预算形态：2-3 个标准窗；收益天花板 = 12.4ms/step 的重叠份额；
死刑条件 = torch_npu 双流在 FULL graph capture 不成立（③ 步即判）。

不是"开 flag 就行"——是三层闸门的最小解锁工程，属结构面/upstream 范畴。
