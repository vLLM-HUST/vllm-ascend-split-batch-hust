# GDN QKV Unpack and L2Norm Fusion (`gdn_qkv_norm`)

## 1. 背景与优化原理
在 Qwen3.5-35B-A3B (TP=2) 架构中，GDN（Gated DeltaNet）层卷积后的输入包含混合投影张量 `mixed_qkv`（形状 `[T, 4096]`）。原生实现需要调用 `rearrange_mixed_qkv`（拆分切片、重塑与拼接）并随后对 Q 和 K 分别执行两次 `l2norm_fwd`。

通过 `tools.op_agent` 自动调优内环生成的 Triton-Ascend 融合算子（`_gdn_qkv_norm_kernel_2d`）：
- 按 `(head_idx, token_idx)` 二维网格并发调度，单核同时处理同一注意力头对应的 Q 头、K 头与两个 V 头；
- 在片上一次完成 Q 与 K 的 FP32 平方和规约与 `rsqrt` 归一化，同时直接写出连续 V 张量；
- 对行数 `T` 及各输出步长标记 `do_not_specialize`，消除分块预填充与长尾动态批次下的 Triton 重编译开销。

## 2. 硬件环境与单算子收益
- **评测方式**: Ascend 910B2 单算子 NPUGraph 图重放，覆盖多批次分桶与未见动态行数探针（详见 `evidence.json`）。
- **L3 数值验证**: 全部分桶与动态行数探针均严格通过容差校验。

## 3. 开关与控制
- 环境变量门控：`VLLM_HUST_GDN_QKV_NORM=1`（默认关闭）
- 未启用时回退原生路径，零开销零副作用。
