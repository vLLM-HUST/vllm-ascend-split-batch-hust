# GDN in_proj 输入投影权重加载期合并 — 验收报告

## 1. 形状事实与单变量上下文

| 项 | 值 |
|---|---|
| 算子 | GDN in_proj 权重加载期合并（消除每层每步一次 MatMulV2 与一次结果拼接） |
| callsite | `vllm_ascend/ops/gdn.py:AscendGatedDeltaNetAttention.process_weights_after_loading`（宿主） |
| 相位 | 权重加载后、dynamo 编译/图捕获前（一次性）；收益作用于 decode/prefill 全部前向 |
| 目标模型 | Qwen3.5-35B-A3B（30 层 GDN） |
| 权重形状事实 | `in_proj_qkvz [2048, 4096]` 与 `in_proj_ba [2048, 2048]` 沿输出维拼接为 `[2048, 6144]` 单一矩阵 |
| 硬件/拓扑 | 2×910B2 TP2 |
| 开关 | `VLLM_HUST_GDN_MERGE_INPROJ=1`（default-off） |
| 交付层 | host_fork（随宿主镜像发布，插件仓登记证据） |

## 2. 四级验证阶梯状态

按 [docs/operator-benchmark-standard.md](../../docs/operator-benchmark-standard.md) 四级阶梯逐级记录：

### L0 单算子微基准

- **测试方案**: 对比 2 个分立 MatMulV2（`qkvz [2048, 4096]` 与 `ba [2048, 2048]`）vs 1 个合并后单 MatMulV2（`[2048, 6144]`），输入 `[32, 2048]` (bf16)。
- **测量结果**: 见配套证据文件 `evidence.json`。
- **基线耗时**: 147.42 µs
- **优化耗时**: 99.52 µs
- **纯算子加速比**: 1.48x（每层单步省 47.90 µs，全模型 30 层 GDN 每步减少 30 次 MatMul 发射）。

### L3 数值正确性护栏

- **核验标准**: 拼接权重单次 MatMul 输出与两矩阵分立 MatMul 拼接结果进行逐位比较。
- **核验结果**: PASS（`max_abs_diff = 0.0`，完全逐位恒等）。

### L4 调用链路归因

- **归因验证**: 30 层 GDN 每步 MatMul 调用次数由 60 次降至 30 次，精确消除 30 次 kernel 发射。

### 服务端到端实测

- **状态**: 保持 default-off，待端到端全链路服务压测。

## 3. 结论与交付状态

| 项 | 状态 |
|---|---|
| 宿主实现 | `work/w3f2-inproj-merge` 提交 `fcbc4458bb8afcc37441e602921ed45dcbe5e775`（宿主 PR #63，Base `based/split-batch-baseline`） |
| 代码搬运等价性 | 与旧分支 `work/issue20-w3f2-host` 的 `ce807cd7c` 树逐字节一致（`git diff` 为空） |
| default-off | 是；未设置开关时与 stock 严格一致 |
| L0/L3/L4 | 均通过 2×910B2 真机核验，记录于 evidence.json |
