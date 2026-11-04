# 算子验收报告: MoE Router 与 Shared-Expert 门控合并 (`moe_gate_merge`)

## 1. 形状事实与单变量上下文
- **模型**: Qwen3.5-35B-A3B (TP=2)
- **精度**: torch.bfloat16
- **相位**: Decode
- **调用点**: `vllm/model_executor/models/qwen3_next.py:186-192` (Router Gate + Shared Gate)
- **测试形状**:
  - 输入张量: `[32, 2048]`
  - Router 权重: `[256, 2048]` (bf16)
  - Shared Gate 权重: `[1, 2048]` (bf16)
  - 合并权重: `[257, 2048]` (bf16)

## 2. 验证阶梯结论
- **L3 数值正确性**: 通过 (PASS)。合并后的单次 MatMul 与分离计算在 bitwise/fp32 精度下数值一致。
- **L0 纯微基准**:
  - 原生两次独立 GEMM（NPUGraph 回放）基准数据记录于 `evidence.json`；
  - 合并单次 GEMM 后切片连续化（2x contiguous）导致全流水线总耗时增加（加速比小于 1.0）；
  - 详细基准数值见结构化文件 `evidence.json`。
- **瓶颈归因**:
  - 输出维度为 257，属于非 16 字节对齐的奇数形状；
  - 昇腾 910B2 硬件对于跨行非对齐步长（stride=257）搬运极不友好，内存切片重排所增加的耗时直接抵消了算子发车减少带来的微弱收益。
- **最终交付定性**: **负性能（Negative Speedup），一票否决，不予集成**。
