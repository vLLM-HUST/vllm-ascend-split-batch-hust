# 算子验收报告: gdn_qkv_norm (vllm_ascend/ops/gdn.py:552-553,692-693)

## 1. 形状事实与单变量上下文
- 模型: Qwen3.5-35B-A3B (TP=2)
- 精度: torch.bfloat16
- Callsite: `vllm_ascend/ops/gdn.py:552-553,692-693`
- 形状: 覆盖稳态解码与动态批次分桶，同时包含未见行数零重编译探针（详见 `evidence.json`）。

## 2. 验证阶梯结论
- L0 纯算子基准: 在 Ascend 910B2 上采用 NPUGraph 图重放完成原生分立链路、Triton 基线与二维融合内核对比，各分桶度量数据见 `evidence.json`。
- L3 数值正确性: PASS (allclose atol=1e-3, rtol=1e-3，含动态行数探针)
- 状态: 算子内环达标交付（外环端到端评测待按 `workload_selection` 执行）
