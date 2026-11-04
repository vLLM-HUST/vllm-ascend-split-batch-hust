# 算子验收报告: gdn_split_ba (vllm_ascend/ops/gdn.py:210)

## 1. 形状事实与单变量上下文
- 模型: Qwen3.5-35B-A3B (TP=2)
- 精度: torch.bfloat16
- Callsite: `vllm_ascend/ops/gdn.py:210`
- 形状与分桶事实: 详见 `evidence.json`

## 2. 验证阶梯结论
- L0 纯算子基准: 已通过 NPUGraph 多桶重放验证，详细指标见 `evidence.json`。
- L3 数值正确性: PASS（含未见动态行数零重编译护栏检查）。
- 状态: 达标交付
