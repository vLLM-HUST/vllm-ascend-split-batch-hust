# 算子验收报告: gdn_split_ba (vllm_ascend/ops/gdn.py:210)

## 1. 形状事实与单变量上下文
- 模型: Qwen3.5-35B-A3B (TP=2)
- 精度: torch.bfloat16
- Callsite: `vllm_ascend/ops/gdn.py:210`
- 形状: {'T': 32, 'TotalDim': 32, 'phase': 'decode'}

## 2. 验证阶梯结论
- L0 纯算子基准: Baseline 6.25 µs ->
  Optimized 3.48 µs (加速比: 1.80x)
- L3 数值正确性: PASS (allclose atol=1e-3, rtol=1e-3)
- 状态: 达标交付
