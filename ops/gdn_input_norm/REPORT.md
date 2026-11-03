# 算子验收报告: gdn_input_norm (vllm_ascend/ops/gdn.py:in_proj_norm)

## 1. 形状事实与单变量上下文
- 模型: Qwen3.5-35B-A3B (TP=2)
- 精度: torch.bfloat16
- Callsite: `vllm_ascend/ops/gdn.py:in_proj_norm`
- 形状: {'T': 32, 'Hidden': 2048, 'phase': 'decode'}

## 2. 验证阶梯结论
- L0 纯算子基准: Baseline 45.34 µs ->
  Optimized 6.48 µs (加速比: 7.00x)
- L3 数值正确性: PASS (allclose atol=1e-3, rtol=1e-3)
- 状态: 达标交付
