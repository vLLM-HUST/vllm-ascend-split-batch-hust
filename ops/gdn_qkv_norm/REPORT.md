# 算子验收报告: gdn_qkv_norm (vllm_ascend/ops/gdn.py:480,619-620)

## 1. 形状事实与单变量上下文
- 模型: Qwen3.5-35B-A3B (TP=2)
- 精度: torch.bfloat16
- Callsite: `vllm_ascend/ops/gdn.py:480,619-620`
- 形状: {'T': 16, 'row_stride': 4096, 'q': 1024, 'k': 1024, 'v': 2048, 'head_dim': 128, 'phase': 'pure decode'}

## 2. 验证阶梯结论
- L0 纯算子基准: Baseline 791.24 µs ->
  Optimized 247.09 µs (加速比: 3.20x)
- L3 数值正确性: PASS (allclose atol=1e-3, rtol=1e-3)
- 状态: 达标交付
