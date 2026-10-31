# 算子验收报告: FlashInfer Attention Demask (`fia-demask`)

## 1. 评测环境与形状事实
- **硬件拓扑**: Atlas 800T A2 (2×910B2, TP=2)
- **软件栈版本**: CANN 9.1.0 / vllm 0.28.1.post1.dev143+gf18cf803c / vllm-ascend 0.25.1rc2.dev125+hust
- **生效条件**: 纯 Decode 阶段，单步 $Q_S=1$/请求
- **图模式**: cudagraph_mode=NONE / PIECEWISE

## 2. L0 纯算子对比 (Microbenchmark)
- **测试状态**: A/B 对比差异处于测量噪声范围内（inconclusive），不主张局部加速。

## 3. L3 数值一致性核验
- **核验结论**: PASS (Decode 步输出与原生全 mask 计算严格对齐)。

## 4. L4 调用链路与图归因
- **Callsite**: Attention 输入预处理阶段。
- **调用状态**: 仅在纯 decode 单 token 场景下命中；prefill 与混合 chunk 自动走原生路径。

## 5. 服务端到端验证 (Serving L3/L4)
- **结论**: 未证明具有统计显著性的服务加速。

## 6. 结论与交付标签
- [x] general plugin 与 bundle 注册就绪
- [x] L3 数值对齐通过
- [x] 差异在噪声内，保持 default-off
- [ ] 待具卡环境下重测长序列并发表现
