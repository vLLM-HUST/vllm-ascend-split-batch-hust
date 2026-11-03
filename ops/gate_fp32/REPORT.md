# 算子验收报告: MoE Gate 权重 FP32 外提 (`gate-fp32`)

## 1. 评测环境与形状事实
- **硬件拓扑**: Atlas 800T A2 (2×910B2, TP=2)
- **软件栈版本**: CANN 9.1.0 / vllm 0.28.1.post1.dev143+gf18cf803c / vllm-ascend 0.25.1rc2.dev125+hust (Commit `74f0c0a27`)
- **模型与权重**: Qwen3.5-35B-A3B (bf16, ND 格式)
- **张量事实**: gate.weight 形状 `[256, 2048]` (bf16) -> 预转置缓存为 `[256, 2048]` (fp32)
- **执行相位**: Decode 阶段热路径
- **调度与图模式**: cudagraph_mode=NONE / FULL

## 2. L0 纯算子对比 (Microbenchmark)
- 基线耗时 (40x Cast): 1823.11 µs
- 优化后耗时 (外提直通): 33.19 µs
- 纯算子加速比: 54.93x (单步节省约 1.79 ms 访存与小算子开销)

## 3. L3 数值一致性核验
- **核验标准**: 缓存的 fp32 权重与原生的 `gate.weight.to(torch.float32)` 逐位严格相等 (`torch.equal`)。
- **核验结论**: PASS (完全等价，无数值漂移)。

## 4. L4 调用链路与图归因
- **目标 Callsite**: `vllm_ascend_split_batch/gate_fp32_plugin.py` 中拦截的 `gate` 模块 forward。
- **命中验证**: 40 层 MoE 的 gate 权重全部正确填充并被命中；开关关闭时 100% 回退原生分支。

## 5. 结论与交付标签
- [x] 插件层入口注册就绪 (`gate-fp32`)
- [x] L3 数值恒等核验通过
- [x] default-off 门控就绪 (`VLLM_HUST_GATE_FP32=1`)
- [x] 2×910B2 有卡实测与微基准闭环
