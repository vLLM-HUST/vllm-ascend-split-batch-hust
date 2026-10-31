# 算子验收报告: GELU and Mul Triton Fused Kernel (`fi-gelu`)

## 1. 评测环境与形状事实
- **硬件拓扑**: Atlas 800T A2 (Ascend910B2 单卡)
- **软件栈版本**: CANN 9.1.0 / Triton-Ascend 3.2.2 / vllm 0.28.1.post1.dev143+gf18cf803c / vllm-ascend 0.25.1rc2.dev125+hust (Commit `74f0c0a27`)
- **评测形状**: `inter=13824`（输入张量 `[B, 27648]`，输出张量 `[B, 13824]`），批大小 `B ∈ {64, 256}`
- **数据类型**: bf16 (`approximate="none"`)

## 2. L0 纯算子对比 (Microbenchmark 数据)
在 910B2 硬件上通过 `torch.npu.Event`（warmup=10, iters=100, 3 轮最小耗时）对比：

| 批大小 (B) | 算子方案 | 运行耗时 (us) | 对比原生加速比 | 对比拷贝下限 |
|---|---|---|---|---|
| **B = 64** | `native_chain` (原生 fp32_cast+gelu+mul) | **135.9 us** | 1.00x (基准) | 4.40x |
| B = 64 | `fi_gelu` (融合 Triton kernel) | **190.6 us** | **0.71x (慢 1.40x)** | 6.17x |
| B = 64 | `fi_gelu(out=)` (预分配输出张量) | 144.9 us | 0.94x (慢 1.07x) | 4.69x |
| B = 64 | `copy_floor` (`x.clone()` 内存下限) | 30.9 us | 4.40x | 1.00x |
| **B = 256** | `native_chain` (原生 fp32_cast+gelu+mul) | **133.2 us** | 1.00x (基准) | 4.48x |
| B = 256 | `fi_gelu` (融合 Triton kernel) | **178.4 us** | **0.75x (慢 1.34x)** | 5.99x |
| B = 256 | `fi_gelu(out=)` (预分配输出张量) | 135.4 us | 0.98x (慢 1.02x) | 4.55x |
| B = 256 | `copy_floor` (`x.clone()` 内存下限) | 29.8 us | 4.47x | 1.00x |

- **瓶颈归因**：空 Triton Kernel 启动开销（Host Launch Overhead）单次高达 **110.4 us**，远超计算本身耗时，导致融合 Kernel 显著慢于 PyTorch 原生 C++ 分立调用链。
- **评测定性**：**明确为负性能（Negative Speedup）**。如实记录测试事实，绝不删除负收益记录。

## 3. L3 数值一致性核验
- **测试用例**: `tests/test_npu_w31_gelu.py`（29 项用例在 910B2 上 **全部 PASSED**）
- **核验标准**: 遵循 `ops-precision-standard` BFLOAT16 规范（rtol=atol=2.0^-6，元素匹配率 >= 99%）
- **核验结论**: **PASS** (融合核函数与 FP32 金标实现误差在容差内完全一致)。

## 4. L4 调用链路与图归因
- **测试用例**: `tests/test_npu_w31_gelu_plugin.py`（13 项用例在 910B2 上 **全部 PASSED**）
- **集成机制**: 通过 `CustomOp.register_oot(name="GeluAndMul")` 注入，支持图捕获与回放；非合法输入 100% 优雅回退宿主。
- **模型匹配**: 当前主基准模型（Qwen3.5-35B-A3B）采用 SwiGLU，真实推理中不走 GELU 算子，调用计数为 0。

## 5. 服务端到端验证 (Serving L3/L4)
- **结论**: 算子级为负性能且主基准不调用，无端到端服务收益预期。

## 6. 结论与交付标签
- [x] Triton 实现与 default-off 插件已落地并通过 NPU 测试
- [x] L3 数值精度核验通过 (29/29 passed)
- [x] 负性能事实如实记录，主开关严格保持关闭 (`VLLM_HUST_FI_GELU=0`)
- [x] 不主张任何端到端服务加速
