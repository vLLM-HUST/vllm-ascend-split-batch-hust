# 算子优化评测标准与验收规范

> **单一事实源约束**：本文件是本仓库内所有算子优化（纯 Python / Triton-Ascend / 外部 `.so` kernel 包装）评测、指标记录与验收报告的权威标准。
> 任何性能或收益主张必须严格遵循本规范的四级阶梯、形状事实合同与数据结构，严禁混淆微基准与服务收益。

---

## 1. 形状事实与单变量上下文合同 (Shape Facts & Context Contract)

两组测试数据“可比”的前提是**环境与形状事实完全一致**。缺乏形状事实的数字不具备可重复性与合规性，严禁记录“加速 X%”而隐去以下关键要素：

### 1.1 必须固化的环境与上下文事实
每一项测试必须显式记录并固化以下参数：
- **硬件与拓扑**：加速卡型号（如 `Atlas 800T A2 / 910B2`）、卡数及并行模式（如 `2×910B2, TP=2`）。
- **底座软件栈**：操作系统版本、CANN Toolkit 版本（如 `9.1.0`）、Python 解释器路径与版本。
- **宿主版本锚点**：vLLM Core 提交/版本（如 `0.28.1.post1.dev143+gf18cf803c`）、vLLM-Ascend 提交/版本（如 `0.25.1rc2.dev125+hust` / `74f0c0a27`）。
- **模型与精度标签**：模型精确名称（如 `Qwen3.5-35B-A3B`）、权重类型与格式（如 `bf16 / ND 格式`）。
- **执行相位 (Phase)**：`Prefill`、`Decode` 或 `Speculative Verify`（投机验证）。
- **执行图模式**：`cudagraph_mode`（`NONE` / `PIECEWISE` / `FULL` / `FULL_AND_PIECEWISE`）。

### 1.2 必须锁死的形状与规模事实 (Tensor Dimensions)
- **输入输出规模**：Batch Size ($B$)、Prefill Input Tokens 长度 ($S_{in}$)、Decode 生成长度 ($S_{out}$)。
- **张量维度事实**：$M$（总 Token 数）、$K$（输入通道/维度）、$N$（输出通道/维度）。
- **特定算子维度**：
  - **GDN (Gated Delta Net)**：Hidden Dim、Conv1d 卷积核尺寸 ($K_{conv}$)、State Dim ($D_{state}$)、Head 数量与维度。
  - **MoE (Mixture of Experts)**：激活专家数 ($Top\text{-}K$)、专家总数 ($E$)、路由 token 分布。
- **单变量测量条件**：
  - 固定随机种子（统一设置 `seed = 42`）；
  - 贪心采样（`temperature = 0.0`）；
  - 固定预热轮次（Warmup $\ge 10$ 次）；
  - 测量轮次（Iters $\ge 100$ 次），记录中位数（Median）、P90 与 P99 耗时。

---

## 2. 算子优化四级验证阶梯

算子交付必须按顺序通过四级阶梯，前置阶梯不通过严禁向下宣称收益：

```
[L0 单算子微基准] ──> [L3 数值正确性护栏] ──> [L4 图内调用与全核耗时归因] ──> [服务端到端真实评测]
 (纯算子 Median/P90)   (一票否决/容差验证)      (Callsite 命中 / All-kernel 分母)   (TTFT/TPOT/Throughput)
```

### L0: 单算子微基准 (Microbenchmark)
- **目的**：在相同硬件与输入张量下，对比优化前后纯算子的纯核执行耗时。
- **度量**：输入/输出形状、基线耗时（Baseline $\mu s$）、优化耗时（Optimized $\mu s$）、纯算子加速比（Speedup）。
- **显存与内存约束**：评估优化实现是否带来额外的显存分配（如中间临时张量、显存峰值开销）。
- **边界声明**：L0 加速比仅反映孤立算子本身的计算吞吐，**严禁将 L0 加速比直接等同于模型或服务加速**。

### L3: 数值正确性护栏 (Numerical Consistency Guard)
- **核心原则**：**数值正确性高于性能，数值错误一票否决**。
- **判据标准**：
  - fp16 / bf16 算子需满足数值容差（如 `torch.allclose(atol=1e-3, rtol=1e-3)`）；
  - 针对带有离散输出或采样的算子（如 argmax / sampling），要求固定 Prompt 集合或随机种子下输出逐字符/逐 token 一致；
  - 若优化算子存在极端数值溢出、NaN 或精度崩塌，**必须立即关闭开关并保持 default-off**，在验收记录中标记为负结果或失败，严禁删除失败记录。

### L4: 图内调用与链路归因 (Tracing & Opstat)
- **目的**：证明在实际模型执行或 CUDA Graph / ACL Graph 捕获回放中，优化实现被真实调用，而非退化回原生。
- **判据标准**：
  - **调用计数 (Call Count)**：在端到端前向中，目标 Callsite 命中次数符合预期模型层数（如 30 层 GDN 则正好命中 30 次）；
  - **总算子耗时（All-kernel 分母）**：算子加速必须在模型的整体算子执行时间中占有真实可测的下降份额；
  - 若调用被外部 Guard 拦截回退到原生路径，不得主张该算子产生任何性能收益。

### 服务收益: 端到端吞吐与时延 (Serving End-to-End Benchmark)
- **目的**：评估算子在并发 HTTP 请求与完整 vLLM 服务生命周期下的收益。
- **度量指标**：
  - **TTFT (Time to First Token)**：首字时延（P50/P90/P99，单位 ms）；
  - **TPOT (Time per Output Token)**：每个解码 token 平均耗时（单位 ms）；
  - **Throughput**：系统整体生成吞吐（tokens/s）。
- **约束**：必须记录输入长度、输出长度、并发并发度（Concurrency）、请求总数，并与原生基线同条件 A/B 对比；差异处于噪声范围（如 $\pm 1\%$）内时必须声明为 `within noise`（在噪声范围内，收益不显著）。

---

## 3. 标准验收报告模板 (Markdown)

新算子完成评测后，必须按如下格式撰写验收报告：

```markdown
# 算子验收报告: <算子名称> (<Callsite>)

## 1. 评测环境与形状事实
- **硬件拓扑**: Atlas 800T A2 (2×910B2, TP=2)
- **环境版本**: CANN 9.1.0 / vllm 0.28.1.post1.dev143+gf18cf803c / vllm-ascend 0.25.1rc2.dev125+hust
- **模型与权重**: Qwen3.5-35B-A3B (bf16, ND)
- **形状参数**: Batch=<B>, SeqLen=<S>, M=<M>, K=<K>, N=<N>, Phase=<Prefill/Decode>
- **图模式与采样**: cudagraph_mode=NONE, Seed=42, Greedy

## 2. L0 纯算子对比 (Microbenchmark)
| 测量项 | 原生基线 (Native) | 优化实现 (Optimized) | 加速比 (Speedup) | 显存变化 |
|---|---|---|---|---|
| Median 耗时 (µs) | 120.5 µs | 85.2 µs | 1.41x | 无额外占用 |
| P90 耗时 (µs) | 122.1 µs | 86.0 µs | 1.42x | - |
| P99 耗时 (µs) | 125.0 µs | 88.5 µs | 1.41x | - |

## 3. L3 数值一致性核验
- **测试方法**: 与原生实现输出比对 `torch.allclose`
- **容差门限**: atol=1e-3, rtol=1e-3
- **核验结论**: PASS (Max Absolute Diff: 2.4e-4)

## 4. L4 调用链路与图归因
- **目标 Callsite**: `vllm_ascend.ops.gdn.AscendGatedDeltaNetAttention`
- **命中次数**: 30 / 30 层全部命中
- **All-kernel 占比**: 占整体 GPU/NPU 活跃耗时由 14.2% 降至 10.5%

## 5. 服务端到端验证 (Serving L3/L4)
- **并发形态**: Concurrency=16, Input=1024, Output=128
- **TTFT (P90)**: 142ms vs 140ms (差异在噪声内)
- **TPOT (P90)**: 28.5ms vs 25.1ms (加速 11.9%)
- **总吞吐**: 512.4 tokens/s vs 573.1 tokens/s (提升 11.8%)

## 6. 结论与交付建议
- [x] L0/L3/L4 全部通过
- [x] default-off 门控已就绪 (`VLLM_HUST_<NAME>=1`)
- [x] 建议纳入发布清单并保持开关默认关闭
```

---

## 4. 证据 JSON 统一模式 (`operator-benchmark-evidence/v1`)

机器可解析的证据记录必须符合以下 JSON 规范：

```json
{
  "$schema": "operator-benchmark-evidence/v1",
  "operator_name": "gdn_conv_pack",
  "callsite": "vllm_ascend/ops/gdn.py:forward",
  "environment": {
    "device": "910B2",
    "world_size": 2,
    "cann_version": "9.1.0",
    "vllm_version": "0.28.1.post1.dev143+gf18cf803c",
    "vllm_ascend_version": "0.25.1rc2.dev125+hust",
    "model": "Qwen3.5-35B-A3B",
    "dtype": "bf16",
    "cudagraph_mode": "NONE"
  },
  "shape_facts": {
    "phase": "decode",
    "batch_size": 16,
    "prompt_tokens": 1024,
    "decode_tokens": 1,
    "m": 16,
    "k": 4096,
    "n": 4096
  },
  "l0_microbenchmark": {
    "iters": 100,
    "warmup": 10,
    "native_median_us": 120.5,
    "optimized_median_us": 85.2,
    "speedup": 1.41
  },
  "l3_numerical_guard": {
    "passed": true,
    "metric": "allclose",
    "max_abs_diff": 0.00024
  },
  "l4_attribution": {
    "call_count_expected": 30,
    "call_count_actual": 30,
    "verified": true
  },
  "serving_end_to_end": {
    "tested": true,
    "concurrency": 16,
    "throughput_native_tokens_s": 512.4,
    "throughput_opt_tokens_s": 573.1,
    "speedup": 1.118
  },
  "evidence_fingerprint": {
    "sha256": "4bf3c6b7af6ddb4cdcf193e348302758386f8a2f...",
    "source_url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/..."
  }
}
```
