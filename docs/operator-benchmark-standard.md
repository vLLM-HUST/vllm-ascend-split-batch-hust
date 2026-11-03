# 算子优化评测标准与验收规范

> **单一事实源约束**：本文件是本仓库内所有算子优化（纯 Python / Triton-Ascend / 外部 `.so` kernel 包装）评测、指标记录与验收报告的权威标准。
> 任何性能或收益主张必须严格遵循本规范的四级阶梯、形状事实合同与数据结构，严禁混淆微基准与服务收益。

---

## 1. 形状事实与单变量上下文合同 (Shape Facts & Context Contract)

两组测试数据“可比”的前提是**环境与形状事实完全一致**。缺乏形状事实的数字不具备可重复性与合规性，严禁记录加速比而隐去以下关键要素：

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
[L0 单算子微基准] ──> [L3 数值正确性护栏] ──> [L4 图内调用与全核耗时归因] ──> [服务端到端评测]
 (纯算子 Median/P90)   (一票否决/容差验证)      (Callsite 命中 / All-kernel 分母)   (首字时延/每Token耗时/生成速率)
```

### L0: 单算子微基准 (Microbenchmark)
- **目的**：在相同硬件与输入张量下，对比优化前后纯算子的纯核执行耗时。
- **度量**：输入/输出形状、基线耗时（Baseline $\mu s$）、优化耗时（Optimized $\mu s$）、纯算子加速比（Speedup）。
- **显存与内存约束**：评估优化实现是否带来额外的显存分配（如中间临时张量、显存峰值开销）。
- **边界声明**：L0 加速比仅反映孤立算子本身的计算表现，**严禁将 L0 加速比直接等同于模型或服务加速**。

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

### 服务收益: 端到端服务表现 (Serving End-to-End Benchmark)
- **目的**：评估算子在并发 HTTP 请求与完整 vLLM 服务生命周期下的收益。
- **度量指标**：
  - **首字耗时 (FTL)**：首字生成耗时（P50/P90/P99，单位毫秒）；
  - **单字耗时 (POTL)**：每个输出字生成耗时（单位毫秒）；
  - **生成速度 (GenRate)**：系统整体生成速率。
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
| Median 耗时 (µs) | <BASELINE_US> | <OPT_US> | <SPEEDUP> | 无额外占用 |
| P90 耗时 (µs) | <BASELINE_P90> | <OPT_P90> | <SPEEDUP> | - |
| P99 耗时 (µs) | <BASELINE_P99> | <OPT_P99> | <SPEEDUP> | - |

## 3. L3 数值一致性核验
- **测试方法**: 与原生实现输出比对 `torch.allclose`
- **容差门限**: atol=1e-3, rtol=1e-3
- **核验结论**: PASS

## 4. L4 调用链路与图归因
- **目标 Callsite**: `vllm_ascend.ops.gdn.AscendGatedDeltaNetAttention`
- **命中次数**: 30 / 30 层全部命中
- **All-kernel 占比**: 占整体 GPU/NPU 活跃耗时由基线下降至优化状态

## 5. 服务端到端验证 (Serving L3/L4)
- **并发形态**: Concurrency=<C>, Input=<IN>, Output=<OUT>
- **FTL (P90)**: <NATIVE_MS> vs <OPT_MS> (差异在噪声内)
- **POTL (P90)**: <NATIVE_MS> vs <OPT_MS>
- **GenRate**: <NATIVE_RATE> vs <OPT_RATE>

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
    "native_median_us": null,
    "optimized_median_us": null,
    "speedup": null
  },
  "l3_numerical_guard": {
    "passed": true,
    "metric": "allclose",
    "max_abs_diff": null
  },
  "l4_attribution": {
    "call_count_expected": 30,
    "call_count_actual": 30,
    "verified": true
  },
  "serving_end_to_end": {
    "tested": false,
    "concurrency": 16,
    "speedup": null
  },
  "evidence_fingerprint": {
    "sha256": null,
    "source_url": null
  }
}
```

---

## 5. 交付合规门禁（推送前逐条自检）

算子交付内容（`ops/<operator>/`、`DELIVERY_INDEX.json`，以及 PR 里任何新增或修改的文本文件）在推送前必须满足：

1. **代码风格零残余**：`ruff check .` 与 `ruff format --check .` 全绿。`ops/` 不在 ruff 排除清单内，`ops/<operator>/bench_l0.py` 等脚本与 `src/` 同标准。
2. **组织基准数据政策合规**：组织工作流 `Benchmark-only data (organization)` 对 PR 的每个变更文件做 fail-closed 审查，命中即拒绝（判据镜像 vLLM-HUST/.github@0cfe5109551e9b1c30ff3a75a2d93ee6ba8c9aa4 的 `scripts/benchmark_data_policy.py`）：
   - 路径判据：路径中任意目录段为 `data`/`datasets`/`results`/`evidence`/`receipts`/`runs`/`traces`，或文件后缀为 `.csv`/`.tsv`/`.jsonl`/`.ndjson`/`.parquet`/`.arrow`/`.log`/`.log.gz`/`.prom`；
   - 行文判据：文本文件新增行命中下行任一关键词后，同一行 60 字符窗口内出现数字即拒绝：
     `ttft`、`tpot`、`throughput`、`latency`、`tokens/s`、`实测`、`吞吐`、`延迟`；
   - 写法要求：度量数字一律落在结构化 `evidence.json` 与验收报告表格字段；行文与标题避免"关键词+数字"同段（如标题写"硬件环境与单算子收益"，不写"硬件实测收益 (<硬件型号>)"）；
   - 仓内把关：`pytest -q tests/test_ops_delivery_policy.py` 对 `ops/**` 与 `DELIVERY_INDEX.json` 全量扫描；`docs/evidence/**` 是有审查基线的存量债，新增文件严禁落入该类目录。
3. **证据指纹同步**：修改 `ops/<operator>/` 下被 `checksum.sha256` 覆盖的文件后，必须重新生成并校验指纹：

   ```bash
   python3 -c "from tools.op_agent.custody import verify_dir_checksum, write_dir_checksum; path='ops/<operator>'; write_dir_checksum(path); ok, message=verify_dir_checksum(path); print(message); assert ok"
   ```

4. **隔离环境与 CI 命令一致**：CPU 门禁须在不继承系统包的虚拟环境中运行，只安装 CI 声明的依赖，并使用 `env -u PYTHONPATH <venv>/bin/pytest -q -m "not host_tree"` 验证。不得仅用 `python -m pytest` 的通过结果代替 CI 的 `pytest` 结果。测试加载仓内未安装的工具时，使用 `importlib.util.spec_from_file_location` 按文件路径加载，不依赖工作目录或外部 `PYTHONPATH`，也不为修复导入错误跳过守护测试。
