# 算子优化评测标准与验收规范

> **单一事实源约束**：本文件是本仓库内所有算子优化（纯 Python / Triton-Ascend / 外部 `.so` kernel 包装）评测、指标记录与验收报告的权威标准。
> 任何性能或收益主张必须严格遵循本规范的四级阶梯、形状事实合同与数据结构，严禁混淆微基准与服务收益。

---

MOD 的最终交付另须通过 [MOD 最终验收规范](mod-acceptance-standard.md)。本文件的受控负载、种子和采样设置只用于算子验证，不得覆盖冻结数据集协议；单算子服务结论不自动升级为 MOD 结论。

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

以下是证据组织层级，不是每个算子必须跑完的串行测试清单。数值正确性必须先检查，通过后才接受任何计时结论。所有纳入组合的候选必须保留数值校验和逐卡调用归因；单算子端到端 A/B 按需选测，选择依据为关键路径占比、风险、交互冲突或需要独立收益归因。没有单项 A/B 不等于组合交付缺项，也不能写成单项收益。

组合须重新通过数值护栏，并逐项证明启用和回退符合预期；可在同一组 profile 中复用采集，但每个候选保留独立判据。组合 A/B 是服务交付判据，MOD 冻结协议是最终验收判据。组合异常时先分组消融，再按需做单项 A/B。

仅记录优化方向、没有可达实现、已拒绝或被替代的目录不进入组合必测清单；宿主已实现的优化按真实宿主调用点纳入，不为其重复的方向记录另设测试。历史结果、负结果和中断产物一律保留。

前置证据不足时不得升级收益主张：

```
[L0 单算子微基准] ──> [L3 数值正确性护栏] ──> [L4 图内调用与全核耗时归因] ──> [服务端到端评测]
 (纯算子 Median/P90)   (一票否决/容差验证)      (Callsite 命中 / 逐卡归因)   (单项选测 / 组合必测)
```

### L0: 单算子微基准 (Microbenchmark)
- **目的**：在相同硬件与输入张量下，对比优化前后纯算子的纯核执行耗时。
- **度量**：输入/输出形状、基线耗时（Baseline $\mu s$）、优化耗时（Optimized $\mu s$）、纯算子加速比（Speedup）。
- **显存与内存约束**：评估优化实现是否带来额外的显存分配（如中间临时张量、显存峰值开销）。
- **边界声明**：L0 加速比仅反映孤立算子本身的计算表现，**严禁将 L0 加速比直接等同于模型或服务加速**。

### L3: 数值正确性护栏 (Numerical Consistency Guard)
- **核心原则**：**数值正确性高于性能，数值错误一票否决**。
- **判据标准**：
  - 按实际任务、dtype、参考实现与中间舍入约定设置容差、误差指标和 NaN/Inf 规则，不套用通用容差；
  - 针对带有离散输出或采样的算子（如 argmax / sampling），要求固定 Prompt 集合或随机种子下输出逐字符/逐 token 一致；
  - 若优化算子存在极端数值溢出、NaN 或精度崩塌，**必须立即关闭开关并保持 default-off**，在验收记录中标记为负结果或失败，严禁删除失败记录。

### L4: 图内调用与链路归因 (Tracing & Opstat)
- **目的**：证明在实际模型执行或 CUDA Graph / ACL Graph 捕获回放中，优化实现被真实调用，而非退化回原生。
- **判据标准**：
  - **调用计数 (Call Count)**：在端到端前向中，目标 Callsite 命中次数符合预期模型层数（如 30 层 GDN 则正好命中 30 次）；
  - **归因范围**：记录逐卡调用、形状和目标耗时；全核耗时总和可用于诊断，但不能代替服务关键路径，也不要求它必须下降才能证明调用命中；
  - 若调用被外部 Guard 拦截回退到原生路径，不得主张该算子产生任何性能收益。

### 服务收益: 端到端服务表现 (Serving End-to-End Benchmark)
- **目的**：评估算子在并发 HTTP 请求与完整 vLLM 服务生命周期下的收益（`tools/operator_tools/serving_collect.py` 负责真实采集，`serving_loop.py` 只规划与汇总；单负载可用 `workloads/run_workload.sh`，与单算子调优解耦）。
- **度量指标**：
  - **首字耗时 (FTL)**：首字生成耗时（P50/P90/P99，单位毫秒）；
  - **单字耗时 (POTL)**：每个输出字生成耗时（单位毫秒）；
  - **生成速度 (GenRate)**：系统整体生成速率。
- **负载（Workload）选择与特征绑定规则**（详见 [workloads/README.md](../workloads/README.md)）：
  - **主测对比负载（`primary_workload`）**：解码算子默认选用受控定长 `random`（消除请求长度方差噪声，锁死稳态解码桶）；预填充与宽张量切分算子选用 `long-prefix`；带行数上限门控的小批量算子必须将并发度控制在激活行数区间内；
  - **动态形状压力负载（`stress_workload`）**：统一配对真实长尾对话分布 `sharegpt`，检查动态行数下的数值、尾部表现和门控命中率；重编译需缓存或编译日志证明；
  - **相位交叉检查（`cross_check_workload`）**：配对另一受控相位；压力与相位交叉负载的显著退化阻止推荐当前配置。
- **约束**：记录负载、输入输出特征、并发、请求总数及完成情况；两组同条件独立起服并保留重复分布。只有服务配对测量支持差异落在波动范围时，才能记 `within_noise`；仅有全核占比或微基准时保持待测。

---

## 3. 标准验收报告模板 (Markdown)

新算子完成评测后，必须按如下格式撰写验收报告：

```markdown
# 算子验收报告: <算子名称>

## 1. 环境与形状合同
记录实际设备与拓扑、宿主和实现提交、依赖版本、模型权重、dtype、
shape/stride、执行相位、图模式、随机种子、调用点和回退条件。
未核实字段标记待补，不从模板继承环境。

## 2. L0 纯算子对比
状态: 未执行
计时与同步范围: 待补
预热、重复分布与缓存条件: 待补
结果: 结构化 evidence.json 对应字段
显存变化: 未测量

## 3. L3 数值检查
状态: 未执行
参考实现、误差指标、容差、NaN/Inf 和别名规则: 待补
覆盖用例与失败记录: 待补

## 4. L4 调用归因
状态: 未执行
实际调用点、逐卡期望与实际计数、形状及回退核查: 待补

## 5. 服务验证
状态: 未执行
主测、压力、相位交叉负载和配置: 待补
请求完整性、两组重复结果、组内波动与逐卡归因: 待补

## 6. 交付审核
- [ ] 证据支持对应层级的结论
- [ ] 已核查公开接入与实际门控，保持默认关闭
- [ ] 已保留失败、未执行项和历史适用性
- [ ] 已更新证据、索引与目录指纹并通过离线检查

```

---

## 4. 证据 JSON 统一模式 (`operator-benchmark-evidence/v1`)

机器可解析的证据记录必须符合以下 JSON 规范：

```json
{
  "$schema": "operator-benchmark-evidence/v1",
  "operator_name": "<operator>",
  "callsite": null,
  "environment": {
    "device": null,
    "world_size": null,
    "cann_version": null,
    "vllm_version": null,
    "vllm_ascend_version": null,
    "model": null,
    "dtype": null,
    "cudagraph_mode": null
  },
  "shape_facts": {
    "phase": null,
    "batch_size": null,
    "prompt_tokens": null,
    "decode_tokens": null,
    "m": null,
    "k": null,
    "n": null
  },
  "l0_microbenchmark": {
    "iters": null,
    "warmup": null,
    "native_median_us": null,
    "optimized_median_us": null,
    "speedup": null
  },
  "l3_numerical_guard": {
    "passed": null,
    "metric": null,
    "max_abs_diff": null
  },
  "l4_attribution": {
    "call_count_expected": null,
    "call_count_actual": null,
    "verified": false
  },
  "serving_end_to_end": {
    "tested": false,
    "concurrency": null,
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
   - 写法要求：度量数字一律落在结构化 `evidence.json`；行文与标题避免"关键词+数字"同段（如标题写"硬件环境与单算子收益"，不写"硬件实测收益 (<硬件型号>)"）；
   - 仓内把关：`pytest -q tests/test_ops_delivery_policy.py` 对 `ops/**` 与 `DELIVERY_INDEX.json` 全量扫描；`docs/evidence/**` 是有审查基线的存量债，新增文件严禁落入该类目录。
3. **证据指纹同步**：修改 `ops/<operator>/` 下被 `checksum.sha256` 覆盖的文件后，必须重新生成并校验指纹：

   ```bash
   python3 -c "from tools.operator_tools.custody import verify_dir_checksum, write_dir_checksum; path='ops/<operator>'; write_dir_checksum(path); ok, message=verify_dir_checksum(path); print(message); assert ok"
   ```

4. **隔离环境与 CI 命令一致**：CPU 门禁须在不继承系统包的虚拟环境中运行，只安装 CI 声明的依赖，并使用 `env -u PYTHONPATH <venv>/bin/pytest -q -m "not host_tree"` 验证。不得仅用 `python -m pytest` 的通过结果代替 CI 的 `pytest` 结果。测试加载仓内未安装的工具时，使用 `importlib.util.spec_from_file_location` 按文件路径加载，不依赖工作目录或外部 `PYTHONPATH`，也不为修复导入错误跳过守护测试。

## 6. 产物交接与离线检查

生产者（算子开发或验证 skill）交付实现、输入任务和形状合同的版本及 SHA256，验证器版本、完整命令、退出码、逐项结果和结果文件指纹。代码与临时验证项目置于仓内隔离目录；原始采集记录置于仓外，仓内只保留结构化摘要与不可变归档定位。

消费者先核对文件、指纹、执行环境与验证范围，再将结果写入 `ops/<operator>/evidence.json`。局部验证通过只证明对应输入与容差，不自动完成模型数值护栏、调用归因、公开接入或 MOD 验收。失败和未执行必须保留，不能只交一份成功摘要。更新证据后再写索引结论，最后生成目录指纹。

当前单项服务结论要求数值护栏明确通过、逐卡调用或消除冗余操作的判据通过、单项服务确已测量；调用计数不一致时必须明确核验并记录原因。单项服务标签只表示该项已有测量，不承担组合候选资格判断；没有单项测量时保留无结论标签，不伪造通过或噪声内结果。组合计划与结果放在 MOD 证据的独立组合记录中，不能回填成各算子的服务收益；组合诊断通过仍不能升级 MOD qualification。历史服务结果使用 `serving_end_to_end.applicability=historical` 和 `historical_only`，不得复用为当前结论。未公开接入的算子使用 `gate_env=unexposed`，不标记已集成。

没有独立测量入口的 `bench_l0.py` 必须在导入设备依赖前退出，退出码为三，表示无测量；保留可导入的实现函数不等于提供测量入口。

L0 计时与 `bench_l0.py` 执行状态由 `tools/operator_tools/l0_trace_record.py` 机械入账：计时 receipt 或执行输出原样追加进 `ops/<operator>/evidence.json` 的 `l0_traces` 并自动刷新目录指纹，结论标签不由该工具改动；度量数字禁止手抄进 evidence。

统一离线清单为 `tests/offline-checks.args`。本地、pre-commit 和 Text invariants 执行同一命令 `pytest @tests/offline-checks.args`；CPU 全套另执行 `pytest -q -m "not host_tree"`。

纯逻辑、环境变量、默认关闭和元数据测试不得因缺少 torch 或宿主而整文件跳过。CPU 张量测试在用例或 fixture 内申请 torch；缺包时明确跳过，不要求 torch_npu 或宿主包。额外安装 CPU torch 后使用同一命令补测，结果与只安装 CI 依赖的门禁分别记录。

`host_tree` 包括宿主源码漂移检查和真实宿主入口集成；显式选择后，缺文件、符号或导入失败必须报错，不得转成跳过。设备测试只在缺少必要依赖或设备时跳过；已安装依赖损坏应失败。CPU 张量通过不等于设备数值通过，跳过数量也不能视作覆盖数量。
