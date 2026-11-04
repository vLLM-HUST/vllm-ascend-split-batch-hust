# 算子调优 Harness 规范：内外环分离架构与负载契约

本文档是 `tools/op_agent/` 算子调优与整网评测 Harness 的权威技术规范。算子本身的计算加速与整网端到端的系统加速关注点不同，因此本规范采用 **“内环（单算子加速）+ 外环（整网归因与负载验收）”** 双环分离架构。

---

## 1. 为什么采用内外环分离设计

| 维度 | 内环（Inner Loop：`optimizer_loop.py`） | 外环（Outer Loop：`serving_loop.py`） |
| :--- | :--- | :--- |
| **核心目标** | 单算子片上计算加速、真实内存跨距兼容、消除动态尺寸重编译 | 整网图捕获兼容、真实调用命中率（L4）、Amdahl 占比与服务级对比 |
| **迭代粒度** | 秒级单卡闭环，供 `CodeAgent` 高频迭代候选代码 | 分钟级双卡服务闭环，在候选算子通过内环门禁后独立执行 |
| **验证阶梯** | L3 算子数值一致性 -> 动态形状零重编译门禁 -> L0 分桶微基准 | 整网 L3 贪心输出对拍 -> L4 `op_statistic` 归因与信噪比评估 -> 负载对比 |
| **失败归因** | 数值超差、非连续视图崩溃、漏标 `do_not_specialize` 导致重编译 | 上游视图不匹配导致静默回退、整网算子耗时占比低于服务噪声底线 |

---

## 2. 内环契约（算子自动调优循环）

1. **真实布局与可插拔 `OpCase`（`tools/op_agent/cases/`）**：
   - 所有算子目标继承 `tools/op_agent/cases/base.py:OpCase` 并在 `CASE_REGISTRY` 注册；
   - **真实跨距契约**：`get_bucket_input_gen_fn()` 必须复现宿主上游算子的真实内存视图（例如从宽矩阵切片得到的非连续列视图），禁止仅用理想连续随机张量代替；
   - **多桶与动态探针**：通过 `shape_buckets` 覆盖解码图捕获桶，通过 `dynamic_shape_probes`（如非 2 的幂次奇数行数）覆盖分块预填充与长尾动态尺寸。
2. **短路评估与零重编译护栏（`tools/op_agent/evaluator.py`）**：
   - **第一步（L3 数值校验）**：对默认形状与全部 `shape_buckets` 校验容差，不通过立即短路返回；
   - **第二步（动态尺寸零重编译门禁）**：在预热后输入未见过的 `dynamic_shape_probes` 行数，校验数值并检查单次执行耗时是否超过 `recompile_budget_ms`，拦截将动态行数固化为编译期常量的实现；
   - **第三步（L0 分桶计时）**：基于 NPUGraph 重放测量默认形状与各解码桶耗时。
3. **产物导出与外环字段保护（`tools/op_agent/exporter.py`）**：
   - 自动导出 `ops/<operator>/{bench_l0.py,evidence.json,README.md,REPORT.md}` 并调用 `custody.py` 写入 `checksum.sha256`；
   - 若目录已有外环采集的 `l4_attribution` 或 `serving_end_to_end` 记录，重跑内环时保留已有外环证据不被清空。

---

## 3. 外环契约与负载（Workload）选择规范

外环由 `tools/op_agent/workloads.py`、`tools/op_agent/serving_loop.py` 与 `workloads/run_workload.sh` 组成，对接容器内四类标准负载：

1. **负载特征与配对原则**（详见 [workloads/README.md](../workloads/README.md)）：
   - **`random`（定长受控负载）**：无长度方差，稳态解码步锁定在并发桶，组内波动最小，作为**解码算子的主测对比负载（`primary_workload`）**；
   - **`long-prefix`（长上下文预填充负载）**：大尺寸分块预填充占主导，作为**预填充与宽张量切分重排算子的主测对比负载**；
   - **`sharegpt`（真实长尾对话分布）**：请求异步进出导致步间行数剧烈跳变，作为所有算子的**动态形状与门控鲁棒性压力负载（`stress_workload`）**。
2. **L4 归因与信噪比（SNR）前置判定**：
   - 外环首先完成固定 8 条提示词贪心对拍（`compare_greedy_outputs`）与 `op_statistic.csv` 调用计数解析（`parse_op_statistic_csv`），确认目标算子 100% 命中预期层数且零静默回退；
   - 通过 `evaluate_e2e_snr_gate()` 计算算子净省时占全网总算子耗时的比例：若占比远低于服务波动底线，如实预期并归档为 `within_noise`，严禁强行外推局部微基准加速比。

---

## 4. 外挂式多算子发现与批次调优插件（`campaign.py` & `timeline_insight.py`）

针对多算子连续优化场景，采用纯外挂插件接入单算子内环，解决裸读原始 CSV 成本高、缺乏时序邻接层级与跨算子环境污染问题：

1. **时序子图折叠与泳道图渲染（`tools/op_agent/timeline_insight.py`）**：
   - 以大型矩阵乘、状态更新及通信算子（`AI_CORE`、`MIX_AIC`、`HCCL`）为边界锚点，自动从 `kernel_details.csv` 中切分出中间连续的 `AI_VECTOR_CORE` 夹心序列；
   - 将跨层重复模式自动聚类折叠为 Top-K 高层级卡片（包含前后锚点、各算子输入输出形状、空泡等待与片外内存搬运占比 `aiv_mte2_ratio`），并渲染直观的 SVG 泳道图供人工或多模态模型审阅。
2. **三大防呆预检与两阶段批次执行（`tools/op_agent/campaign.py`）**：
   - **第一阶段（`discover`）**：扫描候选算子池并跑单卡 `NPUGraph` 重放基线，自动执行三项预检：① 剔除图内耗时低于硬件底线的纯视图伪热点；② 与 `op_statistic.csv` 单次均值对账，将偏差过大的条目设为 `pending_review`（拦截将宿主底层算子错写为慢速原生算子的虚假基线）；③ 按 `单次基线耗时 × 层数` 计算每步预算并排序，输出 `campaign_batch.json` 批次审核单与 SVG 泳道图；
   - **第二阶段（`run`）**：经过批次审核勾选后，仅对 `status == "approved"` 的条目按独立子进程依次调用 `optimizer_loop.py`，汇总产出 `campaign_summary.json`。

---

## 5. 核心命令入口

```bash
# 1. 运行内环单算子自动调优循环
python3 -m tools.op_agent.optimizer_loop --case gdn_output_norm_fusion --max-iters 5

# 2. 外挂多算子热点扫描：折叠 timeline 子图 + 生成 SVG 泳道图 + 产出批次审核单
python3 -m tools.op_agent.campaign discover --out /tmp/campaign_batch.json --kernel-details-csv /path/to/kernel_details.csv --op-statistic-csv /path/to/op_statistic.csv

# 3. 批量运行批次审核单中已批准 (status=approved) 的多算子内环调优
python3 -m tools.op_agent.campaign run --batch /tmp/campaign_batch.json --max-iters 5 --summary-out /tmp/campaign_summary.json

# 4. 生成外环负载绑定与评测执行计划
python3 -m tools.op_agent.serving_loop plan --op gdn_output_norm_fusion --gate-env VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1 --phase decode

# 5. 解析 profiler op_statistic.csv 核验 L4 命中次数与全核占比
python3 -m tools.op_agent.serving_loop parse-l4 --csv /path/to/op_statistic.csv --kernel _output_norm_kernel --expected-calls 15360

# 6. 汇总多轮独立起服结果并回填 evidence.json 与 checksum.sha256
python3 -m tools.op_agent.serving_loop summarize-ab --workload random --off /tmp/off_1.json /tmp/off_2.json --on /tmp/on_1.json /tmp/on_2.json --update-op-dir ops/gdn_output_norm_fusion
```
