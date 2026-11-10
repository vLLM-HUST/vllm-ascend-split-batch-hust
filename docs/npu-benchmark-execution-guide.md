# NPU 环境算子评测执行计划与操作手册

本指南为切换到带昇腾 NPU 卡（如 2×910B2）环境后，直接对仓内算子进行基准评测、采集真实性能数据并更新交付成果的**标准操作流程**。

---

## 1. 架构原则与隔离规范

1. **宿主与插件两层分工**：
   - 宿主修改已随宿主 Fork 仓库（`vLLM-HUST/vllm-ascend-hust` @ `work/issue20-w3f2-host`）直出，容器内**禁止**运行任何动态打补丁脚本。
   - 本仓仅承载 Python/Triton 算子、default-off 门控、调度和验收凭据。
2. **知识库与仓库严格隔离**：
   - 原始分析数据库、大体积 Profiler Trace（如 `<仓外结果目录>/`）和外部知识库（`<仓外知识库>`）**严禁**混入 Git 仓库。
   - Git 仓库内仅收录结构化结果：算子专属目录下的 `REPORT.md` 与 `evidence.json`，以及根目录索引 `DELIVERY_INDEX.json`。
3. **四级评测阶梯原则**：
   - 必须严格遵循 `L0 纯算子微基准 -> L3 数值等价护栏 -> L4 调用链路归因 -> 服务端到端实测` 阶梯，任何数值失败必须保持 default-off。

---

## 2. 切换到 NPU 规格后的环境初始化

在云工作台网页端停止实例，将规格更改为带卡规格（如 `npu-2x`）并启动。容器重建后按以下步骤进行环境自愈与状态核验：

### 第一步：容器自愈与环境恢复
```bash
# 执行自愈脚本：自动恢复 SSHD、用户私钥权限、挂载 editable 插件包
bash <容器自愈脚本>
```
自检输出需确认：
- `sshd: running`
- `插件包 vllm-ascend-split-batch: installed`（entry points 正常列出 9 个插件）
- `宿主层分工: 下沉至宿主 Fork 仓库，无运行时动态补丁`

### 第二步：硬件拓扑与驱动确认
```bash
# 确认昇腾卡与设备节点就绪
npu-smi info
ls -la /dev/davinci*
```

### 第三步：宿主源码与 Python 依赖检查
```bash
# 确认宿主 Fork 分支处于最新的开发基线
git -C $VLLM_ASCEND_REPO branch -v
# 确认 torch_npu、vllm、vllm_ascend 正常加载
python3 -c "import torch, torch_npu, vllm, vllm_ascend; print('Environment Stack OK')"
```

---

## 3. 分算子实测与数据采集指南

测试时，请在独立目录（如 `/tmp`）执行，避免在 `/vllm-workspace` 下直接运行造成包遮蔽。

### 算子 1：GDN 卷积权重加载期预打包
- **目录位置**：`ops/gdn_conv_pack/`
- **核心优化**：将 `conv1d.weight` `[4096, 1, 4]` 在加载期预打包为连续 `[4, 4096]`，消除 Decode 每步的 `transpose(0, 1)` 重复转置开销。
- **评测命令**：
  ```bash
  # 1. 运行 L3 数值核验与布局基准
  python3 tools/bench_matmul_m1_layout.py
  python3 tools/bench_gdn_l3.py
  
  # 2. 验证 L4 归因与端到端消除效果 (开启门控)
  export VLLM_HUST_D4_CONV_PACK=1
  pytest tests/test_bench_gdn_l3.py -v
  ```
- **预期数据**：Decode 阶段 960 次 Transpose 调用全部消除至 0 次，数值位级恒等。

---

### 算子 2：MoE Gate 权重 FP32 外提 (`gate-fp32`)
- **目录位置**：`ops/gate_fp32/`
- **核心优化**：预先缓存转置为 FP32 的 gate 权重，消除 40 层 MoE 架构每 Decode 步 40 次 `Cast[256, 2048]`。
- **评测命令**：
  ```bash
  # 1. 运行 L3 数值与外提逻辑验证
  export VLLM_HUST_GATE_FP32=1
  pytest tests/test_gate_fp32_plugin.py -v
  
  # 2. 测量 L0/L4 收益
  python3 tools/bench_moe_init_m1.py
  ```
- **预期数据**：每步前向稳定消除 40 次类型转换调用，无额外内存开销。

---

## 4. 全链路端到端 Profiling 采集与多算子热点发现计划

针对整网未见真实 Trace 产物的现状，通过外挂式采集与分析工具链完成从整网 Trace 到候选算子批次的闭环流转：

### 4.1 CPU-Only 准备阶段（工程就绪）
1. **Trace 解析与防呆预检演练**：
   - 验证 `tools/op_agent/timeline_insight.py` 的子图折叠逻辑与 SVG 泳道图渲染；
   - 验证 `tools/op_agent/campaign.py` discover 阶段的三项对抗性预检（剔除图内极小耗时伪热点、CSV 均值对账偏差过大拦截、每步预算排序）；
2. **自动化起服与压测调用封装**：
   - 验证 `workloads/run_workload.sh` 与 `tools/op_agent/serving_loop.py` 的参数组合（`random` 定长受控基准与 `sharegpt` 真实长尾压力分布）。

### 4.2 NPU 2x 实测采集阶段（带卡执行）
1. **真实整网 Trace 采集**：
   - 在双卡 910B2 TP=2 环境下启动全基线服务（所有门控置 0）；
   - 在 `random` 稳态解码（512 输入 / 256 输出 / 并发 32）与 `sharegpt` 长尾分布下，各采集一个代表性窗口（跳过预热，采集连续稳态步骤），生成 `kernel_details.csv` 与 `op_statistic.csv`；
   - 原始 Trace 一律输出至仓外持久层（如 `<仓外结果目录>/`），严禁混入 Git 仓库。
2. **外挂式子图聚类与批次审核单生成**：
   ```bash
   python3 -m tools.op_agent.campaign discover \
       --out /tmp/campaign_batch.json \
       --kernel-details-csv <仓外结果目录>/random_baseline/kernel_details.csv \
       --op-statistic-csv <仓外结果目录>/random_baseline/op_statistic.csv \
       --svg-out /tmp/timeline_hotspots.svg
   ```
   - 输出审查卡片（前后锚点、非锚点向量链、等待空泡、MTE2 片外搬运占比）；
   - 审阅生成的 `campaign_batch.json` 批次清单，仅对确认具有算子优化潜力的项标记 `status: "approved"`。

---

## 5. 挂起算子外环补齐与闭环更新流程

针对当前 `evidence.json` 中标记为 `pending` 的算子（`qwen_text_split_qkv_fuse`、`gdn_qkv_norm`、`gdn_norm_gate`、`gdn_split_ba`），按以下标准化流程在 NPU 2x 环境下推进落地：

### Step 1：执行 Serving L3 贪心对拍
通过 `tools/op_agent/serving_loop.py` 内置的 8 条固定问答提示词（`temperature=0.0`），分别采集关闭门控（OFF）与开启门控（ON）的生成文本，断言 100% 逐字严格一致。

### Step 2：解析 `op_statistic.csv` 核验 L4 归因与信噪比
```bash
python3 -m tools.op_agent.serving_loop parse-l4 \
    --csv <仓外结果目录>/eval_op/op_statistic.csv \
    --kernel <target_kernel_name> \
    --expected-calls <expected_count>
```
核验融合内核命中率达 100% 且零静默回退，评估其在整网所有算子总耗时中的占比是否跨过信噪比门槛。

### Step 3：独立起服 A/B 压测与成果自动回填
对关闭组与开启组各执行多轮独立起服平衡压测（主测 `random`，复核 `sharegpt`），并汇总回填交付证据与刷新指纹：
```bash
python3 -m tools.op_agent.serving_loop summarize-ab \
    --workload random \
    --off /tmp/off_1.json /tmp/off_2.json \
    --on /tmp/on_1.json /tmp/on_2.json \
    --update-op-dir ops/<operator>
```

### Step 4：更新交付索引与合规校验
将各算子的 `serving_status` 从 `pending` 调整为 `verified_faster` 或 `within_noise`，并运行合规测试校验：
```bash
pytest tests/test_benchmark_standard.py tests/test_json_hygiene.py tests/test_ops_delivery_policy.py -v
```
确保全量 JSON 校验、结构索引与防漂移门禁持续保持全绿。
