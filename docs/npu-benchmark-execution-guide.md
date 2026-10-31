# NPU 环境算子评测执行计划与操作手册

本指南为切换到带昇腾 NPU 卡（如 2×910B2）环境后，直接对仓内算子进行基准评测、采集真实性能数据并更新交付成果的**标准操作流程**。

---

## 1. 架构原则与隔离规范

1. **宿主与插件两层分工**：
   - 宿主修改已随宿主 Fork 仓库（`vLLM-HUST/vllm-ascend-hust` @ `work/issue20-w3f2-host`）直出，容器内**禁止**运行任何动态打补丁脚本。
   - 本仓仅承载 Python/Triton 算子、default-off 门控、调度和验收凭据。
2. **知识库与仓库严格隔离**：
   - 原始分析数据库、大体积 Profiler Trace（如 `/root/prof/`）和外部知识库（`/root/cannbot-knowledge`）**严禁**混入 Git 仓库。
   - Git 仓库内仅收录结构化结果：算子专属目录下的 `REPORT.md` 与 `evidence.json`，以及根目录索引 `DELIVERY_INDEX.json`。
3. **四级评测阶梯原则**：
   - 必须严格遵循 `L0 纯算子微基准 -> L3 数值等价护栏 -> L4 调用链路归因 -> 服务端到端实测` 阶梯，任何数值失败必须保持 default-off。

---

## 2. 切换到 NPU 规格后的环境初始化

在云工作台网页端停止实例，将规格更改为带卡规格（如 `npu-2x`）并启动。容器重建后按以下步骤进行环境自愈与状态核验：

### 第一步：容器自愈与环境恢复
```bash
# 执行自愈脚本：自动恢复 SSHD、用户私钥权限、挂载 editable 插件包
bash /root/start-sshd.sh
```
自检输出需确认：
- `sshd: running`
- `husk 插件: installed`（entry points 正常列出 9 个插件）
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
git -C /root/host-trees/vllm-ascend branch -v
# 确认 torch_npu、vllm、vllm_ascend 正常加载
python3 -c "import torch, torch_npu, vllm, vllm_ascend; print('Environment Stack OK')"
```

---

## 3. 分算子实测与数据采集指南

测试时，请在独立目录（如 `/tmp`）执行，避免在 `/vllm-workspace` 下直接运行造成包遮蔽。

### 算子 1：GDN 卷积权重转置外提 (D4)
- **目录位置**：`ops/gdn_conv_pack/`
- **核心优化**：将 `conv1d.weight` `[4096, 1, 4]` 预打包为连续 `[4, 4096]`，消除 Decode 每步的 `transpose(0, 1)` 开销。
- **评测命令**：
  ```bash
  # 1. 运行 L3 数值核验与布局基准
  python3 /root/vllm-workspace/vllm-ascend-split-batch-hust/tools/bench_matmul_m1_layout.py
  python3 /root/vllm-workspace/vllm-ascend-split-batch-hust/tools/bench_gdn_l3.py
  
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
  python3 /root/vllm-workspace/vllm-ascend-split-batch-hust/tools/bench_moe_init_m1.py
  ```
- **预期数据**：每步前向稳定消除 40 次类型转换调用，无额外内存开销。

---

### 算子 3：FlashInfer Attention Demask (`fia-demask`)
- **目录位置**：`ops/fia_demask/`
- **核心优化**：在纯 Decode（Q_S=1）场景跳过全 1 的注意力掩码张量准备与核函数加载。
- **评测命令**：
  ```bash
  export VLLM_HUST_FIA_DEMASK=1
  pytest tests/test_fia_demask.py -v
  ```
- **预期数据**：数值恒等；在端到端并发下测量微小加速比，若在噪声范围内（Within noise），则如实记录结论。

---

### 算子 4：FlashInfer Sampling Triton 移植 (`fi-sampling`)
- **目录位置**：`ops/fi_sampling/`
- **核心优化**：使用针对昇腾 Triton 优化的采样算子替换分立的 Top-P / Top-K 宿主原生实现。
- **评测命令**：
  ```bash
  # 使用官方预制驱动脚本（卡 7 隔离运行，离线模式）
  cd /tmp && FISAMPLING_CARD=0 bash /root/vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_fisampling_refresh_e2e.sh
  ```
- **数据回填**：从生成的日志和 json 提取当前 0.28 镜像与 Qwen3.5 栈下的实际速度提升。

---

### 算子 5：GELU and Mul Triton 融合算子 (`fi-gelu`)
- **目录位置**：`ops/fi_gelu/`
- **核心状态**：当前已知主模型 Qwen3.5 采用 SwiGLU 结构，该插件为负收益/不适用，保持 default-off。
- **评测命令**（如需在支持模型如 Gemma 下复测）：
  ```bash
  cd /tmp && bash /root/vllm-workspace/vllm-ascend-split-batch-hust/bench/bench_w31_e2e.sh
  ```
- **数据回填**：保持其负收益与适用边界的客观记录，不强行主张性能加速。

---

## 4. 评测成果回填与闭环更新流程

采集到真实硬件性能数字后，按以下三步完成成果落盘：

### Step 1：更新各算子的 `evidence.json`
将最新的硬件信息、测试 shape、L0 耗时比、L4 命中次数及端到端加速比更新至对应算子的 `ops/<operator>/evidence.json`。例如：
```json
"l0_microbenchmark": {
  "status": "passed",
  "speedup": 1.12
},
"serving_end_to_end": {
  "tested": true,
  "speedup": 1.04
}
```

### Step 2：更新各算子的 `REPORT.md`
在 `ops/<operator>/REPORT.md` 中填入规范结果表格，包含微基准对比、数值误差判定、L4 消除次数及端到端服务状态。

### Step 3：更新根目录索引 `DELIVERY_INDEX.json`
将各算子的 `serving_status` 从 `pending_npu_refresh` 调整为 `measured_speedup`、`within_noise` 或 `negative_speedup`，并运行合规测试校验：
```bash
pytest tests/test_benchmark_standard.py tests/test_json_hygiene.py -v
```
确保全量 JSON 校验与结构索引测试持续保持全绿。
