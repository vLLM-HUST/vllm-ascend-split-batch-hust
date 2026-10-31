# NPU 环境算子评测执行计划与操作手册

本指南为切换到带昇腾 NPU 卡（如 2×910B2）环境后，对仓内算子进行基准评测、采集性能数据并更新交付成果的操作流程。

---

## 1. 核心约束与运行前置

1. **两层分工**：宿主修改由宿主 Fork（`vLLM-HUST/vllm-ascend-hust`）承载，严禁在容器内运行动态打补丁脚本；本仓仅承载插件、算子、门控及交付凭据。
2. **端到端测试硬性约束（TP=2 双卡）**：
   - 目标主模型为 **Qwen3.5-35B-A3B**（40 层：30 GDN + 10 MoE）；
   - 该模型在 910B2 规格下**必须以双卡张量并行（Tensor Parallelism, TP=2）**运行；
   - **启动端到端压测前，必须运行 `npu-smi info` 确认两张卡均处于空闲状态（无其他高负载进程占用 HBM 与 AICore）**；若有单卡被占用，需等待其任务完成或与相关开发者确认。
3. **物理隔离**：大体积 Trace 和日志存放在 `/tmp` 或外部目录，禁止直接提交进 Git 仓。

---

## 2. 切换到 NPU 规格后的环境初始化

在云工作台网页端停止实例，将规格更改为带卡规格（如 `npu-2x`）并启动。容器重建后按以下步骤进行环境自愈与状态核验：

### 第一步：容器自愈与环境恢复
```bash
# 执行自愈脚本：自动恢复 SSHD、用户私钥权限、挂载 editable 插件包
bash /root/start-sshd.sh
```

### 第二步：硬件拓扑与双卡状态核验
```bash
# 确认昇腾卡与设备节点就绪，核验双卡是否全空闲
npu-smi info
```

### 第三步：宿主源码与 Python 依赖检查
```bash
python3 -c "import torch, torch_npu, vllm, vllm_ascend; print('Environment Stack OK')"
```

---

## 3. Qwen3.5-35B 端到端 (E2E TP=2) 评测流程

测试前务必确认 Device 0 与 Device 1 两张卡皆空闲。

### 3.1 端到端服务启动与 A/B 对比
- **基准组 (Baseline - Native)**：
  ```bash
  # 关闭所有优化开关
  unset VLLM_HUST_D4_CONV_PACK
  unset VLLM_HUST_GATE_FP32
  unset VLLM_HUST_FI_SAMPLING
  
  vllm serve /data/shared_models/Qwen3.5-35B-A3B \
      --served-model-name qwen35-35b \
      --tensor-parallel-size 2 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85 \
      --port 8350
  ```

- **优化组 (Optimized)**：
  ```bash
  # 开启验证项门控（例如开启 D4 卷积权重预打包与 Gate FP32 外提）
  export VLLM_HUST_D4_CONV_PACK=1
  export VLLM_HUST_GATE_FP32=1
  
  vllm serve /data/shared_models/Qwen3.5-35B-A3B \
      --served-model-name qwen35-35b \
      --tensor-parallel-size 2 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85 \
      --port 8350
  ```

### 3.2 客户端并发基准压测
另起终端使用 `vllm bench serve` 进行端到端请求压测：
```bash
vllm bench serve \
    --backend openai-chat \
    --model qwen35-35b \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --num-prompts 256 \
    --max-concurrency 64 \
    --port 8350 \
    --percentile-metrics ttft,tpot,itl
```

仓内已建立标准 Workload 目录 `workloads/`，包含 `random-input`、`sharegpt` 和 `long-prefix`。当前阶段主要使用受控 `random-input`；后续扩展压测可通过 `bash workloads/run_workload.sh <workload>` 直接执行。

---

## 4. 评测成果回填与闭环更新流程

采集到真实硬件性能数字后，按以下三步完成成果落盘：

### Step 1：更新各算子的 `evidence.json`
将硬件信息、测试 shape、L0 耗时比及端到端加速比更新至对应算子的 `ops/<operator>/evidence.json`。

### Step 2：更新各算子的 `REPORT.md`
在 `ops/<operator>/REPORT.md` 中填入规范结果表格，包含微基准对比、数值误差判定、L4 消除次数及端到端服务状态。

### Step 3：更新根目录索引 `DELIVERY_INDEX.json`
将各算子的 `serving_status` 调整为对应状态，并运行合规测试校验：
```bash
pytest tests/test_benchmark_standard.py tests/test_json_hygiene.py tests/test_document_custody.py -v
```
确保全量 JSON 校验与结构索引测试持续保持全绿。
