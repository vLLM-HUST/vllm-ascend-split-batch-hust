# op_agent — 算子自动调优循环入门

本目录是仓库的算子自动调优 Harness：用一个 smolagents `CodeAgent` 循环生成候选算子实现，
先过 L3 数值护栏、再测 L0 微基准，最优结果自动导出为 `ops/<operator>/` 标准交付物。
架构契约与权威规范见 [docs/op-agent-harness-spec.md](../../docs/op-agent-harness-spec.md)，
本 README 只解决两件事：怎么跑起来、API key 怎么接。

## 1. 目录组成

| 文件 | 职责 |
|---|---|
| `optimizer_loop.py` | 内环主入口：装配 agent 与工具、跑基线、维护最优记录、触发导出（支持 `--case` 与 `--max-iters`） |
| `evaluator.py` | 内环短路评估器：L3 数值校验 -> 动态形状零重编译门禁 -> L0 分桶计时（Eager 或 NPUGraph） |
| `campaign.py` | 外挂多算子批次插件：`discover` 生成带防呆预检的批次审核单，`run` 隔离调度获批算子 |
| `timeline_insight.py` | 时序子图折叠与可视化：将 `kernel_details.csv` 折叠为连续 Vector 夹心卡片并渲染泳道图 SVG |
| `serving_loop.py` | 外环整网评测入口：负载配对计划生成、整网贪心输出对拍、L4 `op_statistic` 解析与服务对比回填 |
| `workloads.py` | 标准服务负载目录（`random`、`sharegpt`、`long-prefix`）特征画像与算子适配映射 |
| `guard.py` | NPU 环境前置检查（卡数、torch_npu 可用性），fail-closed |
| `kb.py` | 本地 CANN/Ascend 知识库查询，带进程内缓存 |
| `cases/base.py` | `OpCase` 抽象契约（含多桶形状、非连续跨距视图、动态形状探针与负载配对） |
| `cases/__init__.py` | `CASE_REGISTRY` 注册表，通过 `get_case(case_name)` 动态加载 |
| `qwen35_shapes.py` | Qwen3.5-35B-A3B (TP=2) 的形状事实与权重标签 |
| `exporter.py` | 把内环胜出候选导出为 `ops/<op>/` 并保护已有的外环 L4/服务记录 |
| `custody.py` | 生成并校验 `checksum.sha256` 防漂移指纹，登记 `DELIVERY_INDEX.json` |

当前已注册的 case：`moe_shared_epilogue_decode`、`gdn_input_norm_decode`、
`gdn_norm_gate`、`gdn_output_norm_fusion`、`gdn_split_ba`、`gdn_qkv_norm`。

## 2. 循环怎么转

1. `guard.check_npu_environment` 做环境预检，打印卡数与 TP=2 是否就绪；
2. 先用 case 的基线实现跑一次 L0，记录基线值；
3. CodeAgent 每一步：
   - 用 `write_candidate_file` 把候选代码写到仓库内临时文件（按引用传递，不往对话里塞大段代码）；
   - 用 `evaluate_kernel(file_path=...)` 一次完成 L3 校验 + L0 计时，L3 不过直接短路返回错误；
   - 耗时优于历史最优就更新 `BEST_RECORD`；
4. 结束后若最优优于基线，`exporter` 落盘交付目录，`custody` 生成指纹并写 `DELIVERY_INDEX.json`。

## 3. 快速开始

### 3.1 零配置冒烟（不调 LLM）

不设任何环境变量时，`get_configured_model()` 返回 `MockKernelModel`，
回放固定候选序列，用于验证链路本身，不产生真实优化：

```bash
cd <仓库根目录>
python3 -m tools.op_agent.optimizer_loop
```

### 3.2 接入真实模型（API key）

模型接入完全由环境变量驱动，`get_configured_model()`（`optimizer_loop.py`）读取顺序如下：

| 变量 | 作用 | 缺省回退 |
|---|---|---|
| `OP_AGENT_MODEL_ID` | 模型名，例如 `qwen3-235b-a22b` | 未设置则退回 Mock 模型 |
| `OP_AGENT_API_BASE` | OpenAI 兼容服务地址，需带 `/v1` | 回退 `OPENAI_BASE_URL` |
| `OP_AGENT_API_KEY` | 鉴权 key | 回退 `OPENAI_API_KEY` |

**默认 key 的位置：`<仓库外的凭据环境变量文件>`（容器持久层，非仓内文件）。**
该文件由三个 `export OP_AGENT_*` 行组成。它符合"凭据不进 Git 仓库"的隔离规范，但有两点
边界事实需要知道：

- `/root` 是 PVC 持久卷（规格切换、Pod 重建不丢失），key 不会因容器重建而消失；
- 没有任何机制自动加载它：容器自愈流程（见
  [docs/container-and-plugin-boundary-spec.md](../../docs/container-and-plugin-boundary-spec.md) §2.2）
  不处理该文件，新 shell 也不会 source 它。本 README 是仓内唯一引用它的地方，
  属描述性引用、非功能依赖；跑循环前必须手动 `source`。

```bash
cd <仓库根目录>
source <仓库外的凭据环境变量文件>        # 载入 OP_AGENT_MODEL_ID / OP_AGENT_API_BASE / OP_AGENT_API_KEY
python3 -m tools.op_agent.optimizer_loop
```

校验是否加载成功（两条命令都应打印出模型名而非空）：

```bash
source <仓库外的凭据环境变量文件> && echo "$OP_AGENT_MODEL_ID"
python3 -c "import os; print(os.environ.get('OP_AGENT_MODEL_ID', 'NOT LOADED'))"
```

注意：忘了 source 不会报错，而是静默退回 Mock 模型；如果跑起来"每步都秒回且结果雷同"，
先怀疑 key 没加载。换成其他 OpenAI 兼容服务时，改写 `.env.local` 里的三个变量即可，
不需要动仓内代码。

### 3.3 指定算子 case 与衔接外环评测

内环支持直接通过 CLI 参数指定 `--case` 与 `--max-iters`：

```bash
python3 -m tools.op_agent.optimizer_loop --case gdn_output_norm_fusion --max-iters 5
```

内环通过后，使用外环 `serving_loop.py` 生成对应的受控负载（如 `random`）与真实长尾压力负载（`sharegpt`）评测计划，并在采集后汇总回填：

```bash
python3 -m tools.op_agent.serving_loop plan --op gdn_output_norm_fusion --gate-env VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1 --phase decode
```

## 4. 运行前提

- NPU 规格 `npu-2x`（两张 910B2）：真实计时走 NPUGraph replay，消除 launch 开销；
- 无卡 CPU 容器也能跑：guard 检测不到 NPU 时自动退化为 numpy 模拟，只验证链路与
  数值护栏，不产生可交付的计时结论；
- Python 依赖：`smolagents`、`numpy`（`torch`/`torch_npu` 可选）。

## 5. 交付与合规

- 只有 L3 全过且 L0 优于基线的候选才会写 `ops/<operator>/`；
- 导出后必须保持 `checksum.sha256` 一致；人工改动目录内文件后需用
  `custody.write_dir_checksum()` 重新生成指纹，否则 `tests/test_ops_delivery_policy.py`
  门禁不通过；
- 度量数字只落在 `evidence.json`，报告行文不写"关键词 + 数字"。

## 6. 验证命令

```bash
pytest -q tests/test_op_agent.py     # 本目录契约测试（模型装配、短路评估、指纹、case 注册）
```
