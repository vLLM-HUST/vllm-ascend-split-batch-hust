# GDN in_proj 输入投影权重加载期合并（宿主 Fork 承载）

## 1. 技术原理

Qwen3.5 GDN 层的输入投影由 `in_proj_qkvz` 与 `in_proj_ba` 两个权重矩阵组成，decode 每步对二者各执行一次 MatMulV2 再拼接结果。本优化在权重加载完成后、dynamo 编译/计算图捕获之前，把两个权重**一次性拼接合并**为单一权重矩阵，使前向每层每步只做一次 MatMulV2。合并后权重在推理期恒定，无每步开销。

## 2. 为什么必须改宿主

1. 宿主未提供权重加载后处理的公开接入点（Hook），插件无法无侵入拦截权重生命周期；
2. 本栈 serve 走 netloader，其 `load_model` 不经过 vllm model_loader 的 HpcModule 协议，即使层实现了 `process_weights_after_loading` 也不会被 model_loader 触发，必须在宿主 `NPUModelRunner.load_model` 中显式调用；
3. 合并必须发生在 dynamo 编译/图捕获之前，否则合并后的权重替换会使已捕获图失效。

因此该算子只能下沉到宿主 Fork（`vLLM-HUST/vllm-ascend-hust`），随宿主镜像原生发布；插件仓不承载实现。

## 3. 涉及的宿主文件与方法

| 文件 | 改动 |
|---|---|
| `vllm_ascend/ops/gdn.py` | `AscendGatedDeltaNetAttention.process_weights_after_loading`：合并 `in_proj_ba` 与 `in_proj_qkvz` 并替换权重（`_w3f2_merged` 标记防重复合并） |
| `vllm_ascend/worker/model_runner_v1.py` | `NPUModelRunner.load_model`：`get_model` 之后遍历 `QwenGatedDeltaNetAttention` 实例显式调用上述 hook |
| `vllm_ascend/patch/worker/patch_qwen3_5.py` | 把 `process_weights_after_loading` rebind 到 Qwen3.5 GDN patch target（缺失则 hook 不会被调用） |

## 4. 开关与 fail-closed 语义

- **开关**：`VLLM_HUST_GDN_MERGE_INPROJ=1`（default-off；未设置时宿主行为与 stock 严格一致，零开销零副作用）。
- **fail-closed**：runner 侧调用整体被 try/except 包裹，任何合并失败只打 warning 不阻塞起服；合并异常时层保留原有两个权重矩阵，行为回退 stock。
- **判类依据**：runner 遍历时按 vllm 核心类 `QwenGatedDeltaNetAttention` 判型（ascend 方法经 patch rebind 上去，实例本身是核心类）。

## 5. 交付位置与证据

- 宿主仓库：`git@github.com:vLLM-HUST/vllm-ascend-hust.git`
- 基线分支：`based/split-batch-baseline`（头 `ad82d77cb`）
- 开发分支：`work/w3f2-inproj-merge`（宿主 PR #63），提交 `fcbc4458bb8afcc37441e602921ed45dcbe5e775`（cherry-pick 自旧分支 `work/issue20-w3f2-host` 的 `ce807cd7c`，树逐字节一致）
- 结构化证据：[evidence.json](evidence.json)；验收报告：[REPORT.md](REPORT.md)
- 配套插件入口：`gdn-merge-inproj` general plugin 仅承担注册与 fail-closed 探测，自身不合并权重（见 [docs/delivery-inventory.md](../../docs/delivery-inventory.md)）。
