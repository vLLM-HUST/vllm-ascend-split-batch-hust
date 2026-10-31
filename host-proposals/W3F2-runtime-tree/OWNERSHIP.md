# W3F2: GDN 输入投影权重合并（已下沉至宿主 Fork）

## 1. 方案目标

- **修改文件**：
  - `vllm_ascend/ops/gdn.py`：实现 `process_weights_after_loading` 回调函数；
  - `vllm_ascend/worker/model_runner_v1.py`：在加载权重后调用该合并逻辑；
  - `vllm_ascend/patch/worker/patch_qwen3_5.py`：适配 Qwen3.5 模型层的权重重绑定。
- **核心机制**：在权重加载完成后、计算图捕获之前，对 `QwenGatedDeltaNetAttention` 的 `in_proj_ba` 与 `in_proj_qkv` 执行一次性合并。由环境变量 `VLLM_HUST_GDN_MERGE_INPROJ=1` 门控（默认关闭）。
- **为什么改宿主**：宿主官方未提供权重后处理的公开接入点（Hook），外部插件无法无侵入拦截，必须在宿主模型加载流程中增加调用。

## 2. 最终归宿与交付状态

- **已下沉至宿主 Fork 源码**：
  - 宿主仓库：`vLLM-HUST/vllm-ascend-hust`
  - 所在分支：`work/issue20-w3f2-host`（基于基线 `based/image-baseline` / Commit `74f0c0a`，Commit `ce807cd7ce972b9a410663cebe10b7ea25fb2a31`）
- **本目录性质**：
  - 历史补丁原件仅作溯源参考备查，插件仓内不执行任何运行时补丁操作。
