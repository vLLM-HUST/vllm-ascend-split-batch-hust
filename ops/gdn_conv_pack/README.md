# GDN Conv1d Weight Pre-pack & Transpose Hoist (D4)

## 1. 背景与优化点
- **宿主位置**：`vllm_ascend/ops/gdn.py` 中的 `AscendGatedDeltaNetAttention`
- **原生痛点**：原生 GDN 实现中，`conv1d.weight` 的张量形状为 `[4096, 1, 4]`，在每一次前向（forward step）都需要执行一次 `.transpose(0, 1).contiguous()`，产生重复的转置与显存拷贝开销。
- **优化方案**：在权重加载期（`process_weights_after_loading`）将权重一次性物化为连续的 `[4, 4096]`（由环境变量 `VLLM_HUST_D4_CONV_PACK=1` 开启），并在 forward 内部直接复用预打包权重；关闭时自动安全回退原生逐步转置。

## 2. 架构落地
本改动属于宿主源码修改，已下沉至宿主 Fork 仓库 `vLLM-HUST/vllm-ascend-hust`（分支 `work/issue20-w3f2-host`，Commit `deb5ffefe`），随镜像原生发布。
