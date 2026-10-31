# GDN 卷积权重加载期预打包与转置外提 (GDN Conv1d Weight Pre-pack)

## 1. 背景与优化点
- **宿主位置**：`vllm_ascend/ops/gdn.py` 中的 `AscendGatedDeltaNetAttention`
- **原生痛点**：原生 GDN 实现中，`conv1d.weight` 的张量形状为 `[4096, 1, 4]`，在每一次前向（forward step）都需要执行一次 `.transpose(0, 1).contiguous()`，产生重复的转置与显存拷贝开销。
- **优化方案**：在权重加载期（`process_weights_after_loading`）将权重一次性物化为连续的 `[4, 4096]`（由环境变量 `VLLM_HUST_D4_CONV_PACK=1` 开启），并在 forward 内部直接复用预打包权重；关闭时自动安全回退原生逐步转置。

## 2. 架构落地与收益定性
- **为什么由宿主承载**：插件层仅能介入公开接口，无法侵入模型私有前向与内部权重结构。因此该优化直接下沉至宿主 Fork 源码，随镜像原生发布。
- **宿主 Fork 归属**：仓库 `vLLM-HUST/vllm-ascend-hust`，严格基于 **`based/image-baseline`** 基线分支（对应镜像 `74f0c0a27`），开发分支为 `work/issue20-w3f2-host`（Commit `deb5ffefe`）。
- **评测定性**：在 910B2 硬件上，单层转置开销从 308.62 us 降低到 0.20 us（直接复用预打包连续张量，加速比 >1500x）；全模型 30 层 GDN 每步推理净消除 ~9.25 ms 的纯拷贝与转置时延。
- **与插件解包的关系**：插件侧曾尝试通过 Triton 单 pass 解包优化（`ops/gdn_qkv_unpack`），但因内核发射时延出现负收益；生产中 GDN 的核心耗时瓶颈实际由本宿主卷积权重加载期预打包优化彻底解决。
