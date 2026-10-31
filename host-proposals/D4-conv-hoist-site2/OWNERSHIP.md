# D4: GDN 卷积权重转置外提与连续打包（已下沉至宿主 Fork）

## 1. 方案目标

- **修改文件**：宿主 `vllm_ascend/ops/gdn.py`。
- **核心机制**：在模型加载期将卷积权重 `conv1d.weight`（形状 `[4096, 1, 4]`）直接预打包为内存连续的 `[4, 4096]` 格式，由环境变量 `VLLM_HUST_D4_CONV_PACK=1` 门控（默认关闭）。
- **优化收益**：消除推理前向传播中每一步反复执行的 `transpose(0, 1)` 转置开销（在测试窗口中消除 960 次转置调用）。未开启时自动回退为原生逐步转置。

## 2. 最终归宿与交付状态

- **已下沉至宿主 Fork 源码**：
  - 宿主仓库：`vLLM-HUST/vllm-ascend-hust`
  - 所在分支：`work/issue20-w3f2-host`（基于基线 `based/image-baseline` / Commit `74f0c0a`，Commit `deb5ffefe32b53c61ae6193ff4ee4368cf1889ba`）
- **独立算子报告与索引**：
  - 本优化算子的专属报告与证据已按新标准归档至 [ops/gdn_conv_pack/](computer:///root/vllm-workspace/vllm-ascend-split-batch-hust/ops/gdn_conv_pack)。
- **本目录性质**：
  - 仅作为该优化的原始修改记录快照备查，插件仓内不执行任何运行时补丁操作。
