# GDN Part 3 Output Norm and Rearrange Fusion (`gdn_output_norm_fusion`)

## 1. 背景与优化点

- **宿主源码位置**：`vllm_ascend/ops/gdn.py` 与 `vllm_ascend/ops/gdn_epilogue.py`
- **算子接口**：`torch.ops.vllm_ascend.gdn_output_norm_rearrange`
- **开关**：`VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1`（默认 `0`）

在 Qwen3.5 模型的 GDN（Gated DeltaNet）层解码阶段，Part 3 输出端原始链路为：
1. `core_attn_out.reshape(-1, core_attn_out.shape[-1])`
2. `z.reshape(-1, z.shape[-1])`
3. `self.norm(core_attn_out, z)` (RMSNormGated)
4. `core_attn_out.reshape(z_shape_og)`
5. `rearrange(core_attn_out, "... h d -> ... (h d)")`
6. `self.out_proj(core_attn_out)`

经过物理内存布局推导与验证，直接分配目标形状为 2D 连续张量并将其作为算子写回地址，其每个元素物理位置与 einops rearrange 重排结果完全一致。
本优化使用 Triton 融合算子将上述多次 Python 视图重塑、动态归一化规约与重排完全合并为单步写入，直接输出连续 2D 张量给后续投影层。

## 2. 为什么由宿主承载

GDN 前向执行路径属于宿主私有前向层（`AscendGatedDeltaNetAttention.forward`），处于模型执行流的核心内部，无法通过插件公开扩展点非侵入式介入。根据两层边界规范，该逻辑在宿主 Fork 源码承载并提供 default-off 环境变量门控。

## 3. 宿主 Fork 归属

- **仓库**：`git@github.com:vLLM-HUST/vllm-ascend-hust.git`
- **Base 分支**：`based/split-batch-baseline`
- **开发分支**：`work/gdn-output-norm-fusion`
- **提交哈希**：`68c0790818dd15ff28b55d95713c2f0f4b3df66e`
