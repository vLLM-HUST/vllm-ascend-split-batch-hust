# GDN Part 3 Output Norm and Rearrange Fusion (`gdn_output_norm_fusion`)

## 1. 背景与优化点

- **宿主源码位置**：`vllm_ascend/ops/gdn.py` 与 `vllm_ascend/ops/gdn_epilogue.py`
- **算子接口**：`torch.ops.vllm_ascend.gdn_output_norm_rearrange`
- **开关**：`VLLM_HUST_GDN_OUTPUT_NORM_FUSED=1`（默认 `0`）

在 Qwen3.5 模型的 GDN（Gated DeltaNet）层前向计算中，Part 3 输出端原始链路为：
1. `core_attn_out.reshape(-1, core_attn_out.shape[-1])`
2. `z.reshape(-1, z.shape[-1])`
3. `self.norm(core_attn_out, z)` (`AscendRMSNormGated`)
4. `core_attn_out.reshape(z_shape_og)`
5. `rearrange(core_attn_out, "... h d -> ... (h d)")`
6. `self.out_proj(core_attn_out)`

经过物理内存布局推导与验证：
1. 直接分配目标形状为二维连续张量并将其二维视图作为算子写回地址，其每个元素物理位置与 `einops.rearrange` 重排结果完全重合；
2. Qwen3.5 非交错投影路径（`gqa_interleaved_layout=False`）下，`z` 来自 `mixed_qkvz` 列切片重塑，行跨距大于特征维但末维保持单位步长。算子通过二维行跨距寻址直接读取非连续 `z`，并将行数与跨距声明为 `do_not_specialize`，避免在分块预填充动态尺寸下触发重复编译。

## 2. 为什么由宿主承载

GDN 前向执行路径属于宿主私有前向层（`AscendGatedDeltaNetAttention.forward`），处于模型执行流的核心内部，无法通过插件公开扩展点非侵入式介入。根据两层边界规范，该逻辑在宿主 Fork 源码承载并提供 default-off 环境变量门控。

## 3. 宿主 Fork 归属

- **仓库**：`git@github.com:vLLM-HUST/vllm-ascend-hust.git`
- **Base 分支**：`based/split-batch-baseline`
- **开发分支**：`work/gdn-output-norm-fusion`
- **初始提交**：`68c0790818dd15ff28b55d95713c2f0f4b3df66e`（PR #69，合并提交 `0ee020516`）
- **后续修复提交**：`8499a55aa`（放宽非连续 `z` 跨距判定并消除动态行数重复编译）
