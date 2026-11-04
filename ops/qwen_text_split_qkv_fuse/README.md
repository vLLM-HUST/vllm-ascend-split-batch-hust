# Qwen Text Attention Split-QKV RoPE Fusion (`qwen_text_split_qkv_fuse`)

## 1. 背景与优化点

- **宿主源码位置**：`vllm_ascend/patch/worker/patch_qwen3_5.py` → `AscendQwen3NextAttention.forward`
- **配套 kernel**：`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`
- **算子接口**：`torch.ops.vllm.triton_split_qkv_rmsnorm_mrope`
- **开关**：`VLLM_HUST_SPLIT_QKV_ROPE_FUSED=1`（默认 `0`）

宿主多模态 RoPE 路径原生启用了 Triton 融合算子，但纯文本路径退化为离散多算子链路：
1. `qkv.split(...)`
2. `chunk(2)` 提取 Q 与 Gate
3. `q_norm` (GemmaRMSNorm)
4. `k_norm` (GemmaRMSNorm)
5. `rotary_emb` (`torch_npu.npu_mrope`)

本优化在纯文本 1D RoPE 场景下将上述离散链路接入已有高性能 Triton 融合 Kernel，单 Kernel 完成切分、归一化与旋转嵌入。

## 2. 为什么由宿主承载

`AscendQwen3NextAttention` 是宿主模型前向补丁类，处于模型执行链路核心内部，无法通过插件外部公开接口拦截。根据边界规范，严禁在运行树上动态打补丁，因此该改动必须下沉至宿主 Fork 源码承载。

## 3. 宿主 Fork 归属

- **仓库**：`git@github.com:vLLM-HUST/vllm-ascend-hust.git`
- **Base 分支**：`based/split-batch-baseline`
- **开发分支**：`work/qwen-text-split-qkv-fuse`（PR #67）
- **提交哈希**：`493ec93d243816217e15daaab480fbe073051981`
- **合并提交**：`d1014aadb`（已合入 `based/split-batch-baseline`）
