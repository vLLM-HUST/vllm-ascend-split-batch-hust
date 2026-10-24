# 0.31 配对判定（2026-10-24）

结论先行：插件 `vllm-ascend-hust @ 212a949e5`（自报 `0.25.1rc2.dev1104+hust.20260903.4.g212a949e5`）
**不存在与之配对的单一公开 core 提交**。任何"换个 tag 再试"的做法都不成立，重开必须有
同源 core commit。

## 一、212a949e5 的 API 期望 vs 公开 core

| 锚点 | 212a949e5 期望 | 官方 v0.31.0 | 官方 v0.32.0 | vllm-hust `5bc5ad2cb`（0.31.1 线） |
|---|---|---|---|---|
| `kv_cache_utils._pool_bytes_per_block` | `(vllm_config, groups)` | `(groups)` | `(vllm_config, groups)` | `(groups)`（两参版本在 `kv_cache_planning.py`） |
| `BlockTables.__init__` | 含 `kernel_block_sizes` + `dcp_sharded` | 无 `kernel_block_sizes` | 无 | 含 `kernel_block_sizes`，无 `dcp_sharded` |
| `PassConfig.enable_sp` / `sp_min_token_num` / `fuse_gemm_comms` | 必需 | 无 | 无 | 无 |
| `vllm.v1.outputs.RoutedExpertsLists` | 不使用 | 无 | 无 | 无（0.28 基线有） |
| `vllm.transformers_utils.configs.deepseek_v4` | 需要 | 无 | 无 | 无 |
| `kv_cache_utils._approximate_gcd` | 需要 | 无 | 无 | 无（已迁到 `kv_cache_planning`） |

- 全历史扫描：vllm-hust 里同时满足前两行的提交共 27 个，**全部落在 2026-08-17 ~ 08-20**。
- `PassConfig.enable_sp` 在**任何**公开 vllm 版本都不存在（v0.31.0 的 4 处命中是
  `enable_sparse_*` 前缀误匹配），属 fork 侧私有字段；`model_runner_v1.py` 里作者自己也写了
  `if not hasattr(pass_config, "enable_sp")` 的兼容分支，佐证它依赖一个非上游字段。
- 结论：`212a949e5` 是对着某个**内部拼装树**构建的，不能靠 checkout 复现。

## 二、备选路线：0.31 core + 镜像插件（74f0c0a27）

0.31 core（vllm-hust `5bc5ad2cb`）与镜像插件之间的 core API 分歧共 5 处，按 "437 个插件模块"
扫描量化：

| 分歧 | 影响模块 | 性质 |
|---|---|---|
| `configs.deepseek_v4` 缺失 | 20 | DeepSeek-V4 专用，Qwen3.5 路径用不到 |
| `v1.worker.gpu.spec_decode.autoregressive` 缺失 | 4 | 投机解码 |
| `async_copy_to_gpu`（buffer_utils 改名/移动） | 3 | 小改 |
| `RoutedExpertsLists`（0.31 删除） | 2 | 小改 |
| `get_kv_cache_groups`（迁至 `kv_cache_planning.py`，且上游 #61183 重构了规划逻辑） | 2 | **不可控**：昇腾补丁重写的正是规划内部实现，前向适配等于重写该补丁 |

另有 `mooncake` / `xlite` / `ucm` / `acl` / `flash_attn_npu_v3` 属可选三方件，不计入。

## 三、已验证可复用的环境事实

| 项 | 结论 |
|---|---|
| torch 2.14 栈 | `torch 2.14.0+cpu` + `torch_npu 2.14.0` 在 CANN 9.1.0 上可用（2 卡、fp32 matmul 误差 5e-5） |
| triton | `triton_ascend 3.6.0` 的 `triton/backends/ascend/utils.py:709` 硬编码 `-std=c++17`，torch 2.14 头文件要求 C++20 → 需改 `-std=c++20` |
| 插件 CMake | 顶层 `CMAKE_CXX_STANDARD` 必须留 17（AscendC 用的老 GCC 不支持 `gnu++20`）；只把 `vllm_ascend_C` 单目标提到 20 |
| 插件依赖钉 | 需改为 torch 2.14.0 / torch-npu 2.14.0 / triton-ascend 3.6.0；torchvision 取 `0.29.1+cpu`、torchaudio 取 `2.11.0+cpu`（须走 PyTorch CPU 索引） |
| 编译结果 | 0.31 分支上 `vllm_ascend_C` 与 `libvllm_ascend_kernels` 已成功编出，真正卡住的只有 Python 侧 API 前向适配 |

保留下来的探索件（未删）：`/root/venvs/hust-031`（3.3 GB，torch 2.14 栈 + 已编扩展），
可与 `/root/host-trees` 下的基线并存。日志归档在
`/root/包体恢复/records/probes/20261024T023000Z_v031_pairing/`。

## 四、重开条件

1. 取得同源 core commit（同门环境的 `git -C <vllm 树> log -1 --format=%H`）——这是唯一
   能保证"和他跑的一样"的路径；
2. 无该 commit 时不得再逐个 tag 试配对；
3. 若改走"0.31 core + 前向适配"，先出 kv cache 规划补丁的重写方案与验收门（L3 必须 8/8），
   再动代码。
