# 镜像基线栈（2026-10-23）

本轮基线 = 容器镜像运行时那套可服务组合，落在 `/root` 下两条 git 分支，供后续改动在其上生长。
它只代表"本机能服务且历史产出可复现"，不代表任何新栈配对的正确性。

## 锚点

| 组件 | 本地树 | 分支 | commit | 安装版本串 |
|---|---|---|---|---|
| core | `/root/host-trees/vllm-hust` | `based/image-baseline` | `f18cf803c5f63625e2c71253ddaf8b0bad0bad1a`（= vllm-hust tag `v1`） | `0.28.1.post1.dev143+gf18cf803c.empty` |
| 插件 | `/root/host-trees/vllm-ascend-base` | `based/image-baseline` | `4d2c82742024c9bed0faa4f98734ba2535e4845e` | `0.25.1rc2.dev126+hust.20260903.4.g4d2c82742` |

远端副本：`Raing5Days/vllm:based/image-baseline`、`Raing5Days/vllm-ascend-hust:based/image-baseline`，两端 hash 已核。

插件侧 `4d2c82742` = `74f0c0a272376412b51e1c1864803d5f3a0f1b5f` + 10 文件补丁集（打包 3 项 + 代码 7 项，
`git diff --stat HEAD~1` 为 653 插入 / 29 删除）。镜像侧同一位置的 scm 串带 `.d20260909` 脏标记，
与本基线只差该标记，树内容逐文件相同。

## 非 git 产物（必须随树保留，否则起服失败）

| 路径 | 体量 | 缺失症状 |
|---|---|---|
| `vllm_ascend/vllm_ascend_C.*.so`、`vllm_ascend/libvllm_ascend_kernels.so` | 909 KB / 442 KB | 扩展导入失败 |
| `vllm_ascend/_cann_ops_custom/` | 58 MB | `aclnn* not in libopapi.so` |
| `csrc/build/`（含 `libcust_opapi.so`）、`csrc/build_out/`、`csrc/output/` | 307 MB + 19 MB | 同上 |

来源：`/vllm-workspace/vllm-ascend`（镜像树）。均被 `.gitignore` 覆盖，不进入提交。

## 解释器

`/root/venvs/hust-base`：torch `2.13.0+cpu`、torch_npu `2.13.0rc1`、torchvision `0.28.0+cpu`、
torchaudio `2.11.0+cpu`、triton `3.5.0`、triton_ascend `3.2.2`、numpy `1.26.4`、transformers `5.14.1`。
两棵树均 `pip install -e`（core 用 `VLLM_TARGET_DEVICE=empty`，插件用 `COMPILE_CUSTOM_KERNELS=0` 复用预编译扩展）。

## 起服固定态

```bash
TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
/root/venvs/hust-base/bin/vllm serve /models/Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 --max-model-len 8192 --port 8000 \
  --no-enable-prefix-caching --dtype bfloat16 --served-model-name Qwen3.5-35B-A3B
```

两个 inductor 开关必需：缺任一项，图捕获阶段抛
`AssertionError: expected OutputCode, got <GraphModule>`（torch 2.13 AOT 自动微分缓存路径）。

## 验证证据

- `/health` → 200，首字准备约 6–7 分钟（含 32.85 GB/卡权重加载与 Triton warmup）。
- `POST /v1/completions`，prompt `The capital of France is`，`max_tokens=16`、`temperature=0`、`seed=1234`
  → `" Paris.\nThe capital of France is Paris.\nThe capital of France is"`，usage 5/16，
  与 P6 记录（`probes/20261023T093231Z_new_stack_stability/p6/`）逐字相同。

## 边界

- 本基线只覆盖"镜像运行时态 + 本机可服务"；插件 `vllm-ascend-hust @ 212a949e5`（`dev1104`）与该基线**不兼容**，
  其所需 API（`BlockTables(kernel_block_sizes=…)`、`kv_cache_utils._pool_bytes_per_block(vllm_config, groups)`、
  `PassConfig.enable_sp`）在任何单一公开版本上都不共存，另案处理。
- 后续任何算子改动从上述两个分支拉新分支，禁止直接改 `based/image-baseline`。
