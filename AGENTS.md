# AGENTS.md — vllm-ascend-split-batch-hust

vllm-ascend 宿主的 default-off 插件壳:cascade 两段式 decode(eager+图模式)
与 split-batch 规划契约。算子在仓库外 `ascend_kernel` wheel,依赖单向。

## 硬约束

1. 宿主 `/vllm-workspace/vllm-hust`、`/vllm-workspace/vllm-ascend-hust` **只读**;
   宿主能力只走官方公开面(general_plugins entry / `--additional-config` /
   `--worker-cls` / 官方 CLI+env)。不注册 `vllm.platform_plugins`。
   (旧基线 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend` 已于 2026-09-10 删除。)
2. default-off:新能力必须 env-gated 且默认与原生行为零差异。
3. bundle 保持 `import_only` 直到三项证据齐(default-off 冒烟/正确性/性能);
   `host.version_range` 禁止写 `>=0`。
4. `provenance/` 只读;提取代码保留 Huawei 版权头与溯源。
5. **不要以 `/vllm-workspace` 为 CWD 跑任何 python/vllm 命令**(目录会遮蔽
   真实 vllm 包)。
6. **cohort 与归属(2026-09-28/29 上游裁定,长期核心约束)**——详见
   [docs/cohort-and-ownership.md](docs/cohort-and-ownership.md):
   - 中央团队只保留"分类与发布边界";**下列四项由本仓 owner 负责**:
     ① independent cohort definition;② kernel-wheel licensing/distribution;
     ③ NPU evidence;④ performance claims。
   - 任何正确性/性能声明**必须带 cohort 边界**:`vllm-ascend` 0.25.x 线、
     Qwen2.5 真模型与替身、每请求 1 个 query row、APC + `FULL_AND_PIECEWISE`。
     **不得外推**到 Qwen3.5 hybrid TP2 / native MTP2 / chunked prefill。
   - MTP 是**边界不是支持**:任何 `speculative_config` fail-closed;该配置下
     "开 cascade"跑的是原生路径,不构成 cascade 效果。
   - "进统一 cohort"需先有 k+1 verification rows 的 dual-pad/graph bucket 合同
     + 该配置真实 replay correctness;**当前未立项**,未经裁定不得绕过上述守卫。

## 常用命令

```bash
cd /vllm-workspace/vllm-ascend-split-batch-hust
pytest -q && ruff check .        # CPU 层门槛,提交前必过
python -m pip install -e ".[test]"
vllm-hust-ext extension inspect org.vllm-hust.split-batch-full-graph
```

## 知识库

规范、架构、发布、踩坑、进度全部在 [docs/README.md](docs/README.md),
按其阅读顺序走;改行为必须同步对应权威文档(分工表在 docs/README.md)。
