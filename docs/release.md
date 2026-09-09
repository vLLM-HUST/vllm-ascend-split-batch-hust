# 发布流程与检查表

基于 [bidkv 打包发布指南](https://github.com/vLLM-HUST/vllm-hust-docs/blob/main/operations/bidkv-packaging-and-release-guide.md)
裁剪,补充本插件特有事项。Bundle ID: `org.vllm-hust.split-batch-full-graph`。

## 0. 状态纪律(先于一切)

- manifest 保持 `import_only`,直到三项证据齐备:default-off 零回归冒烟、
  与参考/基线的正确性对齐、TPOT/latency 性能对比(见工作区 AGENTS.md,
  工作流见 [agent-guide.md](agent-guide.md))。
- 翻 `active` 前,`activation.environment` 必须是可注入的真实值
  (`"1"`,由 `tests/test_manifest.py` 保证格式);default-off 语义由
  "不 enable 就不注入 + `load()` 内部门控" 双层保证。
- `host.version_range` 钉住**实际验证域**(当前 `>=0.23.0rc1,<0.24`),禁止
  `>=0`。宿主装的是 `vllm-ascend==0.23.0rc1`,packaging 语义下 prerelease
  不属于 `>=0.23.0,<0.24`,写后者会让 `extension check` 判 INCOMPATIBLE。
  区间收窄/放宽前先过 §4 清单并留档验证环境(见 §0.1)。
- `protocols[].version_range` 为 `null`:这四个协议(`vllm.graph.runtime-key` /
  `vllm.forward.split-context` / `vllm.ascend.graph-pool` /
  `vllm.worker.split-executor`)是本仓库单方面遵守的弱契约,**宿主不独立
  版本化**——本宿主不存在 `vllm.plugins.contracts`,manager 的
  `_detect_protocol_versions()` 返回 `{}`。声明 `>=1,<2` 只会让 `run` 因
  "protocol version is unavailable" 拒启(与真实兼容性无关);置 `null` 后
  由宿主区间 + 验收证据共同约束,manager 会记录一条 "not independently
  versioned" 证据。

### 0.1 已验证域留档(F1)

| 项 | 值 |
|---|---|
| `host.version_range` | `>=0.23.0rc1,<0.24` |
| vllm-ascend | `0.23.0rc1`,`f4a08bddd`(`v0.23.0rc1` tag) |
| vllm | `0.23.0+empty`,`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` |
| 验证环境 | Python 3.12.13 / torch 2.10.0+cpu / torch_npu 2.10.0.post2 / CANN 9.0.1 / 910B2 |
| 验证证据 | `extension check` → compatible;default-off serve 冒烟;`flashinfer-migration/cascade-evidence/EVIDENCE.md` |

## 1. 版本与构建

- 版本在 `pyproject.toml` 与 manifest `extension_version` 两处出现,
  一致性由 `test_extension_version_matches_distribution_version` 守护;
  发新版本时两处同步改(PyPI 不允许覆盖同版本文件)。
- 构建与产物检查:

```bash
cd /vllm-workspace/vllm-ascend-split-batch-hust
git status --short && git rev-parse HEAD   # 记录发布 commit,工作树须干净
rm -rf dist && pip wheel . --no-deps -w dist
python -m zipfile -l dist/*.whl            # 必须含 manifest JSON
python -c 'import zipfile,glob; z=zipfile.ZipFile(glob.glob("dist/*.whl")[0]); \
  n=next(x for x in z.namelist() if x.endswith("entry_points.txt")); print(z.read(n).decode())'
# 必须同时含 vllm_hust.extension_bundles 与 vllm.general_plugins 两个入口
```

## 2. 隔离安装冒烟

```bash
python -m venv /tmp/sb-release-smoke
/tmp/sb-release-smoke/bin/pip install --no-deps dist/*.whl
/tmp/sb-release-smoke/bin/python -c \
  'from importlib.metadata import version; print(version("vllm-ascend-split-batch"))'
```

与 vllm/vllm-hust-ext 同环境时,追加 `vllm-hust-ext extension list` 应能发现
本 bundle;`inspect` 的 `activation_ready` 反映当前 `implementation[].status`
(cascade carrier 已 `active` → `true`;planner 仍 `import_only`)。

## 3. 启用验证(翻 active 后执行)

按指南阶梯执行:`extension check` → `status` → `run --dry-run` 预览注入的
environment/additional_config → 正式 `vllm-hust-ext run -- vllm serve ...`。
启动日志应出现 extension startup snapshot 与 cascade 门控生效的证明;
`/health` 通过;功能/性能与 baseline 对比验收。`run` 拒绝启动的排查见
[pitfalls.md](pitfalls.md) §3.4。

## 4. 宿主升级核对清单

`host.version_range` 升级前,逐项核对 monkeypatch/公开面签名(细节见
HOST_CONTRACT.md),差异收敛在 `cascade_runner_patch.py`:

- [ ] `vllm.v1.cudagraph_dispatcher.add_cudagraph_key` / `dispatch`
- [ ] Runner: `_capture_cudagraphs` / `_warmup_and_capture` /
      `_determine_batch_execution_and_padding` / `_model_forward` /
      `_update_full_graph_params_if_needed`
- [ ] vllm-ascend: `update_full_graph_params` / `get_graph_params` / `GraphParams`
- [ ] `ACLGraphWrapper` variant-entry 表结构(标准 `BatchDescriptor` 键)
- [ ] `vllm.general_plugins` 加载时机未变
- [ ] fi_sampling 面:`vllm_ascend.sample.sampler.AscendTopKTopPSampler` 名称与
      `forward_native(logits, generators, k, p)` 签名、`AscendSampler.__init__`
      的构造点、`vllm.envs.VLLM_BATCH_INVARIANT` 与
      `get_ascend_config().enable_reduce_sample` / `enable_async_exponential`

## 5. 检查表

**构建前**

- [ ] `pytest -q` 全绿、`ruff check .` 通过
- [ ] 版本两处一致(测试守护)
- [ ] 发布 commit 已记录、工作树干净

**构建后**

- [ ] wheel 含 manifest JSON、发行元数据、两个 entry point
- [ ] 隔离安装冒烟通过
- [ ] `vllm-hust-ext extension list` 能发现(同环境时)

**翻 active 时(追加)**

- [ ] 三项证据已归档(default-off 冒烟 / 正确性 / 性能)
- [ ] `activation.environment` 值复核(enable=注入 `"1"`)
- [ ] `run --dry-run` 注入结果正确、服务健康检查通过
- [ ] 记录已验证的宿主版本/commit 与运行环境(Python、设备、镜像)
