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
- `host.version_range` 钉住**实际验证域**(当前
  `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27` 点钉,2026-09-11 起),禁止
  `>=0`。历史教训:packaging 语义下 prerelease 不属于
  `>=X.Y.Z,<next` 形式的下界(如 `0.23.0rc1 ∉ >=0.23.0`),区间写法必须对照
  实装版本核 packaging 判定。区间收窄/放宽前先过 §4 清单并留档验证环境
  (见 §0.1)。
- `protocols[].version_range` 为 `null`:这四个协议(`vllm.graph.runtime-key` /
  `vllm.forward.split-context` / `vllm.ascend.graph-pool` /
  `vllm.worker.split-executor`)是本仓库单方面遵守的弱契约,**宿主不独立
  版本化**——本宿主不存在 `vllm.plugins.contracts`,manager 的
  `_detect_protocol_versions()` 返回 `{}`。声明 `>=1,<2` 只会让 `run` 因
  "protocol version is unavailable" 拒启(与真实兼容性无关);置 `null` 后
  由宿主区间 + 验收证据共同约束,manager 会记录一条 "not independently
  versioned" 证据。

### 0.1 已验证域留档(F1,2026-09-11 更新为现行基线)

| 项 | 值 |
|---|---|
| `host.version_range` | `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`(点钉) |
| vllm-ascend | `0.25.1rc2.dev125+hust.20260903.4`,`74f0c0a272376412b51e1c1864803d5f3a0f1b5f`(main) |
| vllm | `0.28.1.post1.dev143+gf18cf803c.empty`,`f18cf803c5f63625e2c71253ddaf8b0bad0bad1a`(vllm-hust release v1) |
| 验证环境 | Python 3.12.14 / torch 2.13.0+cpu / torch_npu 2.13.0rc1 / CANN 9.1.0 (`/usr/local/ascend91`) / 910B2 |
| 验证证据 | `extension check` → compatible;§3 启用验证(`knowledge/evidence/cascade/section4-active-enablement.md`);正确性 `section2-correctness.md`(真模型 6/64);历史域(0.23.0rc1 / CANN 9.0.1,已退役)留档于 `EVIDENCE.md` 与 c3-legacy 产物 |

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
- [ ] rope_fix 面(`tests/test_rope_fix_drift.py` 跑一遍;它的**反向断言**会
      自动红,`upstream fixed defect ①/②/③` = 该覆写已变冗余,`anchor drifted`
      = seam 挪位需重审 HOST_CONTRACT.md「rope_fix component」小节;
      宿主树找不到时**直接红,不 skip**):
      三缺陷锚点(`REGISTERED_ASCEND_OPS` 仍缺 `Llama3RotaryEmbedding` /
      `AscendMRotaryEmbedding.forward_triton` 仍是 8 参 `triton_mrope` 调用 /
      `AscendYaRNRotaryEmbedding.__init__` 的 `truncate` 缺省仍是 `False`)、
      `register_ascend_customop`(含 `worker.py` 仍以 `from ... import` 直接绑定
      并在 `NPUWorker.__init__` 调用它)、`op_registry_oot` 的实例化期读取与重名断言、
      `Llama3RotaryEmbedding` 10 参构造签名、`AscendRotaryEmbedding.forward_oot` 签名、
      `rotary_embedding.triton_mrope` 存在(当 fork 报 `HAS_TRITON`)且为 9 参;
      ①的委托目标跟随 `op_registry_oot["RotaryEmbedding"]`——进入 310P 验证域时,
      需补一条 310P 真机核对(本机 910B2 只有构造级 mock 证据)

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

## 6. 默认关闭时的宿主隔离契约(2026-09-26)

**判据**:`load()` 在 `VLLM_ASCEND_ENABLE_CASCADE_DECODE` 未置 1 时,除了打一条
启动标记行,不得对宿主进程做任何改动。逐项禁止:

| 项 | 关闭时 | 依据 |
|---|---|---|
| 注入 `vllm_ascend.envs.env_variables` | 不注入 | `cascade_plugin._cascade_gate_env()` 先读环境变量,不依赖注入结果 |
| import 宿主模块(`vllm_ascend.ops` / `attention_v1`) | 不 import | 同上早退 |
| 安装 fail-open shim(policy_factory / spec_decode / ngram) | 不安装 | 同上早退 |
| import kernel wheel(`ascend_kernel`) | 不探测 | 标记行报 `kernel_wheel=not-probed`,不谎报 `unavailable` |
| 替换 attention / graph 条目(`builder_cls` / `impl_cls` / twin / runner) | 不替换 | 同上早退 |

- **可观测性不受影响**:关闭态仍打印
  `cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=not-probed)`——
  vLLM 会吞掉 `general_plugins` 的加载异常,这行是"插件确实跑过"的唯一判据。
- **启用态的 fail-open 不变**:内核 wheel 缺失/注册失败时仍是"整体禁用 + 单条
  warning"(见根 README「Kernel wheel dependency」),只是探测时机从"每次 load"
  收窄为"仅在启用时"。
- 测试:`tests/test_cascade_plugin.py::test_disabled_discovery_touches_nothing`
  (把 env 注入/shim/探测全部替换为 `pytest.fail`,关闭态调用 `load()`)、
  `::test_disabled_discovery_reports_not_probed_wheel`、
  `tests/test_cascade_fail_open.py::test_disabled_load_does_not_probe_the_wheel`。

## 7. 未做(登记,勿误读为已具备)

- `ruff format --check .` 当前**不通过**(2026-09-26 实测:25 个文件需重排,含
  `cascade_plugin.py` / `cascade_runner_patch.py` / `tests/test_cascade_*.py`;
  同一批文件在 `9951e41` 上同样不通过 ⇒ 存量问题,非本轮引入)。
  本仓 CI 模板 `.github/extension-ci.yml` 含这一步 ⇒ 原样启用 CI 会红;
  启用前需先做一次纯格式化提交。
