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
- **代价(明示)**:三个 fail-open shim 现在只在启用路径安装 ⇒ 关闭态下,若宿主的
  `policy_factory` / `spec_decode` import 本身是坏的,不再有插件侧静默兜底。
  这是"关闭即不动宿主"的必然代价(宿主自身缺陷不应由关闭态的插件掩盖),已由
  `test_disabled_discovery_touches_nothing` 固定。
- 测试:`tests/test_cascade_plugin.py::test_disabled_discovery_touches_nothing`
  (把 env 注入/shim/探测全部替换为 `pytest.fail`,关闭态调用 `load()`)、
  `::test_disabled_discovery_reports_not_probed_wheel`、
  `tests/test_cascade_fail_open.py::test_disabled_load_does_not_probe_the_wheel`。

## 7. CI(包级检查)与两套运行面

**工作流**:`.github/workflows/ci.yml`(2026-09-26 由 `.github/extension-ci.yml` 迁入——
GitHub 只执行 `workflows/` 下的文件,原模板位置从未生效,API 侧
`actions/workflows total_count = 0` 可证)。矩阵 Python 3.10/3.12/3.14,步骤:
`ruff check .` → `ruff format --check .` → `pytest -q -m "not host_tree"` →
`python -m build` → `extension inspect`。

CI runner 是**干净 ubuntu**(只装 `.[test]`),没有 torch / torch_npu / vllm-ascend,
也没有宿主源码树。两套运行面的分工:

| 测试类别 | CI(无设备/无宿主树) | 本机 NPU 宿主 |
|---|---|---|
| 纯逻辑(manifest / planner / op-selector / fi_sampling 路由 / fia-demask / rope-fix / zerocost) | 运行 | 运行 |
| 需要设备栈(cascade / mlp-chunk / gelu / host-signature …) | **skip**(可见) | 运行 |
| 源码级漂移守卫(rope-fix drift / aot-cache-guard drift) | **deselect**(`host_tree` 标记) | 运行(**缺宿主树即 fail**) |

- 设备类测试在模块顶部 `import _device_stack`(或 `from _device_stack import torch…`):
  该模块用 `pytest.importorskip` 探测 torch / torch_npu / vllm_ascend,缺任一即
  **整模块 skip**,skip 理由写明缺什么。**为什么不能只靠 `importorskip(carrier)`**:
  pytest ≥ 8.2 对"模块可导入、但其内部抛 ImportError"是**重抛**(避免掩盖真实缺陷),
  于是缺 torch 会变成收集期 ERROR 而不是 skip;必须先把依赖探测掉。
- `host_tree` 标记(`pyproject.toml [tool.pytest.ini_options] markers`)只用于
  **显式排除**,不带自动 skip:`pytest -q` 全量跑时这些守卫仍然"缺宿主树即红"
  (`test_rope_fix_drift.py` 的模块级 fixture、`test_aot_cache_guard_drift.py` 的
  源文件断言)。CI 里的 `-m "not host_tree"` 是有意的、写在 workflow 里的排除,
  而不是静默跳过。
- 复现 CI 判定(本机,不装/不卸任何东西):把 `torch`/`torch_npu`/`vllm`/`vllm_ascend`
  用 meta-path 拦截器屏蔽后重跑
  `pytest -q -m "not host_tree"`(2026-09-26 实测:97 passed / 11 skipped /
  14 deselected / 0 failed)。

## 8. 未做(登记,勿误读为已具备)

- ~~`ruff format --check .` 不通过~~ **已修**:`59e8dfa` 做了纯格式化提交,
  格式与两套运行面自那以后全绿。
- `docs/evidence/cascade/` 与 `src/vllm_ascend_split_batch/fi_sampling/` 是**存档/
  vendor 件**,ruff 通过 `extend-exclude` 排除(重排会破坏 `MANIFEST.sha256`)。
- 发布渠道未启用:证据齐后走 `uv publish` 到 pypi 或内部索引(见 §1)。

## 9. 停止与卸载(可执行命令)

**能力维度**(只关某一项,不动其它):

```bash
# 关 cascade(含图孪生):不注入这两项即等价于关闭
unset VLLM_ASCEND_ENABLE_CASCADE_DECODE VLLM_ASCEND_ENABLE_CASCADE_GRAPH
# 或走管理器(它会重写本地 enable 态,不注入该 bundle 的 environment)
vllm-hust-ext extension disable org.vllm-hust.split-batch-full-graph
vllm-hust-ext extension list          # 期望该 bundle 显示 disabled
```

- **判据**:重启服务后日志应只有
  `cascade plugin loaded (gate=0, graph_gate=0, kernel_wheel=not-probed)`,
  且无 `capture body` 行。默认关闭态**不注入任何 env**、不 import 宿主模块(见 §6)。

**服务维度**(停掉正在跑的服务并等显存释放):

```bash
pkill -f '[V]LLM::'                  # 方括号防自匹配;不要用 pkill -f "VLLM::"
sleep 10                             # 等 HBM 回落到基线,否则下一腿报 Free memory ... is less than desired
npu-smi info -t usages -i <card> | grep 'HBM Usage Rate'   # 期望回到 5% 量级
```

- 若服务由 `vllm-hust-ext run -- vllm serve ...` 启动,杀进程前先记下它注入的 env
  (管理器 enable 态仍会保存,下次 `run` 会重新注入)。

**插件维度**(从环境里移除):

```bash
pip uninstall -y vllm-ascend-split-batch          # entry point 随之消失
pip uninstall -y ascend-kernel                    # 可选:kernel wheel 是软依赖
python -c "import importlib.metadata as m; print(m.version('vllm-ascend-split-batch'))"  # 期望 ModuleNotFoundError
vllm-hust-ext extension list                      # 该 bundle 应不再出现
```

- `uninstall()` 语义在 carrier 级已有测试覆盖(`fia_demask` / `rope_fix` 的
  `test_uninstall_*`:注销注册表项、还原被替换的方法),但**服务级**的
  "卸载后启动服务仍正常"尚未作为验收项跑过 → 见 §8 类未做项。
- 回退到旧版本:重装对应 wheel 即可;本仓不做数据库/权重迁移,卸载无残留状态
  (唯一状态是 `~/.config/vllm-hust-ext/config.json` 里的 enable 位)。

## 10. 安装/配置/启动(最小可用路径)
```bash
python -m pip install -e ".[test]"                                  # 开发安装
pip install ".[kernels]" --find-links /path/to/ascend-kernel/output  # 可选:kernel wheel(cascade 必需)
vllm-hust-ext extension inspect org.vllm-hust.split-batch-full-graph # 期望 activation_ready=true
vllm-hust-ext extension enable  org.vllm-hust.split-batch-full-graph # 写入 enable 态
vllm-hust-ext run -- vllm serve <model> --max-model-len 4096 --port 8000 \
  --compilation-config '{"cudagraph_capture_sizes":[32,64,128]}'
```

- cascade 生效还需:共享前缀 ≥ 8192、并发 ≥ 32、无投机解码(见
  [support-matrix.md](support-matrix.md) §2);否则服务正常但 cascade 不接管。
- 停止/卸载见 §10。

## 11. 发布到 PyPI 的就绪度(2026-09-26 调研)

**路径是什么**:公共 **pypi.org**(TestPyPI 为强制前置门禁)。依据组织文档
`vLLM-HUST/vllm-hust-docs` 的 `operations/extension-author-guide.md` §16(先
`twine upload --repository testpypi`,在 TestPyPI 上跑完
discovery/enable/dry-run/disable/forget/uninstall 门禁后才 `twine upload` 正式 PyPI)与
`operations/bidkv-packaging-and-release-guide.md` §9/§10(`uv publish --check-url
https://pypi.org/simple`;用户按项目名 `uv pip install bidkv` 安装)。
本机 `pip.conf` 里的华为云地址是**下载镜像**(读),**不能**作为上传目标。

**两个包的准备度不同**:

| 项 | 插件包 `vllm-ascend-split-batch` | 算子包 `ascend-kernel` |
|---|---|---|
| 项目名占用 | 未占用(PyPI/TestPyPI 均 404) | 未占用(404) |
| 产物 | `py3-none-any`、125 KB、含 4 个 manifest + 12 个 entry point | `cp312-cp312-linux_aarch64`、211–285 KB、含 2 个自研 `.so` |
| 可转发性 | 纯 Python,无顾虑 | 轮内只有自研 so + 纯 py;**CANN/torch 一律运行时链接、未打包**(kernel README 链接纪律) |
| 验证面 | CI 三腿全绿(ruff/format + pytest + build + `extension inspect`) | 有 S1/精度/图冒烟套件与四元组纪律 |
| **阻塞** | ① §16 的 TestPyPI 往返门禁未跑;② 无 PyPI token | ① **仓在个人账号** `Raing5Days/vllm-hust-cascade-kernel`,不在 org;② **仓内无任何 LICENSE**(GitHub `license=None`,find 全盘无 LICENSE/NOTICE),而轮内 METADATA 却写 `License: BSD 3 License` —— 二者矛盾,公开分发缺许可依据;③ 平台轮子强绑定 CANN 9.1.0 + torch_npu 2.13.0rc1 + cp312 + aarch64 + soc 910B2(`CATLASS_ARCH=2201`);④ 版本号与内容错位(装机 lib = `2026.9.16` 构建的 `ca8de2d7…`,而 `pip show` 仍报 `2026.3.9`),PyPI **不可覆盖**同版本文件 ⇒ 发布前必须按 md5 定版 |

**发布前的最小检查单**(两包通用,补 §5):

- [ ] 版本号三处一致(`pyproject.toml` / manifest `extension_version` / changelog),且**不复用**已发布的版本号
- [ ] `twine check dist/*` 通过;wheel 内容审计(算子包按链接纪律只允许自研 so + 纯 py)
- [ ] 算子包:先补 LICENSE 并修 METADATA 的 license 字段;确认仓归属(个人 → org)与兼容域声明
- [ ] 插件包:TestPyPI 上跑完整生命周期门禁(§16),再发正式 PyPI
- [ ] 仓库/命令历史/CI YAML 中无 token;README 写明兼容范围、实验状态、启停与回滚
- [ ] 发布 commit 已 tag,工作树干净

**它为什么能解掉"管理器仍拒绝启用"**:管理器的判定只取决于**已安装包里的那份 manifest**
(facts.txt §J 的对照实验);公开发布后,"重装到当前状态"对任何人都是一条 `pip install`,
不需要我们手工给轮子。

**现状**:两包均**未发布**;本仓默认不发布(对外且不可逆),需要时按上面的检查单执行。

### 11.1 分发渠道的推荐形态(2026-09-26)

**两个发行物分开**(不把设备算子并进纯 Python 包,理由见 §12.1),但"算子库单独放"要落到具体渠道。
三个可选项:

| 选项 | 插件包 | 算子包 | 评价 |
|---|---|---|---|
| **A(推荐)** | **PyPI** | **org 仓的 GitHub Release 附件** + README 写稳定的 `--find-links` URL | 避开把窄平台轮永久冻结进公共索引;`pip install "插件[kernels]" --find-links <url>` 一条命令装齐 |
| B | PyPI | PyPI(另发一个平台轮) | 用户侧最省事(`.[kernels]` 直接可用),但要接受:四元组每变一次就得发新版本、名字被公共索引长期占用 |
| C | git/源码装 | 同 A | 现状;对第三方不友好(需自己构建或找轮子) |

**无论选哪条,两件事必须先做**(否则任何形式的分发都不成立):

1. **算子仓补 LICENSE**:`third_party/catlass` 是 **CANN OSL 2.0**(实测头文件声明),
   而 `csrc/**` 有 **40 处** include 该树 ⇒ `.so` 是 CANN OSL 2.0 衍生件:
   - §2.1 允许为**华为 AI 处理器**系统分发(我们正是该场景);
   - §3.3 要求分发时**随附协议副本**并保留 notices ⇒ 仓与轮内都要有该协议文本,catlass 副本本身漏了 LICENSE;
   - 轮内 METADATA 现写 `License: BSD 3 License`,与实际不符 ⇒ 必须改。
2. **对齐 `kernels` extra 的钉子**:现钉 `ascend-kernel==2026.3.9`,而该版本**任何索引上都查不到**,
   且装机内容已不是它(lib 来自 2026.9.16 构建,`pip show` 仍报 2026.3.9)⇒ 钉到真实存在的版本,
   并让"发布版本号 ↔ 内容"一一对应(README 的 md5 表就是现成的判据)。

**为什么插件包放 PyPI 就能解掉维护者的"仍拒绝启用"**:管理器读**已安装发行版里的 manifest**
(§11 末段),插件上了索引后任何人 `pip install` 都是当前状态,不需要我们再手工给轮子。

## 12. 两道决策题的事实底稿(2026-09-26 实测)

### 12.1 要不要把算子库并进插件包

**体积不是理由**:插件轮 `122 KB(py3-none-any)` + 算子轮 `278 KB(cp312-cp312-linux_aarch64)`,
把算子载荷塞进插件轮后实测 **≈398 KB**。真正的代价在别处:

| 代价 | 后果 |
|---|---|
| 发行物 tag 变成"平台+ABI" | x86_64 与其它 Python 版本 `pip` 无匹配发行物 ⇒ **装不上**;今天的 CI(ubuntu x64 × 3.10/3.12/3.14)既建不了也装不上 |
| 构建环境 | 算子侧要 CANN toolkit + bisheng/ccec + catlass(约 2 min);GitHub 托管 runner 建不出来 |
| 设计语义 | README 与 §6 的"软依赖 + fail-open(缺算子 ⇒ 整体禁用 + 单条 warning)"**作废** |
| 许可混合 | 插件 Apache-2.0 vs 算子轮 METADATA 声称 BSD-3 却**仓内无 LICENSE 文件** ⇒ 先解决 |
| 仓侧体积 | 算子仓 tracked = 67.6 MB / 317 文件,其中 `catlass-example-data/{k,v}.bin` 占 **64 MB**(S1 锚点数据) |
| 既存硬伤(与是否合并无关) | `kernels` extra 钉 `ascend-kernel==2026.3.9`,而该包**任何索引上都没有** ⇒ 第三方 `pip install ".[kernels]"` 必然失败 |

**建议**:两个 wheel 分开发布(插件保持 `py3-none-any`);若想少一个仓,可"同仓 monorepo、两个发行物",
但**不要把设备算子并进纯 Python 的那个包**。

### 12.2 fork 一个 CANN 官方算子仓把算子放进去,有没有用

实况(`gitcode.com/cann/*`,2026-09-26 实测):`ops-nn` 24731 文件、`ops-transformer` 14269 文件;
两仓 LICENSE = **CANN Open Software License Agreement 2.0**(非 Apache:2.1 只允许为**华为 AI 处理器**系统
使用/修改/分发;3.3 分发须随附协议副本并保留声明);有 `CONTRIBUTING.md` + SIG + `experimental/` 自定义算子入口
+ `.gitcode/workflows/`;两仓都有**官方 torch 扩展机制**(`cann_ops_nn`, `bash build.sh --torch_extension`
→ `build_out/*.whl`),但其形态是"JIT 编薄 C++ wrapper 桥接 **aclnn**",设备 kernel 由 CANN 算子库提供。

| 判断 | 内容 |
|---|---|
| 买得到 | 官方构建/测试/评审规范;自定义算子入口;**若被上游接受**,算子进 CANN 算子库 ⇒ 插件可丢掉自编 `.so`、走官方桥,同时解掉"算子轮无人发布/四元组轮子" |
| 买不到 | **fork 本身不发布任何东西**(个人副本,PyPI 仍需自己上传);且 **CANN 没有算子轮子的 pip 渠道**——华为云 ascend 索引实测只有 `nightly/ test/ torch-npu/ triton-ascend/ variant/`,公共 PyPI 无 `cann-ops*`(404),算子随 CANN toolkit 分发 ⇒ "放进 fork 就能被装到"不成立 |
| 隐含成本 | 不是搬家而是**重写**(op_host tiling/infershape + op_kernel + aclnn API + 其测试套件);我们当前是 catlass + `NpuExtension` 自编设备 kernel、注册 `torch.ops.npu.*`,**不在 aclnn 层**;另需长期同步 2.4 万文件量级的仓,并接受 CANN OSL 2.0 的处理器范围限制与随附协议要求 |
| 纪律 | 工作区 `AGENTS.md`:不做上游 PR、改动只在本地落地(文义限定 vllm/vllm-ascend)。CANN 方向的上游化需单独裁定 |

**建议**:只有当目标是"算子最终进 CANN 官方库、由官方维护"时才值得做,且应作为**上游贡献项目**
(重写 + 评审 + 许可)立项;若目标只是"让第三方能装到",它的收益为零 —— org 自建仓 + PyPI 轮子是等价且
便宜得多的路径。

### 11.2 传到 PyPI 组织:机制与步骤(2026-09-26 核实)

**先纠一个直觉**:PyPI 的"组织"**不改变上传目标**,也不给独立的索引地址。

| 问题 | 事实 | 出处 |
|---|---|---|
| 上传到哪 | 仍是 `pypi.org`(TestPyPI 是 `test.pypi.org`);`twine upload` / `uv publish` 的目标由"仓库 URL"决定,与组织无关 | PyPI Upload API |
| 组织给什么 | **所有权 + 权限管理**:Organization / Team / Member;组织角色 Owner·Manager·Member·Billing manager;项目角色 Maintainer(可上传发行物)·Owner(可管理项目与协作者) | `docs/user/organization-accounts/roles-entities.md` |
| 有没有命名空间 | **没有**。组织账户**不支持 namespace** ⇒ 项目名在全 PyPI 仍是**全局唯一**、无前缀保护 | `org-acc-faq.md` |
| 有没有私有包 | **没有**。组织账户不支持私有包,项目一律公开 | `org-acc-faq.md` |
| 怎么把项目放进组织 | 两条:(a) 组织成员在 **Your organizations → Manage → Projects → Create** 直接创建;(b) 先传到个人账户,再由 Owner **Transfer project** 转入。删除组织前必须先转走全部项目 | `actions/project-actions.md` |
| 有没有 CLI 管组织 | **没有**(仅 Web UI:`https://pypi.org/manage/organizations/`);个人项目上传仍是 CLI | `org-acc-faq.md` |
| 要花时间/钱吗 | 商业组织按月订阅、社区项目免费;组织申请由 PyPI admin 人工审核,**无时限承诺**;重名冲突也由 admin 仲裁 | `org-acc-faq.md` |

**推荐做法:用 Trusted Publishing(OIDC),不要 token**(仓库已备好 `.github/workflows/publish.yml`):

1. **PyPI 侧**(只有你能做):登录 → 若还没有项目,用 **Your account → Publishing → 新建 pending publisher**;
   填 GitHub 仓库 `vLLM-HUST/vllm-ascend-split-batch-hust` + workflow 文件名 `publish.yml` +
   environment `testpypi`(先在 TestPyPI 建)/ `pypi`(正式)。
   ⚠️ **pending publisher 不预留名字** —— 别人先注册同名项目则它作废;确定名字后尽快完成首次发布。
2. **TestPyPI 先行**:`Actions → Publish → Run workflow → target: testpypi`;
   然后在干净环境从 TestPyPI 装该版本并跑完整生命周期门禁(discovery / enable / dry-run / disable / forget / uninstall)。
3. **正式 PyPI**:`target: pypi`。工作流已内置与 CI 相同的门槛(ruff / pytest / build / `twine check`)与
   **发布回执**(commit + `sha256sum dist/*`),所以在索引上出现的任何文件都可追溯到 commit。
4. **项目归属**:首次发布若落在个人账户,再用组织的 **Transfer existing project** 转进组织(需 Owner)。

**名字现在就要定**(发布后改名不可逆):`vllm-ascend-split-batch` 与 `vllm-hust-split-batch` 在
PyPI / TestPyPI 上**都未被占用**(2026-09-26 实测 404)。若要带 org 前缀,趁现在改 `pyproject.toml`
的 `name` 与 `tests/test_manifest.py` 里的项目名断言。

**若你仍想用 API token 而不是 OIDC**:上传用户名固定 `__token__`,token 只在当次 shell 导出、
不写进仓库/命令历史/CI YAML(组织文档 §16 原话)。两种认证方式可并存,OIDC 只是免去保管长期凭据。

### 11.3 表单怎么填(逐栏 + 两处易错点,2026-09-26 由 warehouse 源码确认)

**先判断你在哪一页**:环境名输入框里的**灰色占位文字就是站点指纹** ——
warehouse 模板的取值是 `placeholder="testpypi" if testPyPI else "pypi"`
(`templates/manage/account/publishing.html` 与 `.../organization/publishing.html` 同逻辑)。
占位显示 `pypi` ⇒ 你在 **pypi.org**;显示 `testpypi` ⇒ 你在 **test.pypi.org**。

| 栏位 | 填 | 依据 |
|---|---|---|
| PyPI Project Name | `vllm-ascend-split-batch` | `pyproject.toml` 的 `name`(**不是**仓名) |
| Owner | `vLLM-HUST` | GitHub 侧仓归属;与 PyPI 组织无关 |
| Repository name | `vllm-ascend-split-batch-hust` | 仓名带 `-hust` |
| Workflow name | `publish.yml` | 写**文件名**(该文件已在 `main`) |
| Environment name | **你在 pypi.org ⇒ `pypi`;在 test.pypi.org ⇒ `testpypi`** | 必须与工作流实际使用的环境逐字一致(本仓工作流按 `target` 取值) |

**易错点 1:组织页与账户页的字段文案不同,据此可判定归属。**
`warehouse/templates/manage/organization/publishing.html` 的 GitHub 表单:
- 标签是 **`PyPI Project Name`**(账户页是 `Project Name`);
- 帮助文字明确写 **"created and owned by the '<org>' organization when this publisher is used"**;
- 页内 Tip:**"Trusted publishers created here will be owned by this organization when the project is created."**

⇒ 在**组织**的 publishing 页填表,**首次发布时项目直接归属该组织,无需再 Transfer**。

**易错点 2:环境名与目标索引耦合。**
本仓工作流 `target: testpypi` 用环境 `testpypi`、`target: pypi` 用环境 `pypi`(OIDC 交换要与对应索引上的
publisher 配置逐字匹配)。因此:
- 只建了 PyPI 侧 publisher ⇒ 跑 `target: testpypi` 会在 TestPyPI 侧认证失败(那里没有对应 publisher);
- 要做 §16 的 TestPyPI 往返,需在 **test.pypi.org** 上另建一个(TestPyPI 是独立站点与独立账号,组织不一定在那边存在)——
  用账户级 `pending publisher`、环境名 `testpypi` 即可。

**建议的保护**:给仓库的 `pypi` 环境加 required reviewers(Settings → Environments)。环境会在首次引用时
自动创建,但**批准门**要手动加;正式发布不可覆盖,留一个批准门是廉价的保险。

### 11.4 点 "Add" 之后发生什么(2026-09-26 由 warehouse 源码定案)

**它不会"把仓库加进 PyPI",也不会在 GitHub 侧建立任何连接** —— 没有 webhook、没有 App 安装。
PyPI 只在 Add 时做**一次只读校验**,然后在你名下存一条 "pending publisher" 记录。

**Add 时的校验(逐条,`warehouse/oidc/forms/github.py`)**:

| 字段 | 校验 | 失败提示 |
|---|---|---|
| Owner | 先过正则,再调 GitHub API `GET /users/{owner}`(只读,取规范化的 `login` 与 `id`) | `Unknown GitHub user or organization.` / rate-limited / 连接或超时错误 |
| Workflow name | 必须以 `.yml`/`.yaml` 结尾,**且不能含 `/`** | `Workflow name must end with .yml or .yaml` / `Workflow filename must be a filename only` |
| Environment name | 可空;≤255 字符;首尾不能有空白;不能含 `"` `'` `,` `;` `\` 或不可打印字符;**存储时转小写**(大小写不敏感) | 对应三条提示 |
| 项目名 | 只校验**合法名格式**(`PROJECT_NAME_PATTERN`) | — |

- **仓库名不在校验范围**:源码注释写明 "We can't do this for the repository, since it might be private" ⇒ 仓名/工作流是否真存在,要到**首次发布**才被验证。
- **项目名只在首次发布时才被占用** ⇒ Add **不预留名字**。
- **Add 页面要求重新认证**:组织视图 `permission=Permissions.OrganizationsManage` + `require_reauth=True` ⇒ 提交前 PyPI 会让你重输密码(2FA)。
- 重复提交同一条(同仓 + 同 owner + 同 workflow + 同 environment)会被拒:`This publisher has already been registered in your organization.`

**Add 之后的状态**:组织页 `Pending publishers` 列表里出现一条,并 flash
`Registered a new pending publisher to create the project 'X' owned by the '<org>' organization.`
(同时写组织审计事件 `PendingOIDCPublisherAdded`)。**此时 PyPI 上还没有项目。**

**⏳ pending publisher 有 30 天 TTL**(`warehouse/oidc/tasks.py`):
`PENDING_PUBLISHER_EXPIRY_DAYS = 30`,到期前 5 天(`PENDING_PUBLISHER_REMINDER_DAYS = 5`)发提醒邮件,
到期后由定时任务**删除记录**并通知。⇒ **别加了不发布**。

**首次发布时怎么匹配并"转正"(`warehouse/oidc/models/github.py` 的 `PendingGitHubPublisher.reify`)**:

| OIDC claim | 要求 |
|---|---|
| `repository` | 与 `owner/repo` **大小写不敏感相等** |
| `job_workflow_ref` | 必须等于 `OWNER/REPO/.github/workflows/<文件名>@<ref 或 sha>` ⇒ **工作流文件必须在 `.github/workflows/` 顶层**(不能放子目录),且文件名逐字一致 |
| `environment` | 若 publisher 填了环境名,则 token **必须**带同名环境(大小写不敏感);留空则不校验 |
| `event_name` | **除 `pull_request_target` 外全部允许** ⇒ `workflow_dispatch` 可用 |

匹配成功后:`reify()` **找到或新建**一条普通 `GitHubPublisher`(同仓+owner+workflow+environment),
**删除 pending 记录**,项目由该组织创建并归属该组织。此后就是普通 publisher,**不需要再注册**;
要改配置则在项目页的 Publishing 里增删。
