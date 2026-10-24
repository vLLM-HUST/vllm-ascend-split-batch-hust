# 发布流程与检查表

## Cascade 接入边界

- 不发布 `cascade-attention` general plugin，不注入 cascade 环境变量。
- `org.vllm-hust.split-batch-full-graph` 仅保留 `import_only` planner，管理器不得激活。
- 不声明 `[kernels]` extra，不依赖或自动安装外部 cascade kernel wheel。
- cascade 实现模块、Runner 包装、图孪生、gate 子进程和专属测试已删除；历史证据与只读溯源不属于可执行插件源码。
- 其余插件的入口、门控、host 版本范围、宿主只读及禁止 platform plugin 的约束不变。
- 升级须重新安装本轮构建产物并重启 worker/重新捕获图；不支持卸载入口后热撤销既有包装。
- 回滚本轮仅恢复源码；上轮已停用的注册入口和外部依赖不自动恢复。安装回滚产物后须重启并重新捕获图，重新启用另行验证。


基于 [bidkv 打包发布指南](https://github.com/vLLM-HUST/vllm-hust-docs/blob/main/operations/bidkv-packaging-and-release-guide.md)
裁剪,补充本插件特有事项。Bundle ID: `org.vllm-hust.split-batch-full-graph`。

## 算子 bundle 最短交付路径

- **注册契约优先**：先核对实际版本的 IR/CustomOp dispatch 与图替换 API，
  用真实调用点的图探针确认目标子图可见；API 存在不等于调用链可达。
  优先注册实现或 fusion rule，未匹配时保留原生路径；源码字节校验只用于
  尚未迁移的 tier-2 补丁，不作为正式注册实现的接口契约。
- **先接入、后验收**：开发阶段即提供独立 `import_only` bundle、
  `general_plugins` 注册入口、env 门控、接口能力检查及 fallback 测试。
  L0 / L3+compare / light L4 使用待交付的同一 kernel、callsite 与配置；
  不在验收后另写一份融合或迁移到另一条调用链。
- **一个实现、两个发布对象**：Python/Triton kernel 随插件 wheel 发布；
  若宿主没有可替换的融合边界，接缝由仓内可逆 host proposal 单独交付。
  bundle 只核验接缝，不在加载时改写宿主文件、替换整个 forward，
  不全局跳过 norm，也不把依赖宿主补丁的方案称为纯 wheel 即插即用。
- **验收材料复用**：证据绑定形状/权重、实现及宿主哈希、配置、插件 commit；
  相同执行路径复用已有 L0/L3/L4，不因补 descriptor 重跑性能窗口。
  接入或计算语义变化则按明确缺口补 L3+compare、graph replay，必要时补 L4。
  发布生命周期检查与算子验收分开；端到端声明仍遵守相应发布判据。
- **发布收尾**：最终 wheel 隔离安装 → 发现/关闭态检查 → 候选启用的 env
  预览及真实起服 → 双 rank 生效和正确性 → disable/restart 回归原生路径 →
  审核三证据与完整性 → 提交 `active`。最终字节与被测字节的差异须审查；
  `active` 只允许显式 enable，不改变 default-off。推送/发布仍另需授权。
- **回滚单位是进程与图**：关闭 env 后重启并重新 capture；不承诺已捕获图热切换。
  保留独立的 `patch -R` 校验，不将卸载插件等同于撤销宿主补丁。

### GDN 注册接入验证门

`tools/probe_gdn_registration.py --output <新JSON路径>` 使用实际 serving Python，
先导入 `torch_npu` 后核对 npugraph_ex API，并对真实注册的
`qwen_gdn_attention_core` 做 CPU export；不加载模型、不执行 NPU kernel。
该探针不是生产编译图，也不证明 pattern 已命中。

- 外层图仅含 opaque core 时，禁止将内部 unpack/norm 的 pattern 注册记为生效。
  必须另证内部计算可进入编译图，或经已有细粒度 dispatch 点接入。
- `PluggableLayer.register_oot` 是层级注册，不是 unpack/norm 注册点；
  必须核对实际实例类、Ascend 已有注册冲突和 worker patch 时序。
  不通过覆盖全层 forward、分解含 KV/通信副作用的 opaque core 或全局 norm bypass 绕行。
- `torch.ops._C_ascend` 只解决 kernel 调用；不能代替调用点可达性证明。
- 接入成立后补匹配/不匹配、跨流和执行 scope、形状/相位守卫测试，
  再走新接入的 L3+compare 与 graph replay；不继承旧接入的数值 PASS。

### GDN QKV norm tier-2 交付边界

没有已证实可达的细粒度 dispatch 时，停止 bundle 注册，不保留只做源码哈希
检查却不注册融合的入口。GDN 的 kernel、可逆 host proposal 与已有算子证据
仍保留；不开新性能窗口，不改宿主，不将历史 kernel 验收转写为注册接入 PASS。
结论与重开条件见 `host-proposals/GDN-qkv-norm/registration-closure.json`。

tier-2 开关仍为 `VLLM_HUST_GDN_QKV_NORM=1` 且 `USE_DEFAULT_FLA_NORM=0`，
依赖人工审查的宿主补丁；默认关闭。关闭后重启并重新捕获图，宿主补丁单独回滚。
只有宿主版本提供可达窄边界或得到新机制证据，才重新立项 bundle 接入；
届时重新验证调用链、L3+compare、graph replay 和管理器生命周期。

## 0. 状态纪律(先于一切)

- manifest 保持 `import_only`,直到三项证据齐备:default-off 零回归冒烟、
  与参考/基线的正确性对齐、TPOT/latency 性能对比(见工作区 AGENTS.md,
  工作流见 [agent-guide.md](agent-guide.md))。
- 翻 `active` 前,`activation.environment` 必须是可注入的真实值
  (`"1"`,由 `tests/test_manifest.py` 保证格式);default-off 语义由
  "不 enable 就不注入 + `load()` 内部门控" 双层保证。
- `host.version_range` 声明**有界的已验证域**(2026-09-27 起
  `>=0.25.1rc2.dev125,<0.25.2`;见 §0.1),禁止 `>=0`(也无上界=等同 `>=0`)。
  历史教训:packaging 语义下 prerelease 不属于 `>=X.Y.Z,<next` 形式的下界
  (如 `0.23.0rc1 ∉ >=0.23.0`),且**有序比较符内不得出现 `+local` 标签**
  (packaging 直接 `InvalidSpecifier`;已由 `tests/test_manifest.py` 守住),
  区间写法必须对照实装版本核 packaging 判定。区间收窄/放宽前先过 §4 清单并
  留档验证环境(见 §0.1)。
- **区间的语义边界(2026-09-27 裁定)**:区间是"**声明兼容窗口**",不是"逐
  revision 验证"。两个端点都已真机验证(dev125、dev605);端点之间/之后的 build
  按区间**声明**兼容 ⇒ 若在那里出问题,**归因给宿主**(这正是放宽的目的:把
  失败变成宿主侧发现,而不是让插件替宿主扛下来)。裁定与理由见 §0.3。
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
| `host.version_range` | **`>=0.25.1rc2.dev125,<0.25.2`**(有界区间,2026-09-27 起;此前为点钉 `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`) |
| 区间内**已验证端点 1** | vllm-ascend `0.25.1rc2.dev125+hust.20260903.4` @ `74f0c0a272376412b51e1c1864803d5f3a0f1b5f`(main) + vllm `0.28.1.post1.dev143+gf18cf803c.empty` @ `f18cf803c5f63625e2c71253ddaf8b0bad0bad1a`(vllm-hust release v1) |
| 区间内**已验证端点 2** | vllm-ascend `0.25.1rc2.dev605+hust.20260903.4` @ `fbe4911bb`(本地自编;**必须配 vllm core `0aee727ff6`**,见 §0.3 的配对警告) + 正确性/顺序/性能证据 `handoffs/receipts/20260926-rebuild-verify/VERIFY-PROGRAM-20260927.md` |
| 区间**之外** | `0.24.*` / `0.26.*` 判 `incompatible`(有界性由 `tests/test_manifest.py` 机械守住) |
| 验证环境 | Python 3.12.14 / torch 2.13.0+cpu / torch_npu 2.13.0rc1 / CANN 9.1.0 (`/usr/local/ascend91`) / 910B2 |
| 验证证据 | `extension check` → compatible;§3 启用验证(`knowledge/evidence/cascade/section4-active-enablement.md`);正确性 `section2-correctness.md`(真模型 6/64);历史域(0.23.0rc1 / CANN 9.0.1,已退役)留档于 `EVIDENCE.md` 与 c3-legacy 产物 |

### 0.2 点钉到底是什么,以及它拦住的是哪条路(2026-09-26；**2026-09-27 起已被 §0.3 的有界区间取代**)

> 历史留档:本节叙述**点钉时期**的行为与解法,保留用于解释"当时为什么只认一个 build"以及
> 点钉字符串怎么解开。现行声明是**有界区间**(§0.1/§0.3);下面表格里的"钉值"按历史读。

**点钉的字符串是可以解开的**:`==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`
= git tag `v0.25.1rc1+hust.20260903.4`(→ commit `4e57439e`,2026-09-03)
+ **距该 tag 125 个提交** + commit `74f0c0a27`(2026-09-08)。实测
`git describe --tags 74f0c0a27` = `v0.25.1rc1+hust.20260903.4-125-g74f0c0a27`。
⇒ 版本号由 **git 状态**决定,不是机器指纹:**任何人在公开仓里 fetch 到那个 tag、checkout
`74f0c0a27`、完整(非 shallow)构建,就会得到同一个版本串**。所以它是
**锁 revision,不是锁机器**;但现实是组织仓 main 已前进到 `fbe4911bb5`(2026-09-25),
**默认装出来的都不在这个钉子里**。

**它只拦 `vllm-hust-ext run`,不拦安装与 enable**:

| 路径 | 版本判定 | 依据 |
|---|---|---|
| `extension enable <id>` | **不看**版本 | `extension-manager/src/vllm_hust_ext/cli.py:150` 只查 `activation_blocker` |
| `vllm-hust-ext run -- vllm serve …` | **看**;不匹配即拒启 | `cli.py:264`:`INCOMPATIBLE` → `refusing to launch incompatible extension` |
| 自己给 `vllm serve` 传 env(README 的 env 表) | **不经过管理器**,因此无此判定(属未验证域) | `vllm.general_plugins` 入口与 manifest 无关 |

**管理器有一条"操作者自述"通道**(实测):`configure --file` 里的 `host_version`
会**替代**自动探测(`providers/base.py:83`)⇒ 声明成钉子的值,`run` 就会放行。
**但这是自述,不是证据** —— 等于操作者自己认下"我在未验证宿主上跑"。本仓**不会**
把这种声明写进 manifest 当兼容性结论;要用由使用者自行决定并自负其责。

**要真正扩大受众,只有两条路**:

1. **在目标 build 上重核再放宽**(正路):按 §4 清单在装着目标 vllm-ascend 的机器上逐项验证,
   然后改 `host.version_range`、发新版本(`AGENTS.md` 禁止写 `>=0`)。该核验属"环境核验",
   按工作区分工归测试机(本机只有 `74f0c0a27` 一个宿主)。
2. **让消费方固定到已验证的宿主 revision**:即 checkout `74f0c0a27` 自行构建 ——
   可复现但代价明显(旧提交 + 完整历史 + CANN 配齐),只适合"就要这一份"的场景。

> **现状(2026-09-27)**:上表第 1 条已落地 —— 区间 `>=0.25.1rc2.dev125,<0.25.2` 覆盖整条 0.25.1 线,
> 两个端点均已真机验证;窗口内失败的归因规则、实测放行效果与 core 配对警告见 §0.3。

### 0.3 为什么从点钉改为有界区间(2026-09-27 裁定)

**裁定**:`host.version_range` 从点钉改为 **`>=0.25.1rc2.dev125,<0.25.2`**。
理由(owner 原话的意思,逐条落账):

1. **点钉让插件在除一个 build 之外的地方都无法验证**。`vllm-hust-ext run` 只在点钉上放行,
   于是 480 个后续提交、以及任何其它装机形态都到不了"能不能跑"这一步 —— 连"哪里会出问题"
   都测不出来,这种正确性对使用者没有价值。
2. **失败的归因应当落在宿主侧**。区间是"声明兼容窗口":端点已验证,窗口内的 build 声明兼容;
   窗口内出问题 ⇒ 那是宿主 build 的发现(我们据此登记/修),而不是让插件替宿主承担"不可用"。
3. **纪律不变**:仍然禁止 `>=0`(以及任何无上界写法)。区间必须**有界**,由
   `tests/test_manifest.py::test_host_version_range_is_a_bounded_verified_interval` 机械守住
   (两个已验证端点必须在区间内;`0.24.*`/`0.26.*` 必须在区间外);
   `test_no_local_version_label_in_ordered_comparators` 守住"有序比较符内不得有 `+local`"
   (packaging 会直接 `InvalidSpecifier`)。

**已实测的放行效果**(同一份 manifest,两种宿主):

| 宿主 | 放行前 | 放行后 |
|---|---|---|
| `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`(旧点钉) | compatible | **compatible** |
| `0.25.1rc2.dev605+hust.20260903.4.gfbe4911bb`(目标栈,自编) | **incompatible**(`check` 拒、`run` 拒启) | **compatible**;`run --dry-run` exit 0 且正确注入 `activation.environment` |
| `0.26.0` / `0.24.0` | incompatible | **incompatible**(有界性未破) |

⚠️ **配对警告(区间表达不了的维度)**:`host.version_range` 只覆盖 **vllm-ascend** 的版本,
**不覆盖 vllm core**(两个 fork 各自独立演进)。2026-09-27 实测:asc `fbe4911bb`
**必须**配 core `0aee727ff6`;配 core 最新 `fb1fd64f93` 会
`ImportError: cannot import name 'RoutedExpertsLists' from 'vllm.v1.outputs'`
(上游 #45635 删了该符号,而 asc 仍 import 它),配 release v1 `f18cf803c` 会
`ModuleNotFoundError: vllm.models.deepseek_v41`。这类错**在 import 期就炸、信息明确**,
不会被误读成 cascade 的行为问题;选配对时以上面两个已验证端点为准。

**使用者在窗口外怎么办**(两条,均为操作者自担):① 换到窗口内的 build;② 用管理器的
自述通道 `configure --file` 声明 `host_version`(等于操作者认下"我在未验证宿主上跑",
见 §0.2)。

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

**先跑机械前半程**（2026-09-26 起）：`tools/seam_diff.py` 把下面这份清单的**静态部分**
（签名 + 调用点 + 结构面）变成可执行判定，而且是纯 AST、**不需要安装宿主**，
因此可以在只有一份已安装宿主的环境里对拍两棵树：

```bash
git worktree add --detach /tmp/new-ascend  <待升 vllm-ascend commit>
git worktree add --detach /tmp/new-core    <待升 vllm-hust commit>
python3 tools/seam_diff.py <已钉 vllm-ascend 树> /tmp/new-ascend --repo ascend
python3 tools/seam_diff.py <已钉 vllm-hust 树>   /tmp/new-core   --repo core
python3 tools/seam_diff.py --selftest      # 判别力自证（tools 没在空跑）
```

- 退出码 0 = 静态面一致；1 = 有签名/调用点/结构面差异，需逐条处理。
- 它**只证明静态面**：方法体语义（对象如何被构造、`BatchDescriptor` 如何被填充、
  aclgraph 变体表在运行时怎么组织）必须靠下面的真机冒烟。
- 首次对拍报告与原始输出：[归档](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/host-drift-2026-09-26.md)
  历史检查结果以原始归档为准，不代替待升级宿主的重新核对。

**再逐项核对（人眼 + 真机）**：

`host.version_range` 升级前,逐项核对 monkeypatch/公开面签名(细节见
HOST_CONTRACT.md),差异收敛在 `cascade_runner_patch.py`:

- [ ] `vllm.v1.cudagraph_dispatcher.add_cudagraph_key` / `dispatch`
- [ ] Runner: `_capture_cudagraphs` / `_warmup_and_capture` /
      `_determine_batch_execution_and_padding` / `_model_forward` /
      `_update_full_graph_params_if_needed`
- [ ] vllm-ascend: `update_full_graph_params` / `get_graph_params` / `GraphParams`
- [ ] `ACLGraphWrapper` variant-entry 表结构(标准 `BatchDescriptor` 键)
- [ ] **图重放机制代际**(2026-09-27 新增,两代语义不同,必须逐条核对):
      `vllm_ascend/compilation/updatable_graph.py` 是否存在(基线 `74f0c0a27` 无)、
      `vllm_ascend/utils.py:use_updatable_graph`、`acl_graph.py` 里
      `ACLGraphWrapper._updatable_graph_replay` 的**签名与调用点**、
      以及 `update_full_graph_params` 是否对可更新图提前 `return`。
      判据:新机制宿主上冒烟日志出现 `updatable replay: cascade update ran=True`
      且 `cascade key hit` = 0;旧宿主上出现 `updatable-replay seam absent on this host`
      且 `cascade key hit` > 0。机理与顺序约束见 [pitfalls.md](pitfalls.md) §2.4
- [ ] `vllm.general_plugins` 加载时机未变
- [ ] fi_gelu 注册面：`CustomOp.register_oot` 以 `GeluAndMul` 类名建键，
      `CustomOp.__new__` 实际消费此键；Ascend 注册列表不得新增同名 provider。
      原宿主类不变，provider 自身保留 `name="gelu_and_mul"`，验证 all/none
      与 `+gelu_and_mul` / `-gelu_and_mul`；关闭态无重型导入、冲突不覆盖、
      卸载仅删除自有注册。先跑 `tests/test_fi_gelu_plugin.py`，再跑
      `tools/probe_gelu_registration.py --core <实际 core 树> --ascend <实际 ascend 树>
      --output <新证据.json>`（设置 `VLLM_HUST_FI_GELU=1`）。后者是受控依赖下的
      宿主类源码执行，不是正常宿主 import 或起服验收；还须在 NPU 跑
      `tests/test_npu_w31_gelu_plugin.py`、L3 + compare 与 light L4。
      回滚：移除 enable 环境变量并重启 worker/重捕获图；不得承诺实例热切换。
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

### 4.1 0.28 镜像 host 实测通过(2026-10-11 切镜像日)

宿主 vllm `0.28.1.post1.dev143` + vllm_ascend `0.25.1rc2.dev125+hust…`(CANN 9.1.0 /
Py 3.12.13 / glibc 2.35 底座)。**门 3 处 JSON 零修改放行**(门读 `version("vllm-ascend")`,
落在有界窗口内)+ 依赖面 **22/22** + 全量测试 **461/466**(5 失败=cascade 内核 wheel 的
GLIBC 环境缺口,fail-open 兜底,非代码冲突)。两点警示见证据文件:
①版本串漂移——hust 构建的 vllm_ascend 版本串停在 0.25.1rc2.dev 窗口而代码已追平
0.28,**门的满足不能推断宿主代码代际**,升级核对仍须走本清单;②cascade 内核在 2.35
底座不可载,0.28 主栈暂不启用(用户裁定),重启用条件=22.04 底座重建 wheel。
完整证据:[evidence/host-028-2026-10-11.md](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/host-028-2026-10-11.md)。

### 4.2 W1R GMM 补丁核对（tier-2 提案，未发布）

`host-proposals/W1R-gmm-tuning-config/` 是**提案补丁**，不是插件能力；升级宿主时若
`vllm_ascend/ops/fused_moe/routed_experts.py` 的 sha256 变化，必须重跑：

```bash
python3 host-proposals/W1R-gmm-tuning-config/gen_patch.py --check \
  --host /vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py
```

- **`--check` 只对未打补丁的宿主/镜像 blob 有效**：运行树处于部署态时会得到 drift（2）
  的误导结果。此时二选一：①直接对镜像 blob 核；②先把运行树还原 —— 注意
  `w1r-gmm-tuning` 层**已从 MANIFEST 移除**，脚本里没有该名字可回滚：可对已核过的
  `patch.diff` 直接 `patch -R`（先 `--dry-run`），或按 `OWNERSHIP.md §5` 重新加层后
  再用 `apply-runtime-patches.sh --rollback`。**不要**对未登记的层名执行静默命令。
- `--check` 语义：0 = 磁盘 `patch.diff` 与现算一致且宿主仍是基线；2 = 宿主漂移（fail-closed，
  先重做 P0 静态判定）；3 = 锚点缺失/重复；4 = 磁盘 patch 已过期（去掉 `--check` 重新生成）。
- 重新生成后必须同步 `proposal_meta.json` 的 `hashes`（patch / patched），
  并核对 `patch -R` 可回滚。
- 该补丁的两个开关都 default-off；未设时**无设备差异、零 shape 读取、零 IO、零 import**，
  但**每次 GMM 仍有两次早退 Python 调用**（tuning kwargs 查询 + dump 早退）——
  **不得**表述为「宿主行为完全一致」或「字节等价」（宿主文件文本必然不同）。
  CPU 门见 `tests/test_w1r_gmm_tuning_patch.py`。

## 5. 检查表

**构建前**

- [ ] `pytest -q` 全绿、`ruff check .` 通过
- [ ] **`host_tree` 类守卫在本机跑过**(发布检查的固定步骤,零成本 —— 见下方"为什么单列")
- [ ] 版本两处一致(测试守护)
- [ ] 发布 commit 已记录、工作树干净

**为什么 `host_tree` 单列**:CI 跑的是 `pytest -q -m "not host_tree"`(依赖无关的 runner 上
没有 `vllm-ascend` 源码树),所以两个**源码级漂移守卫**
(`tests/test_rope_fix_drift.py` / `tests/test_aot_cache_guard_drift.py`)在 CI 里是
**显式 deselect**,不会替你发现宿主源码漂移。发布前必须在本机(有宿主树的环境)跑一次**全量**:

```bash
cd /vllm-workspace/vllm-ascend-split-batch-hust
python -m pytest -q            # 不带 -m:含 host_tree;缺宿主树即红(这是设计意图)
python -m pytest -q -m "not host_tree"   # 复现 CI 判定(两套运行面的对照)
```

判据:全量里 `host_tree` 两例既不是 skip 也不是 deselect;若报"缺宿主树",说明本机不是
验证环境,换到有 `vllm-ascend` 源码树的机器跑完再发(见 §7 的两套运行面分工)。

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
- 复现 CI 判定(本机,不装/不卸任何东西):**两种办法,第二种更忠实,建议优先** ——
  1. meta-path 拦截器:把 `torch`/`torch_npu`/`vllm`/`vllm_ascend` 屏蔽后重跑
     `pytest -q -m "not host_tree"`。拦截器要能覆盖**子进程**,否则只遮住父进程:
     用 `sitecustomize.py`(Python 启动即 import,父子进程同效),不要用 `-p` 插件。
  2. **干净 venv(与 CI 逐步一致)**:`python -m venv /tmp/ci-venv` → 装管理器
     (本地 clone 或 git)→ `pip install -e ".[test]"` → `pytest -q -m "not host_tree"`。
     注意拦截器法**不等于**真 CI:宿主包"被屏蔽"与"压根没装"在管理器眼里是**不同 state**
     (见下),只有 venv 法能重现后者。
- **`degraded` 是第三种宿主形态(2026-09-26 实测,由 CI 红发现)**:管理器对
  "**宿主包不存在**"给的是 `['installed','discovered','configured','degraded']`
  (evidence 逐字 "host version is unavailable; compatibility is unverified"),
  **不是** `incompatible`;`configured` 仍在。三种形态对照:

  | 宿主 | states |
  |---|---|
  | 不存在(CI runner) | `installed, discovered, configured, degraded` |
  | 存在且命中点钉(`hust`) | `installed, discovered, compatible, configured` |
  | 存在但不命中点钉(隔离重编的目标 revision) | `installed, discovered, incompatible` |

  教训:`tests/test_extension_config_boundary.py` 首版只写了后两支 ⇒ CI 三腿全红
  (run `36255078500`,main@`084f299`),补 `degraded` 支后 run `36255434671`(main@`57238b1`)三腿全绿。
- 复现 CI 判定(2026-09-26 早前口径,同上第 1 法):**2026-09-26 实测 97 passed / 11 skipped /
  14 deselected / 0 failed;2026-09-26 续做后复测 112 passed / 11 skipped / 14 deselected /
  0 failed**(差值来自新增的 6 例)。
- **发布检查单侧已固化**(2026-09-26):`host_tree` 守卫是 CI **唯一不覆盖**的一类,
  故 §5「构建前」把它列为固定步骤(本机跑全量 `pytest -q`),并写明判据与两套运行面的差异。
  自托管 NPU runner job 仍属基础设施决策,本轮不动(登记于 §8)。

## 8. 未做(登记,勿误读为已具备)

- ~~`ruff format --check .` 不通过~~ **已修**:`59e8dfa` 做了纯格式化提交,
  格式与两套运行面自那以后全绿。
- [docs/evidence/cascade/](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/67505eee3dc71fa3dbbc0fc365620f0a2e5658f4/archive/mod-git-20261022/split-current-git/snapshot/docs/evidence/cascade/) 与 `src/vllm_ascend_split_batch/fi_sampling/` 是**存档/
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
| **阻塞** | ~~① §16 的 TestPyPI 往返门禁未跑;② 无 PyPI token~~ **已解**(2026-09-26:走 OIDC Trusted Publishing,门禁以正向/反向 + 干净 venv 判据替代,见 §11.8) | ① **仓在个人账号** `Raing5Days/vllm-hust-cascade-kernel`,不在 org;② **仓内无任何 LICENSE**(GitHub `license=None`,find 全盘无 LICENSE/NOTICE),而轮内 METADATA 却写 `License: BSD 3 License` —— 二者矛盾,公开分发缺许可依据;③ 平台轮子强绑定 CANN 9.1.0 + torch_npu 2.13.0rc1 + cp312 + aarch64 + soc 910B2(`CATLASS_ARCH=2201`);④ 版本号与内容错位(装机 lib = `2026.9.16` 构建的 `ca8de2d7…`,而 `pip show` 仍报 `2026.3.9`),PyPI **不可覆盖**同版本文件 ⇒ 发布前必须按 md5 定版 |

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

**现状(2026-09-26 更新)**:插件包**已发布** 0.1.0 / 0.1.1 / **0.1.2**(回执见 §11.6–§11.8),
其"阻塞"两项均已解除(TestPyPI 往返门禁按 §11.7 的实测口径走
**正向/反向 + 安装判据**替代;token 改用 **Trusted Publishing/OIDC**,无需长期 token)。
算子包仍**未发布**(改走 GitHub Release 附件,§11.7),其四条阻塞里许可证已补
(算子仓 `ca5bd9d`),**仓归属仍在个人账号**(handoff 归属-1,未转仓)。

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
2. ~~**对齐 `kernels` extra 的钉子**~~ **已修(0.1.1)**:原钉 `ascend-kernel==2026.3.9`,
   该版本任何索引都查不到 ⇒ `pip install "vllm-ascend-split-batch[kernels]"` 必然失败。
   现钉 `ascend-kernel==2026.9.26`,该轮作为 **GitHub Release 附件**发布(见 §11.7),
   解析已实测通过。算子仓侧同时修了 METADATA 的 license 字段(原误写 BSD-3)并随包嵌入
   CANN OSL 2.0 协议文本。

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
| ~~既存硬伤~~ **已修(0.1.1)** | `kernels` extra 曾钉 `ascend-kernel==2026.3.9`(任何索引都没有)⇒ 第三方 `pip install ".[kernels]"` 必然失败;现钉 `2026.9.26` 并由 Release 附件提供(§11.7) |

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

### 11.5 首发时机与首发前的最后决定(2026-09-26)

**"首发时机"= 你手动点 `Actions → Publish → Run workflow`(target=`pypi`)的那一刻。** 为什么它是一个需要
主动选的时间点,而不是"随时都行":

| 在首发那一刻发生 | 之前 | 之后 |
|---|---|---|
| 项目名归属 | pending publisher **不占名**(§11.4) | 项目由组织创建,名字归组织 |
| 版本号 | 可任意改 | **永不可覆盖**(PyPI 不允许同版本重传) |
| pending 记录 | 保留 | `reify()` 转正为普通 publisher 并删除该记录 |

⏳ **计时已开始**:pending publisher 自 Add 起 **30 天**过期(`PENDING_PUBLISHER_EXPIRY_DAYS = 30`,
提前 5 天提醒)。⇒ 首发应在 30 天内完成,否则要重新 Add。

**首发前的最后决定(本轮已完成其中一项)**:

1. **版本号:已从 `0.1.0.dev0` 改为 `0.1.0`**(本次改动: `pyproject.toml`、4 个 bundle manifest 的
   `extension_version`、`fi_gelu`/`fi_sampling` 的 `__version__`、README 配对表)。
   压测过的理由:
   - 裸 `pip install vllm-ascend-split-batch` 即使只有 dev 版本也能装上(pip 的
     "找不到正式版则回落到预发布"规则;uv 默认 `if-necessary` 同样会回落到预发布)——但这是**回落**,不是正常路径;
   - 一旦有人写 `vllm-ascend-split-batch>=0.1`,**实测直接失败**:
     `ERROR: Could not find a version that satisfies the requirement ... (from versions: 0.1.0.dev0)`
     —— 因为 `0.1.0.dev0 < 0.1.0`,不满足 `>=0.1`;
   - 永久发行物带 `dev` 后缀在语义上也不对(它不是"开发版",是第一个公开版)。
2. **`host.version_range` 的宽度(未决,见下)**:现值是点钉
   `==0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`。它决定"装上也**只有**匹配宿主能 `run`":
   管理器对不匹配宿主报 `INCOMPATIBLE`,而 `vllm-hust-ext run` **会拒绝启动**(`extension enable`
   仍可用)。放宽带需要按 §4 清单在目标宿主上重验并留档,`AGENTS.md` 禁止写 `>=0`。
3. **可选**:给仓库 `pypi` 环境加 required reviewers;给 METADATA 加
   `Development Status :: 3 - Alpha` 分类器(诚实标注成熟度)。

**不做 TestPyPI 的含义**:`.github/workflows/publish.yml` 的下拉默认值是 `testpypi`(**安全默认**:
误点时最多在 TestPyPI 认证失败,不会污染正式索引)。只发 PyPI 时**必须在下拉里选 `pypi`**;
`testpypi` 那个 GitHub 环境不会被创建,不影响任何东西。

### 11.6 首发回执:0.1.0 已发布(2026-09-26T11:12Z) — 历史留档,已被 11.7 取代

[发布原始记录、安装字节摘要与历史验证](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/release.md#L733).
安装消费者入口：[PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.1.0/)、[Release](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.1.0)。已发布安装包保持不可改写；历史结果不自动继承到新宿主。

### 11.7 0.1.1 已发布:修 `kernels` 钉子 + 算子轮改走 Release 附件(2026-09-26)

[发布原始记录、安装字节摘要与历史验证](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/release.md#L789).
安装消费者入口：[PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.1.1/)、[Release](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.1.1)。已发布安装包保持不可改写；历史结果不自动继承到新宿主。

### 11.8 0.1.2 已发布:让 `test` extra 摘钉生效(2026-09-26T16:31Z)

[发布原始记录、安装字节摘要与历史验证](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/release.md#L879).
安装消费者入口：[PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.1.2/)、[Release](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.1.2)。已发布安装包保持不可改写；历史结果不自动继承到新宿主。

### 11.9 0.1.3 已发布(2026-09-27):`host.version_range` 放开为有界兼容窗口 + UpdatableGraph 接缝修复

[发布原始记录、安装字节摘要与历史验证](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/4bf3c6b7af6ddb4cdcf193e348302758386f8a2f/archive/mod-git-20261022/split-prose-source/snapshot/docs/release.md#L913).
安装消费者入口：[PyPI](https://pypi.org/project/vllm-ascend-split-batch/0.1.3/)、[Release](https://github.com/vLLM-HUST/vllm-ascend-split-batch-hust/releases/tag/v0.1.3)。已发布安装包保持不可改写；历史结果不自动继承到新宿主。

### 11.10 0.2.0 待发布(2026-10-24):cascade 源码删除 + 入口/extra 退役

**版本语义**:0.1.3 → **0.2.0**(minor)。删除 `cascade-attention` 入口点与 `[kernels]` extra 对已安装
消费者是破坏性变更,0.x 系列按 minor 进位而非 patch;同版本号不可覆盖,故不用 0.1.4。

**安装面(取自 0.2.0 wheel,非推测)**

| 项 | 值 | 判据 |
|---|---|---|
| `vllm.general_plugins` | 9 条 | `aot-cache-guard`/`fi-gelu`/`fi-sampling`/`fia-demask`/`gate-fp32`/`gdn-merge-inproj`/`mlp-chunk`/`rope-fix`/`zerocost-wiring` |
| `vllm_hust.extension_bundles` | 6 个 | wheel 内 6 份 `vllm-hust-extension-v0.2.json`,即下表 |
| extras | 仅 `test` | 无 `[kernels]`,无外部算子仓依赖 |
| cascade | 无模块、无入口、无 env 注入 | wheel 内无 `cascade*` 成员 |
| host 窗口 | `>=0.25.1rc2.dev125,<0.25.2` | 6 份 manifest 逐字一致;有界窗口,非点钉(§0.3) |

**bundle 激活状态(以 manifest 为准)**

| bundle | status |
|---|---|
| `org.vllm-hust.fia-demask` / `org.vllm-hust.gate-fp32` / `org.vllm-hust.zerocost-wiring` | `active`(默认 env-gated 关闭) |
| `org.vllm-hust.split-batch-full-graph` / `org.vllm-hust.rope-fix` / `org.vllm-hust.gdn-merge-inproj` | `import_only` |

**发布前本机核对**:`python -m build` + `tools/check_distribution.py dist` + `twine check` 通过
(whl 136 KB / sdist 109 KB);`ruff check`、`ruff format --check` 干净;`pytest -q -m "not host_tree"`
914 passed / 3 failed —— 三项在本轮改动前即红(宿主树漂到 0.28 新栈:`conv_weights` 锚点 x0、
discovery 期 import `planner`、opt-ops runtime helper 版本),与版本号改动无关;CI 侧由
`publish.yml` 再复核一次。

**发布状态**:待发布。本机无 PyPI 凭据,上传由 `v0.2.0` tag push 触发的 `publish.yml`(OIDC)执行;
Release 资产由 `release.yml` 从 PyPI 下载并核 digest 得到。

**发布后的三条机械判据**(任一不满足 = 发布未完成):
1. `https://pypi.org/pypi/vllm-ascend-split-batch/0.2.0/json` 返回 200,whl 与 tar.gz 的 sha256 与 Release 资产 `dist.sha256` 逐字一致;
2. `releases/tag/v0.2.0` 的资产是 PyPI 字节,不是本地重建;
3. 干净 venv `pip install vllm-ascend-split-batch==0.2.0`:`importlib.metadata.version` = `0.2.0`,`entry_points` 无 `cascade-attention`,`Provides-Extra` 只有 `test`。

# Distribution custody

Source distributions include only runtime source, packaging metadata, README,
license and provenance. Benchmark results, logs and evidence directories are not
installation inputs. CI, index publishing and GitHub Release upload must run
`python tools/check_distribution.py <artifact-directory>` after building or
downloading both distributions. The gate checks runtime inventory parity,
extension manifests and entry points as well as evidence exclusion.

Published versions are immutable: do not rebuild or overwrite their assets to
remove historical evidence. Historical result members belong in benchmark custody;
retire an installation artifact only after its installation consumers are resolved.

