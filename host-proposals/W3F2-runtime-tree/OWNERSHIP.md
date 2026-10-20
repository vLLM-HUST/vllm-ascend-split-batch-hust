# W3F2 / W3F3 runtime-tree patches —— 归属记录（PURPOSE §8 档②）

## 判定（2026-10-20 无卡轮，逐条回答「能否只用 general_plugins / --additional-config / --worker-cls / CLI+env 表达」）

| 层 | 内容 | 判定 | 归档 |
|---|---|---|---|
| W3F2 定义点+rebind+runner-hook | 权重加载后、dynamo 编译/图捕获前对 `QwenGatedDeltaNetAttention` 做一次性 GDN in_proj 合并；env `VLLM_HUST_GDN_MERGE_INPROJ=1` 门控，default-off | 宿主是不可修改镜像，`AscendGatedDeltaNetAttention.forward` 无可绑定的 merged-consumption seam；插件载体尝试已判 blocked（候选 `F2-plugin-carrier-migration`，禁止复制 forward）→ **档② 上游提案** | 本目录 patches/ 三件原件 |
| W3F3 shared-gate | shared scalar-gate pack（env `VLLM_HUST_SHARED_GATE_PACK`），定义点 `shared_experts.py` + runner-hook | 方法本体已 rejected（候选 `F3-shared-gate-pack`）；后备 `F3-alt-shared-slot` blocked → **不提案**，补丁仅作历史证据 | patches/ 内 W3F3 两件，无测试义务 |

## 上游约束

更新的上游树（vllm-ascend-hust f939ad767）已不含 `prepare_input_before_routed_experts`
接口对；换镜像时 W3F2/W3F3 必然 DRIFT。上游提案必须声明该 seam 差异；
投递 issue 属对外交互，需用户授权后执行。

## 原件与校验

原件 = L2 `/root/包体恢复/records/tree-patches/2026101*T*_W3F*-runtime-tree.*`，
本目录 patches/ 为逐字节副本（sha256 见 tests/test_w3f2_runtime_patch_replay.py 内联清单）。
测试以 orig 为基做 `patch --dry-run` 重放，保证「原件可作用于其快照基线」；
分层链判定与幂等重放见 tools/runtime-tree/apply-runtime-patches.sh。
