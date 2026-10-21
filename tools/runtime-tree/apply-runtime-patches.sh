#!/usr/bin/env bash
# apply-runtime-patches.sh —— 运行树补丁幂等重放（补丁原件只存 /root PVC）
#
# 背景: 运行树 /vllm-workspace 在容器重建时回退为镜像状态（2026-10-15 实测,
#       F1/F2 补丁被抹）。补丁的原件/patch/校验和全部存 /root/包体恢复/records/tree-patches/,
#       本脚本按 MANIFEST 幂等重放到运行树, 供容器重建/换规格后执行。
#
# ⚠️ 落地归属（2026-10-16 明确, 见 /root/PURPOSE.md v3 §8）:
#       运行树补丁**不是**改动的归宿。归宿是插件库 vllm-ascend-split-batch-hust
#       （/root/vllm-workspace/vllm-ascend-split-batch-hust, git, main）；
#       本目录的 patch 只是「仓库 → 容器」的部署副本。改宿主源码前先确认能否用插件 seam 表达。
#
# 用法:
#   apply-runtime-patches.sh                     # 应用全部（已应用则跳过）
#   apply-runtime-patches.sh --check             # 只报状态不改树
#   apply-runtime-patches.sh --rollback <名字>    # 回滚单个条目（见 MANIFEST 名字列）
#   apply-runtime-patches.sh --rollback <名字> --force   # 被后续层覆盖时强制回滚（会丢后续层）
#
# 判定逻辑（逐条目；2026-10-16 改为**分层链**判定, 修掉同文件叠层导致前层永久 DRIFT 的假警）:
#   同一 <相对路径> 的多个条目按 MANIFEST 中的**出现顺序**构成一条层链 layer1 → layer2 → …
#   运行树文件 md5 == 本层 applied.md5            -> [skip]     已应用
#   运行树文件 md5 == 本层 orig.md5               -> [applied]  应用本层 patch（--check 下报 [todo]）
#   运行树文件 md5 == 后续任一层 applied.md5       -> [skip-cover] 已被后续层覆盖（层链, 非漂移）
#   其它                                          -> [DRIFT]    基线漂移, 人工处置
#   附加诊断: 同文件第 k 层的 orig 快照必须等于第 k-1 层的 applied 状态, 否则 [chain-warn]
#             （快照源不一致: 说明某一层的 .orig 是从别的状态拍下来的, 链不可信）
set -u
PDIR=/root/包体恢复/records/tree-patches
RTREE=/vllm-workspace/vllm-ascend/vllm_ascend
PY=/usr/local/python3.12.13/bin/python
MODE=${1:-apply}
ARG2=${2:-}

# MANIFEST: <名字> <相对运行树路径> <orig 文件名> <patch 文件名>
#   顺序有意义: 同一路径的条目按出现先后构成层链。
# 2026-10-18 04:23 决定（W1R GMM 层曾禁用）：blocker fixture-unreachable-fullgraph-decode
#   —— active FULL graph 下 decode(M=128) 在 capture 内执行、replay 不跑 Python，
#   hook 必须在任何 device 读取前跳过 ⇒ 取不到真实 decode counts；当时运行树回滚到宿主基线。
# 2026-10-18 07:5x 重新启用（用户授权 P1.0a 走 NONE 诊断口径）：
#   同一 blocker 在 cudagraph=NONE（无 capture，eager 执行）下不存在，故本层回到 MANIFEST。
#   约束（不得放宽）：本层产物只用于取**真实 per-expert counts** 做 L0 夹具；
#   NONE 窗**不作测量窗**、不更新 CURRENT、不与 FULL 口径数值相减或相除、
#   不用 NONE counts 推断 FULL 图内路由分布；FULL 下的 decode counts 仍然不可得。
#   default-off：未设 VLLM_HUST_GMM_* 时零 import / 零 IO / 行为与宿主一致（见下 patch 头注释）。
#   生命周期与重新启用条件见 host-proposals/W1R-gmm-tuning-config/OWNERSHIP.md §5。
MANIFEST="
w3f2-gdn-inproj|ops/gdn.py|20261015T1720_W3F2-runtime-tree.gdn.py.orig|20261015T1720_W3F2-runtime-tree.gdn.py.patch
w3f2-runner-hook|worker/model_runner_v1.py|20261015T1730_W3F2-runtime-tree.model_runner_v1.py.orig|20261015T1730_W3F2-runtime-tree.model_runner_v1.py.patch
w3f2-pq35-rebind|patch/worker/patch_qwen3_5.py|20261015T1731_W3F2-runtime-tree.patch_qwen3_5.py.orig|20261015T1731_W3F2-runtime-tree.patch_qwen3_5.py.patch
w3f3-shared-gate|ops/fused_moe/shared_experts.py|20261016T1130_W3F3-runtime-tree.shared_experts.py.orig|20261016T1130_W3F3-runtime-tree.shared_experts.py.patch
w3f3-runner-hook|worker/model_runner_v1.py|20261016T1130_W3F3-runtime-tree.model_runner_v1.py.orig|20261016T1130_W3F3-runtime-tree.model_runner_v1.py.patch
w1r-gmm-tuning|ops/fused_moe/routed_experts.py|20261018T0517_W1R-runtime-tree.routed_experts.py.orig|20261018T0517_W1R-runtime-tree.routed_experts.py.patch
d4site2-views-combined|ops/gdn.py|20261021T1355Z_d4_conv_hoist_site2.orig-gdn.py|20261021T1459Z_concatdViews.gdn.py.patch
moe-init-triton|ops/fused_moe/token_dispatcher.py|20261021T1710Z_moeInitTriton.token_dispatcher.py.orig|20261021T1710Z_moeInitTriton.token_dispatcher.py.patch
nz-skip-gate|ops/linear.py|20261021T1545Z_nzSkipGate.linear.py.orig|20261021T1545Z_nzSkipGate.linear.py.patch
"

md5_of() { md5sum "$1" 2>/dev/null | awk '{print $1}'; }

names=(); rels=(); origs=(); patches=()
while IFS='|' read -r name rel orig patch; do
  [ -z "$name" ] && continue
  names+=("$name"); rels+=("$rel"); origs+=("$orig"); patches+=("$patch")
done <<< "$MANIFEST"
N=${#names[@]}

# before[i]=该层期望的上游状态; after[i]=该层应用后的状态（未应用 = none）
declare -a before after
for ((i=0; i<N; i++)); do
  before[i]=$(md5_of "$PDIR/${origs[i]}")
  after[i]=$(cat "$PDIR/${origs[i]%.orig}.applied.md5" 2>/dev/null || echo none)
done

rc=0
for ((i=0; i<N; i++)); do
  name=${names[i]}; rel=${rels[i]}; orig=${origs[i]}; patch=${patches[i]}
  tgt="$RTREE/$rel"
  cur=$(md5_of "$tgt")

  # --- 层链一致性诊断: 本层 orig 快照应等于同文件上一层的 applied 状态
  for ((j=i-1; j>=0; j--)); do
    if [ "${rels[j]}" = "$rel" ]; then
      [ "${before[i]}" = "${after[j]}" ] || \
        echo "[chain-warn] $name: $rel 第 $((i+1)) 层 orig 快照(${before[i]:0:8}) != 上一层 $((j+1)) 层 applied(${after[j]:0:8})（快照源不一致, 链不可信）"
      break
    fi
  done

  # --- rollback 分支
  if [ "$MODE" = "--rollback" ] && [ "$ARG2" = "$name" ]; then
    for ((k=i+1; k<N; k++)); do
      if [ "${rels[k]}" = "$rel" ] && [ "$cur" = "${after[k]}" ] && [ "${3:-}" != "--force" ]; then
        echo "[rollback-拒绝] $name: 当前状态是后续层 ${names[k]} 的 applied；强制回滚会丢后续层，需 --force"
        rc=1; continue 2
      fi
    done
    cp "$PDIR/$orig" "$tgt" && rm -f "$PDIR/${orig%.orig}.applied.md5"
    "$PY" -m py_compile "$tgt" && echo "[rollback] $name -> $rel 已还原 orig"; continue
  fi

  if [ "$cur" = "${after[i]}" ] && [ "$cur" != "none" ]; then
    echo "[skip] $name: 已应用 ($rel)"
    continue
  fi

  # 被后续层覆盖 = 层链正常推进, 不是漂移
  covered=""
  for ((k=i+1; k<N; k++)); do
    if [ "${rels[k]}" = "$rel" ] && [ "$cur" = "${after[k]}" ] && [ "$cur" != "none" ]; then
      covered="${names[k]}"; break
    fi
  done
  if [ -n "$covered" ]; then
    echo "[skip-cover] $name: 已被后续层 $covered 覆盖（层链推进, 非漂移）"
    continue
  fi

  if [ "$cur" = "${before[i]}" ]; then
    if [ "$MODE" = "--check" ]; then echo "[todo] $name: 待应用 ($rel)"; continue; fi
    if patch -s "$tgt" "$PDIR/$patch"; then
      "$PY" -m py_compile "$tgt" || { echo "[FAIL] $name py_compile"; rc=1; continue; }
      md5_of "$tgt" > "$PDIR/${orig%.orig}.applied.md5"
      after[i]=$(md5_of "$tgt")
      echo "[applied] $name -> $rel"
    else
      echo "[FAIL] $name: patch 应用失败（patch -R 回滚现场后排查）"; patch -R -s "$tgt" "$PDIR/$patch" 2>/dev/null; rc=1
    fi
  else
    echo "[DRIFT] $name: 运行树 $rel 既非本层 orig、非本层 applied、也非后续层 applied（基线漂移, 人工处置）"
    rc=1
  fi
done

# 接口面校验（2026-10-16 加；纯静态/不 import/不占卡，秒级）:
#   py_compile 只保证「语法对」，保证不了「调用点与定义点同一状态」——
#   2026-10-16 11:53 的事故就是这样漏过去的（fused_moe.py 调 AscendSharedExperts.
#   prepare_input_before_routed_experts，而 shared_experts.py 当时没有该方法），
#   代价是连续 5 次起服未就绪。断言表: scripts/tree-seam-assertions.json
if [ "$MODE" != "--check" ]; then
  if ! "$PY" "$(dirname "$0")/check-tree-seams.py"; then
    echo "[FAIL] 接口面校验未过（见上）→ 起服会以 AttributeError 形态延迟 2-5min 才暴露；先修再起"
    rc=1
  fi
fi
exit $rc
