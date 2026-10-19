#!/usr/bin/env bash
set -euo pipefail
export VLLM_HUST_COMM_TRACE=1
export LADDER_PROFILE_MAX_ITERS=32
export LADDER_PROFILE_WITH_STACK=true
export LADDER_SLIM=0
export LADDER_STOP_SERVE_FOR_PARSE=1
export LADDER_PROFILE_LOAD=mmlu-pro
export LADDER_PROFILE_MMLU_N=16
export LADDER_EVAL_CONC=16
export LADDER_LOAD_CALIBER=mmlu-pro-gdn-high-concurrency-longctx
export LADDER_SERVE_EXTRA='--compilation-config {"mode":"VLLM_COMPILE","cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8,16,32]} --worker-cls vllm_ascend_split_batch.comm_trace_worker.CommTraceWorker'
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
preflight="/root/包体恢复/records/probes/$(date -u +%Y%m%dT%H%M%SZ)_comm-stop-preflight-$$"
/usr/local/python3.12.13/bin/python "$script_dir/check_comm_profiler_stop.py" --output "$preflight"
opt bench serve --profile deep
opt bench profile --candidate comm-free-current-window-diagnostic \
  --trace-tier deep \
  --screen-report /root/包体恢复/records/ladder/1018-gdn-norm-B-plain/screen.json \
  --collection-reason mechanism-gap \
  --gap 'Missing rank1 DB, allreduce shapes/stacks and complete graph execution intervals' \
  --reuse-evidence /root/包体恢复/records/probes/20261019T055422Z_comm-free-paired-forensics.json \
  --repeat-reason 'One user-requested dual-rank diagnostic; no overlap or E2E comparison' 16
window=$(cat /root/包体恢复/records/ladder/.current-prof-dir)
mapfile -t ranks < <(find "$window" -mindepth 1 -maxdepth 1 -type d -name '*_ascend_pt' | sort)
[ "${#ranks[@]}" -eq 2 ] || { echo 'Expected exactly two raw rank directories' >&2; exit 3; }
for rank in "${ranks[@]}"; do
  out="$rank/ASCEND_PROFILER_OUTPUT"
  if ! compgen -G "$out/ascend_pytorch_profiler_*.db" >/dev/null; then
    [ ! -e "$out" ] || { echo "Partial parse retained: $out" >&2; exit 3; }
    /root/venvs/rv1/bin/python - "$rank" <<'PY'
import sys
from torch_npu.profiler.profiler import analyse
analyse(sys.argv[1], max_process_number=1)
PY
  fi
  for artifact in 'ascend_pytorch_profiler_*.db' kernel_details.csv trace_view.json 'communication*.json' op_statistic.csv; do
    compgen -G "$out/$artifact" >/dev/null || { echo "Missing $out/$artifact" >&2; exit 3; }
  done
  opt bench recover-profile "$rank" --caliber mmlu-pro-gdn-high-concurrency-longctx
done
