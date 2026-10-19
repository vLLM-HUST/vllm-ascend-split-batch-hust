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
opt bench serve --profile deep
opt bench profile --candidate comm-free-current-window-diagnostic \
  --trace-tier deep \
  --screen-report /root/包体恢复/records/ladder/1018-gdn-norm-B-plain/screen.json \
  --collection-reason mechanism-gap \
  --gap 'Missing rank1 DB, allreduce shapes/stacks and complete graph execution intervals' \
  --reuse-evidence /root/包体恢复/records/probes/20261019T055422Z_comm-free-paired-forensics.json \
  --repeat-reason 'One user-requested dual-rank diagnostic; no overlap or E2E comparison' 16
