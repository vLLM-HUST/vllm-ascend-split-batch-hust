# SPDX-License-Identifier: Apache-2.0
import os

from vllm_ascend.worker.worker import NPUWorker


class CommTraceWorker(NPUWorker):
    def __init__(self, *args, **kwargs):
        if os.environ.get("VLLM_HUST_COMM_TRACE") != "1":
            raise RuntimeError("CommTraceWorker requires VLLM_HUST_COMM_TRACE=1")
        import torch_npu
        from vllm_ascend.profiler.torch_npu_profiler import TorchNPUProfilerWrapper

        from .comm_trace import install

        install(TorchNPUProfilerWrapper, torch_npu.profiler)
        super().__init__(*args, **kwargs)
