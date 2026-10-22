# SPDX-License-Identifier: Apache-2.0
import os


def create_profiler(config, trace_name, profiler):
    if config.profiler != "torch" or not config.torch_profiler_dir:
        raise ValueError("comm trace requires torch profiler and an output directory")
    if not (config.torch_profiler_record_shapes and config.torch_profiler_with_stack):
        raise ValueError("comm trace requires explicit record_shapes and with_stack")
    experimental = profiler._ExperimentalConfig(
        export_type=[profiler.ExportType.Text, profiler.ExportType.Db],
        profiler_level=profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    return profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        record_shapes=True,
        with_stack=True,
        with_modules=True,
        profile_memory=config.torch_profiler_with_memory,
        experimental_config=experimental,
        on_trace_ready=profiler.tensorboard_trace_handler(
            config.torch_profiler_dir,
            worker_name=trace_name,
            analyse_flag=False,
        ),
    )


def install(wrapper, profiler):
    if os.environ.get("VLLM_HUST_COMM_TRACE") != "1":
        return False
    wrapper._create_profiler = staticmethod(
        lambda config, trace_name: create_profiler(config, trace_name, profiler)
    )
    return True
