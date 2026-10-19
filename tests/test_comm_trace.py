from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend_split_batch.comm_trace import create_profiler, install


def config():
    return SimpleNamespace(
        profiler='torch', torch_profiler_dir='/trace',
        torch_profiler_record_shapes=True, torch_profiler_with_stack=True,
        torch_profiler_with_memory=False,
    )


def api():
    return SimpleNamespace(
        _ExperimentalConfig=Mock(), profile=Mock(),
        tensorboard_trace_handler=Mock(),
        ExportType=SimpleNamespace(Text='text', Db='db'),
        ProfilerLevel=SimpleNamespace(Level1=1),
        AiCMetrics=SimpleNamespace(PipeUtilization='pipe'),
        ProfilerActivity=SimpleNamespace(CPU='cpu', NPU='npu'),
    )


def test_full_trace_configuration():
    profiler = api()
    assert create_profiler(config(), 'rank1', profiler) is profiler.profile.return_value
    kw = profiler.profile.call_args.kwargs
    assert kw['record_shapes'] and kw['with_stack'] and kw['with_modules']
    assert kw['activities'] == ['cpu', 'npu']
    exp = profiler._ExperimentalConfig.call_args.kwargs
    assert exp['export_type'] == ['text', 'db']
    assert exp['data_simplification'] is False
    profiler.tensorboard_trace_handler.assert_called_once_with(
        '/trace', worker_name='rank1', analyse_flag=False,
    )


@pytest.mark.parametrize('field', [
    'torch_profiler_record_shapes', 'torch_profiler_with_stack',
    'torch_profiler_dir', 'profiler',
])
def test_missing_contract_fails_before_collection(field):
    cfg, profiler = config(), api()
    setattr(cfg, field, False)
    with pytest.raises(ValueError):
        create_profiler(cfg, 'rank0', profiler)
    profiler.profile.assert_not_called()


def test_default_off_and_explicit_install(monkeypatch):
    class Wrapper:
        _create_profiler = staticmethod(lambda: 'stock')

    original, profiler = Wrapper._create_profiler, api()
    monkeypatch.delenv('VLLM_HUST_COMM_TRACE', raising=False)
    assert not install(Wrapper, profiler)
    assert Wrapper._create_profiler is original
    monkeypatch.setenv('VLLM_HUST_COMM_TRACE', '1')
    assert install(Wrapper, profiler)
    assert Wrapper._create_profiler(config(), 'rank0') is profiler.profile.return_value
