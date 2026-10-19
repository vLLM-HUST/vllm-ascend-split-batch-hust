import ast
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / 'host-proposals/comm-trace-stop-gil'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generator():
    return load(PROPOSAL / 'gen_patch.py', 'comm_gil_generator')


def test_patch_is_exact_gil_acquisition(generator):
    original = generator.ORIGINAL.read_bytes()
    patched = generator.patched_source(original)
    guard = '  pybind11::gil_scoped_acquire gil;\n'
    assert patched.replace(generator.ANCHOR + guard,
                           generator.ANCHOR, 1) == original.decode()
    assert generator.generate(original) == (PROPOSAL / 'patch.diff').read_text()
    npu_stop = (PROPOSAL / 'host_orig/npu_profiler.cpp.orig').read_text()
    stop = npu_stop.split('void stopNpuProfiler() {', 1)[1].split(
        'void finalizeNpuProfiler()', 1)[0]
    assert stop.index('gil_scoped_release') < stop.index('Command::kClear')


def test_drift_and_already_patched_source_rejected(generator):
    original = generator.ORIGINAL.read_bytes()
    for source in [original + b'\n', generator.patched_source(original).encode()]:
        with pytest.raises(ValueError, match='pinned'):
            generator.patched_source(source)


def test_patch_apply_and_rollback(generator, tmp_path):
    target = tmp_path / generator.HOST_REL
    target.parent.mkdir(parents=True)
    original = generator.ORIGINAL.read_bytes()
    target.write_bytes(original)
    patch = PROPOSAL / 'patch.diff'
    for extra in [[], ['-R']]:
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', *extra,
                        '-i', str(patch)], cwd=tmp_path, check=True,
                       capture_output=True)
        expected = (generator.patched_source(original).encode()
                    if not extra else original)
        assert target.read_bytes() == expected


def test_native_probe_uses_original_context_destructor(generator, monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'gen_patch', generator)
    runner = load(PROPOSAL / 'run_native_probe.py', 'comm_gil_native_runner')
    original = generator.ORIGINAL.read_text()
    source = runner.native_source(original)
    block = original.split('struct ThreadLocalResult;', 1)[1].split(
        '\nstruct PyCallInfo', 1)[0]
    assert block in source
    assert 'Py_DECREF((PyObject*)ctx_);' in source
    assert 'PyEval_SaveThread()' in source
    assert 'void PythonTracer::clear() {' in source


@pytest.mark.parametrize(('returncode', 'phases', 'expected'), [
    (0, ['before_stop', 'after_stop', 'after_finalize'], 'PASS'),
    (-11, ['before_stop'], 'FAIL'),
    (1, ['before_stop', 'after_stop'], 'FAIL'),
    (0, ['before_stop'], 'FAIL'),
    ('timeout', ['before_stop'], 'FAIL'),
])
def test_cpu_preflight_reports_failures(tmp_path, monkeypatch,
                                      returncode, phases, expected):
    checker = load(ROOT / 'tools/check_comm_profiler_stop.py', 'comm_gil_check')

    def fake_run(command, *, stdout, stderr, timeout):
        assert command[:2] == ['target-python', '-u']
        assert timeout == 5
        stdout.write(''.join(f'COMM_STOP_PHASE:{phase}\n' for phase in phases))
        stdout.flush()
        if returncode == 'timeout':
            raise subprocess.TimeoutExpired(command, timeout)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(checker.subprocess, 'run', fake_run)
    output = tmp_path / 'preflight'
    report = checker.check(output, python='target-python', timeout=5)
    assert report['status'] == expected
    assert report['mechanism_gates_accepted'] is False
    assert json.loads((output / 'result.json').read_text()) == report
    with pytest.raises(FileExistsError):
        checker.check(output, python='target-python')


def test_cpu_preflight_keeps_required_capture_options():
    checker = load(ROOT / 'tools/check_comm_profiler_stop.py', 'comm_gil_options')
    assert 'activities = {native.ProfilerActivity.CPU}' in checker.PROBE
    assert 'ProfilerActivity.NPU' not in checker.PROBE
    assert 'str(path), True, False, True, False, True,' in checker.PROBE
    collector = (ROOT / 'tools/collect_comm_mechanism.sh').read_text()
    assert collector.index('check_comm_profiler_stop.py') < collector.index(
        'opt bench serve')
    assert 'export LADDER_SLIM=0' in collector
    assert 'export LADDER_PROFILE_WITH_STACK=true' in collector


def test_collector_preflight_uses_serve_python():
    ladder = (ROOT / 'tools/bench-ladder.sh').read_text()
    serve_python = next(line.removeprefix('PY=') for line in ladder.splitlines()
                        if line.startswith('PY='))
    collector = (ROOT / 'tools/collect_comm_mechanism.sh').read_text()
    assert (f'{serve_python} "$script_dir/check_comm_profiler_stop.py" '
            '--output "$preflight"') in collector


def test_identity_hashes_mapped_library_not_package_default():
    checker = load(ROOT / 'tools/check_comm_profiler_stop.py', 'comm_gil_identity')
    functions = [node for node in ast.parse(checker.PROBE).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == 'loaded_library']
    namespace = {'Path': Path}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<probe>', 'exec'),
         namespace)
    locate = namespace['loaded_library']
    mappings = ('100-200 r--p 0 00:00 1 /optional/build/libtorch_npu.so\n'
                '200-300 r-xp 0 00:00 1 /optional/build/libtorch_npu.so\n')
    assert locate(mappings) == Path('/optional/build/libtorch_npu.so')
    for invalid in ['', mappings + '300-400 r-xp 0 00:00 2 /old/libtorch_npu.so\n']:
        with pytest.raises(RuntimeError, match='exactly one'):
            locate(invalid)
