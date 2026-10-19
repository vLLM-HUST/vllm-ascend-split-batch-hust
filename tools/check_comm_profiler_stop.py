#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import subprocess
import sys
from pathlib import Path

PROBE = r'''
import faulthandler
import hashlib
import json
import sys
from pathlib import Path
import torch
import torch_npu
from torch_npu._C import _profiler as native
from torch_npu.version import git_version

faulthandler.enable()
def loaded_library(maps):
    paths = {line.split(maxsplit=5)[-1].strip()
             for line in maps.splitlines() if line.endswith('/libtorch_npu.so')}
    if len(paths) != 1:
        raise RuntimeError('expected exactly one mapped libtorch_npu.so')
    return Path(paths.pop())
library = loaded_library(Path('/proc/self/maps').read_text())
with library.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
print(json.dumps({'python': sys.executable, 'torch': torch.__version__,
                  'torch_npu': torch_npu.__version__, 'git_version': git_version,
                  'library': str(library), 'library_sha256': digest}), flush=True)
path = Path(sys.argv[1])
path.mkdir()
activities = {native.ProfilerActivity.CPU}
config = native.NpuProfilerConfig(
    str(path), True, False, True, False, True, native._ExperimentalConfig())
native._init_profiler(str(path), activities)
native._start_profiler(config, activities)
def work():
    x = torch.ones((16, 32))
    return (x @ x.T).sum()
for _ in range(3):
    work()
print('COMM_STOP_PHASE:before_stop', flush=True)
native._stop_profiler()
print('COMM_STOP_PHASE:after_stop', flush=True)
native._finalize_profiler()
print('COMM_STOP_PHASE:after_finalize', flush=True)
'''


def check(output, *, python=sys.executable, timeout=60):
    output.mkdir(parents=True, exist_ok=False)
    command = [str(python), '-u', '-c', PROBE, str(output / 'raw')]
    timed_out = False
    with (output / 'probe.log').open('x') as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
    text = (output / 'probe.log').read_text(errors='replace')
    passed = (returncode == 0 and all(
        f'COMM_STOP_PHASE:{phase}' in text
        for phase in ['before_stop', 'after_stop', 'after_finalize']))
    report = {'schema': 'comm-profiler-stop-preflight.v1',
              'status': 'PASS' if passed else 'FAIL',
              'returncode': returncode, 'timeout': timed_out,
              'cpu_only': True, 'record_shapes': True,
              'with_stack': True, 'with_modules': True,
              'native_stop_finalize_verified': passed,
              'mechanism_gates_accepted': False}
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    report = check(args.output, python=args.python, timeout=args.timeout)
    print(json.dumps(report, indent=2))
    if report['status'] != 'PASS':
        print('Refusing CommTrace collection: repair the native PythonTracer '
              'GIL cleanup and pass this CPU preflight before starting vLLM.',
              file=sys.stderr)
        raise SystemExit(3)


if __name__ == '__main__':
    main()
