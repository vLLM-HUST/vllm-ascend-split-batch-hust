# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools/bench_gdn_l3.py'
spec = importlib.util.spec_from_file_location('gdn_l3', PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize('bad', [False, True])
def test_batch_indices(monkeypatch, bad):
    choices = [{'index': i, 'text': str(i), 'finish_reason': 'length'}
               for i in reversed(range(16))]
    if bad:
        choices[0]['index'] = 0
    data = {'choices': choices, 'usage': {'completion_tokens': 2048}}

    def respond(req, timeout):
        assert req.full_url.endswith('/v1/completions')
        assert len(json.loads(req.data)['prompt']) == 16
        return io.BytesIO(json.dumps(data).encode())

    monkeypatch.setattr(bench.urllib.request, 'urlopen', respond)
    if bad:
        with pytest.raises(ValueError):
            bench.batch_capture('http://test', 'model', 16, 128)
    else:
        rows = bench.batch_capture('http://test', 'model', 16, 128)
        assert [r['text'] for r in rows] == [str(i) for i in range(16)]


def test_ladder_forwards_l3_arguments():
    shell = (PATH.parent / 'bench-ladder.sh').read_text()
    assert 'l3) shift; cmd_l3 "$@" ;;' in shell
    assert '"$gdn_dir/bench_gdn_l3.py" "$@"' in shell


def test_concurrent_capture_preserves_reasoning():
    def request(body):
        assert body['temperature'] == 0 and body['seed'] == 42
        assert body['ignore_eos'] is True
        return {'choices': [{'message': {'content': '', 'reasoning': 'proof'},
                             'finish_reason': 'length'}],
                'usage': {'completion_tokens': 128, 'prompt_tokens': 600}}

    rows = bench.capture('unused', 'model', 16, 128, request=request)
    assert len({r['id'] for r in rows}) == 16
    assert all('proof' in r['text'] for r in rows)


@pytest.mark.parametrize('tokens,text', [(127, 'x'), (128, '')])
def test_incomplete_capture_fails(tokens, text):
    def request(body):
        return {'choices': [{'message': {'content': text},
                             'finish_reason': 'length'}],
                'usage': {'completion_tokens': tokens, 'prompt_tokens': 600}}

    with pytest.raises(ValueError):
        bench.capture('unused', 'model', 16, 128, request=request)
