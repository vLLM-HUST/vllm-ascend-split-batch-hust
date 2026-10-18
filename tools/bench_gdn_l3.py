# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import argparse
import concurrent.futures
import json
import threading
import urllib.request
from pathlib import Path

PROMPT = (
    'Explain the relationship between matrix multiplication and linear maps. '
    'Give a worked example and verify every arithmetic step. '
) * 16


def capture(base_url, model, count, max_tokens, request=None):
    barrier = threading.Barrier(count)

    def send(index):
        body = {'model': model, 'messages': [{'role': 'user', 'content': PROMPT}],
                'max_tokens': max_tokens, 'temperature': 0, 'seed': 42,
                'ignore_eos': True}
        barrier.wait(timeout=60)
        if request is None:
            req = urllib.request.Request(
                base_url + '/v1/chat/completions',
                data=json.dumps(body).encode(),
                headers={'Content-Type': 'application/json'},
            )
            with urllib.request.urlopen(req, timeout=900) as response:
                result = json.loads(response.read())
        else:
            result = request(body)
        choice = result['choices'][0]
        message = choice['message']
        text = message.get('content') or ''
        reasoning = message.get('reasoning') or message.get('reasoning_content') or ''
        usage = result['usage']
        if usage['completion_tokens'] != max_tokens or not (text or reasoning):
            raise ValueError('incomplete or empty generation')
        return {'id': f'gdn16-{index:02}', 'prompt': PROMPT,
                'text': json.dumps({'content': text, 'reasoning': reasoning},
                                   ensure_ascii=False, sort_keys=True),
                'completion_tokens': usage['completion_tokens'],
                'prompt_tokens': usage['prompt_tokens'],
                'finish_reason': choice['finish_reason']}

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        return list(pool.map(send, range(count)))


def batch_capture(base_url, model, count, max_tokens):
    prompts = [PROMPT + f'\nExample identifier {i}.\nAnswer:' for i in range(count)]
    body = {'model': model, 'prompt': prompts, 'max_tokens': max_tokens,
            'temperature': 0, 'seed': 42, 'ignore_eos': True}
    req = urllib.request.Request(
        base_url + '/v1/completions', data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=900) as response:
        result = json.loads(response.read())
    choices = sorted(result['choices'], key=lambda c: c['index'])
    if [c['index'] for c in choices] != list(range(count)):
        raise ValueError('batch index coverage mismatch')
    if result['usage']['completion_tokens'] != count * max_tokens:
        raise ValueError('incomplete batch generation')
    if any(not c['text'] for c in choices):
        raise ValueError('empty batch output')
    return [{'id': f'batch16-{c["index"]:02}', 'text': c['text'],
             'prompt': prompts[c['index']], 'finish_reason': c['finish_reason'],
             'batch_usage': result['usage']} for c in choices]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--batched-raw', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        p.error('refusing to overwrite evidence')
    if args.max_tokens < 1:
        p.error('--max-tokens must be positive')
    base = f'http://127.0.0.1:{args.port}'
    with urllib.request.urlopen(base + '/v1/models', timeout=30) as response:
        model = json.loads(response.read())['data'][0]['id']
    run = batch_capture if args.batched_raw else capture
    rows = run(base, model, 16, args.max_tokens)
    with args.out.open('x') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'{len(rows)} concurrent captures -> {args.out}')


if __name__ == '__main__':
    main()
