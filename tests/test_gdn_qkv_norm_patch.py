# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import ast
import importlib.util
import os
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend_split_batch import gdn_qkv_norm

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / 'host-proposals/GDN-qkv-norm'
spec = importlib.util.spec_from_file_location(
    'gdn_norm_patch', PROPOSAL / 'gen_patch.py',
)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def test_patch_exact_and_reversible(tmp_path):
    source = (PROPOSAL / 'host_orig/gdn.py.orig').read_text()
    patched = gen.build_patched(source)
    ast.parse(patched)
    assert (PROPOSAL / 'patch.diff').read_text() == gen.make_diff(source)
    dest = tmp_path / gen.TARGET
    dest.parent.mkdir(parents=True)
    dest.write_text(source)
    for options in [[], ['-R']]:
        result = subprocess.run(
            ['patch', '-p1', '--batch', *options, '-i', str(PROPOSAL / 'patch.diff')],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert dest.read_text() == (source if options else patched)
    with pytest.raises(ValueError, match='drift'):
        gen.build_patched(source + '\n')
    with pytest.raises(ValueError, match='drift'):
        gen.build_patched(patched)


@pytest.mark.parametrize('enabled,norm_env,expected', [
    (None, None, False), ('0', '0', False), ('1', '0', True),
    ('1', '1', False), ('yes', '0', False),
])
def test_default_off_and_norm_contract(monkeypatch, enabled, norm_env, expected):
    for name, value in [('VLLM_HUST_GDN_QKV_NORM', enabled),
                        ('USE_DEFAULT_FLA_NORM', norm_env)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    ns = {'os': os}
    exec(gen.FLAG, ns)
    assert ns['_FUSE_GDN_QKV_NORM'] is expected


@pytest.mark.parametrize('reject', [
    None, 'disabled', 'spec_mask', 'spec_config', 'prefill', 'no_decode',
    'decode_tokens', 'none', 'shape', 'stride', 'dtype', 'cpu', 'grad',
    'q_dim', 'v_dim', 'head_k', 'head_v', 'pcp',
])
def test_callsite_guards_and_no_double_norm(monkeypatch, reject):
    calls = []
    layer = SimpleNamespace(
        key_dim=2048, value_dim=4096, tp_size=2,
        head_k_dim=128, head_v_dim=128, num_spec=0,
        rearrange_mixed_qkv=lambda x: calls.append('unpack') or ('q', 'k', 'v'),
    )
    x = SimpleNamespace(shape=(16, 4096), stride=lambda: (4096, 1),
                        dtype='bf16', device=SimpleNamespace(type='npu'),
                        requires_grad=False)
    metadata = SimpleNamespace(num_prefills=0, num_decodes=16, num_decode_tokens=16)
    ns = dict(self=layer, _FUSE_GDN_QKV_NORM=True, spec_sequence_masks=None,
              attn_metadata=metadata, mixed_qkv_non_spec=x,
              torch=SimpleNamespace(bfloat16='bf16'),
              get_pcp_group=lambda: SimpleNamespace(world_size=1),
              l2norm_fwd=lambda t: calls.append('norm') or 'norm_' + t)
    if reject == 'disabled':
        ns['_FUSE_GDN_QKV_NORM'] = False
    elif reject == 'spec_mask':
        ns['spec_sequence_masks'] = object()
    elif reject == 'spec_config':
        layer.num_spec = 1
    elif reject == 'prefill':
        metadata.num_prefills = 1
    elif reject == 'no_decode':
        metadata.num_decodes = 0
    elif reject == 'decode_tokens':
        metadata.num_decode_tokens = 15
    elif reject == 'none':
        ns['mixed_qkv_non_spec'] = None
    elif reject == 'shape':
        x.shape = (17, 4096)
    elif reject == 'stride':
        x.stride = lambda: (6176, 1)
    elif reject == 'dtype':
        x.dtype = 'fp16'
    elif reject == 'cpu':
        x.device.type = 'cpu'
    elif reject == 'grad':
        x.requires_grad = True
    elif reject == 'q_dim':
        layer.key_dim = 1024
    elif reject == 'v_dim':
        layer.value_dim = 2048
    elif reject == 'head_k':
        layer.head_k_dim = 64
    elif reject == 'head_v':
        layer.head_v_dim = 64
    elif reject == 'pcp':
        ns['get_pcp_group'] = lambda: SimpleNamespace(world_size=2)
    monkeypatch.setattr(gdn_qkv_norm, 'unpack_norm',
                        lambda t: calls.append('fused') or ('nq', 'nk', 'v'))
    exec(textwrap.dedent(gen.UNPACK_NEW), ns)
    exec(textwrap.dedent(gen.NORM_NEW), ns)
    assert calls == (['fused'] if reject is None else ['unpack', 'norm', 'norm'])
    assert ns['qk_non_spec_normalized'] is (reject is None)


def test_only_pure_decode_norm_pair_changed():
    source = (PROPOSAL / 'host_orig/gdn.py.orig').read_text()
    patched = gen.build_patched(source)
    for line in [
        'query_spec = l2norm_fwd(query_spec)',
        'key_spec = l2norm_fwd(key_spec)',
        'query_decode = l2norm_fwd(query_decode)',
        'key_decode = l2norm_fwd(key_decode)',
        'use_qk_l2norm_in_kernel=True',
    ]:
        assert source.count(line) == patched.count(line) == 1
