# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 vLLM-HUST contributors.

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from vllm_ascend_split_batch import gdn_qkv_norm as api  # noqa: E402
from vllm_ascend_split_batch.gdn_qkv_unpack.reference import (  # noqa: E402
    rearrange_mixed_qkv_reference,
)


def unpack(x):
    return rearrange_mixed_qkv_reference(
        x, q_dim=1024, v_dim=2048, head_k_dim=128, head_v_dim=128,
    )


def norm(x):
    f = x.float()
    return (f * torch.rsqrt((f * f).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)


@pytest.mark.parametrize('tokens', [17, 825, 1489, 1587, 2048])
@pytest.mark.parametrize('prefix', [1, 16])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
def test_mixed_prefix_reuse_layout_and_alias(tokens, prefix, dtype):
    torch.manual_seed(0)
    x = torch.randn(tokens, 4096, dtype=dtype)
    saved = x.clone()
    full = unpack(x)
    again = unpack(x[:prefix])
    reused = tuple(t[:, :prefix] for t in full)
    for new, old in zip(reused, again, strict=True):
        assert new.is_contiguous()
        assert new.stride()[1:] == old.stride()[1:]
        assert new.squeeze(0).stride() == old.squeeze(0).stride()
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        assert new.view(-1, 128).is_contiguous()
    suffix = tuple(t[:, prefix:].clone() for t in full)
    for i in [0, 1]:
        torch.testing.assert_close(norm(reused[i]), norm(again[i]), rtol=0, atol=0)
        assert norm(reused[i]).untyped_storage().data_ptr() != (
            reused[i].untyped_storage().data_ptr())
    for new, old in zip(full, suffix, strict=True):
        torch.testing.assert_close(new[:, prefix:], old, rtol=0, atol=0)
    torch.testing.assert_close(x, saved, rtol=0, atol=0)


def test_wrapper_mapping_and_output_lifetime(monkeypatch):
    monkeypatch.setattr(api, 'validate', lambda x: None)

    def launch(x, flat):
        q, k, v = unpack(x)
        for dst, src in zip(api.output_views(flat), (norm(q), norm(k), v),
                            strict=True):
            dst.copy_(src)

    monkeypatch.setattr(api, 'launch_into', launch)
    x = torch.randn(16, 4096, dtype=torch.bfloat16)
    before = x.clone()
    first = api.unpack_norm(x)
    saved = [t.clone() for t in first]
    second = api.unpack_norm(torch.zeros_like(x))
    assert first[0].untyped_storage().data_ptr() != (
        second[0].untyped_storage().data_ptr())
    assert len({t.untyped_storage().data_ptr() for t in first}) == 1
    assert [t.storage_offset() for t in first] == [0, 16384, 32768]
    for a, b in zip(first, saved, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize('change', [
    {'shape': (17, 4096)}, {'stride': lambda: (6176, 1)},
    {'dtype': torch.float32}, {'requires_grad': True},
    {'device': SimpleNamespace(type='cpu')},
])
def test_guard_rejects_unsupported(change):
    attrs = dict(shape=(16, 4096), stride=lambda: (4096, 1),
                 dtype=torch.bfloat16, requires_grad=False,
                 device=SimpleNamespace(type='npu'))
    attrs.update(change)
    with pytest.raises(ValueError):
        api.validate(SimpleNamespace(**attrs))


def test_import_has_no_device_or_triton_side_effect():
    script = '''
import sys
import vllm_ascend_split_batch.gdn_qkv_norm
assert 'triton' not in sys.modules
assert 'torch_npu' not in sys.modules
assert 'torch' not in sys.modules
'''
    p = subprocess.run([sys.executable, '-c', script], capture_output=True,
                       text=True, timeout=30)
    assert p.returncode == 0, p.stderr


@pytest.mark.skipif(os.environ.get('GDN_TEST_OFFLINE_COMPILE') != '1',
                    reason='requires Ascend offline compiler')
def test_fused_kernel_offline_compile(tmp_path):
    script = '''
import sys
from pathlib import Path
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from vllm_ascend_split_batch.gdn_qkv_norm_kernel import unpack_norm_kernel
compiled = triton.compile(
    ASTSource(unpack_norm_kernel, {'X': '*bf16', 'Y': '*bf16'}, {}),
    target=GPUTarget('npu', 'Ascend910B2', 0),
)
ir = compiled.asm['ttadapter']
Path(sys.argv[1]).write_text(ir)
assert 'sync_block_lock' not in ir
assert 'arith.cmpf' not in ir
'''
    p = subprocess.run([sys.executable, '-c', script, str(tmp_path / 'fused.ir')],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
