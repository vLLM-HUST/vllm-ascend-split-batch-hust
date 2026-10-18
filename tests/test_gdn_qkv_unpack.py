# Copyright (c) 2026 vLLM-HUST contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for ``gdn_qkv_unpack`` (single-pass packed qkv unpack).

The operator is split so that neither the mapping nor the validation needs a
device: ``api._plan`` (pure validation + geometry) and ``api._views`` (pure
views) are driven directly, with a test-side replay of the kernel's mapping.
``rearrange_mixed_qkv`` is the only allocating/launching entry; its own tests
run with the device check stubbed *and* the kernel module replaced by a
recorder, so no path here can reach the launcher.

The recorder replays the mapping from the scalars the host passed (grid,
segment widths, row stride, tile count, BLOCK), so a wrong host argument shows
up as wrong bytes instead of a passing mock.

Oracle: ``reference.rearrange_mixed_qkv_reference`` (the verbatim host chain,
split -> reshape -> cat -> view) for values, shapes, strides and storage
offsets.

Not covered (needs the device rung): numeric equality of the real kernel, mask
lowering, block-size choice.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch", reason="the operator package needs torch")

from vllm_ascend_split_batch.gdn_qkv_unpack import api, reference  # noqa: E402

_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

HEAD_K = 128
HEAD_V = 128

#: Passed to ``_plan`` so no test imports the triton module; it also exercises
#: multi-tile rows, tile tails and the idle programs of the narrower segments.
BLOCK_SMALL = 64

#: (n_rows, q_dim, v_dim, head_k_dim, head_v_dim)
_CASES = [
    pytest.param(1, 512, 1024, HEAD_K, HEAD_V, id="decode-t1"),
    pytest.param(3, 512, 1024, HEAD_K, HEAD_V, id="decode-t3"),
    pytest.param(17, 136, 280, 8, 8, id="tails-under-block"),
    pytest.param(5, 8, 1024, 8, HEAD_V, id="q-narrower-than-v"),
    pytest.param(7, 384, 128, HEAD_K, HEAD_V, id="v-narrower-than-q"),
]


# --------------------------------------------------------------------- helpers
def _packed(n_rows: int, q_dim: int, v_dim: int, dtype, *, seed=0, pad=0, k_dim=None):
    """Packed qkv; ``pad > 0`` returns a non-contiguous row slice."""
    packed_width = q_dim + (q_dim if k_dim is None else k_dim) + v_dim
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, packed_width + pad, generator=gen, dtype=torch.float32)
    x = x.to(dtype)
    return x if pad == 0 else x[:, :packed_width]


def _plan_of(x, q_dim, v_dim, head_k=HEAD_K, head_v=HEAD_V, k_dim=None, **kwargs):
    return api._plan(
        x,
        q_dim=q_dim,
        v_dim=v_dim,
        head_k_dim=head_k,
        head_v_dim=head_v,
        k_dim=k_dim,
        block=BLOCK_SMALL,
        **kwargs,
    )


def _plan_from_launch(args, kwargs):
    """Rebuild the plan from what the host actually passed to the kernel."""
    _, _, n_rows, q_dim, k_dim, v_dim, row_stride, n_col_tiles = args
    return api._Plan(
        n_rows=n_rows,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        head_k_dim=0,
        head_v_dim=0,
        row_stride=row_stride,
        block=kwargs["BLOCK"],
        n_col_tiles=n_col_tiles,
        flat_elems=n_rows * (q_dim + k_dim + v_dim),
    )


def _replay(mixed, flat, plan):
    """Replay the kernel mapping program by program from the plan's scalars."""
    widths = (plan.q_dim, plan.k_dim, plan.v_dim)
    src_bases = (0, plan.q_dim, plan.q_dim + plan.k_dim)
    dst_bases = (
        0,
        plan.n_rows * plan.q_dim,
        plan.n_rows * plan.q_dim + plan.n_rows * plan.k_dim,
    )
    for seg in range(3):
        width = widths[seg]
        for pid in range(plan.n_rows * plan.n_col_tiles):
            row = pid // plan.n_col_tiles
            col = (pid - row * plan.n_col_tiles) * plan.block + torch.arange(plan.block)
            col = col[col < width]
            flat[dst_bases[seg] + row * width + col] = mixed[row, src_bases[seg] + col]


def _unpacked(x, q_dim, v_dim, head_k=HEAD_K, head_v=HEAD_V, k_dim=None):
    """plan -> test-owned flat buffer -> replay -> official views."""
    plan = _plan_of(x, q_dim, v_dim, head_k, head_v, k_dim)
    flat = torch.empty(plan.flat_elems, dtype=x.dtype)
    _replay(x, flat, plan)
    return plan, flat, api._views(flat, plan)


def _span(tensor):
    start = tensor.storage_offset() * tensor.element_size()
    return start, start + tensor.numel() * tensor.element_size()


# ------------------------------------------------------------ the mapping test
@pytest.mark.parametrize("n_rows,q_dim,v_dim,head_k,head_v", _CASES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_values_layout_and_offsets_match_the_host_chain(
    n_rows, q_dim, v_dim, head_k, head_v, dtype
):
    x = _packed(n_rows, q_dim, v_dim, dtype)
    _, _, got = _unpacked(x, q_dim, v_dim, head_k, head_v)
    want = reference.rearrange_mixed_qkv_reference(
        x, q_dim=q_dim, v_dim=v_dim, head_k_dim=head_k, head_v_dim=head_v
    )

    for name, g, w in zip(("q", "k", "v"), got, want, strict=True):
        assert g.shape == w.shape, name
        assert g.stride() == w.stride(), name
        assert g.storage_offset() == w.storage_offset(), name
        assert g.dtype == x.dtype, name
        assert torch.equal(g, w), name


def test_reference_matches_the_documented_index_mapping():
    n_rows, q_dim, v_dim, head_k, head_v = 3, 8, 16, 4, 8
    x = torch.arange(n_rows * (2 * q_dim + v_dim), dtype=torch.float32).reshape(
        n_rows, -1
    )
    query, key, value = reference.rearrange_mixed_qkv_reference(
        x, q_dim=q_dim, v_dim=v_dim, head_k_dim=head_k, head_v_dim=head_v
    )
    assert query.shape == (1, n_rows, q_dim // head_k, head_k)
    assert key.shape == (1, n_rows, q_dim // head_k, head_k)
    assert value.shape == (1, n_rows, v_dim // head_v, head_v)
    for t in range(n_rows):
        for h in range(q_dim // head_k):
            head = slice(h * head_k, (h + 1) * head_k)
            assert torch.equal(query[0, t, h], x[t, head])
            assert torch.equal(
                key[0, t, h], x[t, q_dim + head.start : q_dim + head.stop]
            )
        for h in range(v_dim // head_v):
            head = slice(h * head_v, (h + 1) * head_v)
            assert torch.equal(
                value[0, t, h], x[t, 2 * q_dim + head.start : 2 * q_dim + head.stop]
            )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_k_dim_other_than_q_dim_matches_the_host_chain(dtype):
    n_rows, q_dim, k_dim, v_dim = 4, 256, 640, 512
    x = _packed(n_rows, q_dim, v_dim, dtype, k_dim=k_dim)
    plan, _, got = _unpacked(x, q_dim, v_dim, HEAD_K, HEAD_K, k_dim=k_dim)
    want = reference.rearrange_mixed_qkv_reference(
        x, q_dim=q_dim, v_dim=v_dim, head_k_dim=HEAD_K, head_v_dim=HEAD_K, k_dim=k_dim
    )
    assert plan.k_dim == k_dim
    assert got[1].shape == (1, n_rows, k_dim // HEAD_K, HEAD_K)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        assert g.stride() == w.stride()
        assert g.storage_offset() == w.storage_offset()
        assert torch.equal(g, w)


@pytest.mark.parametrize("pad", [1, 7, 129])
def test_wider_input_row_stride_is_read_with_its_own_stride(pad):
    n_rows, q_dim, v_dim = 4, 512, 1024
    x = _packed(n_rows, q_dim, v_dim, torch.bfloat16, pad=pad)
    assert x.stride(0) == 2 * q_dim + v_dim + pad
    plan, _, got = _unpacked(x, q_dim, v_dim)
    assert plan.row_stride == x.stride(0)
    want = reference.rearrange_mixed_qkv_reference(
        x, q_dim=q_dim, v_dim=v_dim, head_k_dim=HEAD_K, head_v_dim=HEAD_V
    )
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g, w)


# --------------------------------------------------------------- buffer layout
def test_views_span_one_flat_buffer_with_disjoint_slices():
    n_rows, q_dim, v_dim = 5, 512, 1024
    x = _packed(n_rows, q_dim, v_dim, torch.bfloat16)
    plan, flat, (query, key, value) = _unpacked(x, q_dim, v_dim)

    storage = query.untyped_storage()
    assert key.untyped_storage().data_ptr() == storage.data_ptr()
    assert value.untyped_storage().data_ptr() == storage.data_ptr()
    assert storage.data_ptr() == flat.untyped_storage().data_ptr()
    assert storage.nbytes() == n_rows * (2 * q_dim + v_dim) * x.element_size()

    assert _span(query) == (0, n_rows * q_dim * x.element_size())
    assert _span(key)[0] == _span(query)[1]
    assert _span(value)[0] == _span(key)[1]
    assert _span(value)[1] == storage.nbytes()

    for tensor in (query, key, value):
        assert tensor.is_contiguous()
        assert tensor.dtype == torch.bfloat16


def test_plan_flat_size_is_exactly_the_packed_row_times_rows():
    x = _packed(6, 384, 256, torch.float32)
    plan = _plan_of(x, 384, 256)
    assert plan.flat_elems == x.numel()
    assert plan.grid == (3, 6 * plan.n_col_tiles)


# ---------------------------------------------- the one allocating/launching path
class _Launch:
    def __init__(self, grid, args, kwargs, probes_at_launch):
        self.grid = grid
        self.args = args
        self.kwargs = kwargs
        self.probes_at_launch = probes_at_launch


class _Recorder:
    def __init__(self):
        self.calls = []
        self.probes = 0
        self.block = BLOCK_SMALL


@pytest.fixture
def kernel(monkeypatch):
    """Stub the device check and replace the kernel module with a recorder."""
    monkeypatch.setattr(api, "_require_npu", lambda _t: None)
    recorder = _Recorder()

    def _probe():
        recorder.probes += 1

    class _Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                recorder.calls.append(_Launch(grid, args, kwargs, recorder.probes))
                mixed, flat = args[0], args[1]
                _replay(mixed, flat, _plan_from_launch(args, kwargs))

            return launch

    fake = types.ModuleType("vllm_ascend_split_batch.gdn_qkv_unpack.kernels")
    fake.DEFAULT_BLOCK = recorder.block
    fake.init_device_properties_triton = _probe
    fake._unpack_mixed_qkv_kernel = _Kernel()
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    pkg = sys.modules["vllm_ascend_split_batch.gdn_qkv_unpack"]
    monkeypatch.setattr(pkg, "kernels", fake, raising=False)
    return recorder


def _launch_call(x, q_dim, v_dim, head_k=HEAD_K, head_v=HEAD_V, k_dim=None):
    return api.rearrange_mixed_qkv(
        x,
        q_dim=q_dim,
        v_dim=v_dim,
        head_k_dim=head_k,
        head_v_dim=head_v,
        k_dim=k_dim,
    )


def test_launch_grid_args_and_probe_order(kernel):
    n_rows, q_dim, v_dim = 17, 136, 280
    x = _packed(n_rows, q_dim, v_dim, torch.bfloat16)
    _launch_call(x, q_dim, v_dim, 8, 8)

    (call,) = kernel.calls
    n_col_tiles = -(-max(q_dim, q_dim, v_dim) // BLOCK_SMALL)
    assert call.grid == (3, n_rows * n_col_tiles)
    assert call.args[0] is x
    assert call.args[2:8] == (
        n_rows,
        q_dim,
        q_dim,
        v_dim,
        x.stride(0),
        n_col_tiles,
    )
    assert call.kwargs["BLOCK"] == BLOCK_SMALL
    assert call.probes_at_launch == 1
    assert kernel.probes == 1


@pytest.mark.parametrize("n_rows,q_dim,v_dim,head_k,head_v", _CASES)
def test_launch_end_to_end_against_the_host_chain(
    kernel, n_rows, q_dim, v_dim, head_k, head_v
):
    x = _packed(n_rows, q_dim, v_dim, torch.bfloat16)
    got = _launch_call(x, q_dim, v_dim, head_k, head_v)
    want = reference.rearrange_mixed_qkv_reference(
        x, q_dim=q_dim, v_dim=v_dim, head_k_dim=head_k, head_v_dim=head_v
    )
    assert len(kernel.calls) == 1
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        assert g.storage_offset() == w.storage_offset()
        assert torch.equal(g, w)


def test_each_call_allocates_its_own_buffer_and_leaves_the_input_alone(kernel):
    n_rows, q_dim, v_dim = 3, 512, 1024
    x = _packed(n_rows, q_dim, v_dim, torch.float32)
    before = x.clone()

    first = _launch_call(x, q_dim, v_dim)
    second = _launch_call(x, q_dim, v_dim)

    assert first[0].untyped_storage().data_ptr() != (
        second[0].untyped_storage().data_ptr()
    )
    assert torch.equal(x, before)
    assert len(kernel.calls) == 2


# ------------------------------------------------------------------ device guard
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_public_entry_refuses_non_npu_tensors(device):
    x = torch.zeros(2, 2 * 512 + 1024, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="NPU"):
        api.rearrange_mixed_qkv(
            x, q_dim=512, v_dim=1024, head_k_dim=HEAD_K, head_v_dim=HEAD_V
        )


def test_public_entry_refuses_non_tensors_before_the_device_check():
    with pytest.raises(TypeError, match="torch.Tensor"):
        api.rearrange_mixed_qkv(
            [], q_dim=512, v_dim=1024, head_k_dim=HEAD_K, head_v_dim=HEAD_V
        )


def test_none_propagates():
    assert api.rearrange_mixed_qkv(
        None, q_dim=512, v_dim=1024, head_k_dim=HEAD_K, head_v_dim=HEAD_V
    ) == (None, None, None)


def test_no_other_allocation_or_launch_entry_exists():
    assert not hasattr(api, "_unpack")


_DEVICE_PROBE = """
import sys
import torch
from vllm_ascend_split_batch.gdn_qkv_unpack import api

x = torch.zeros(2, 2 * 512 + 1024, dtype=torch.bfloat16)
try:
    api.rearrange_mixed_qkv(x, q_dim=512, v_dim=1024, head_k_dim=128, head_v_dim=128)
except ValueError as exc:
    assert "NPU" in str(exc), exc
else:
    raise AssertionError("a CPU tensor must be refused")

assert "vllm_ascend_split_batch.gdn_qkv_unpack.kernels" not in sys.modules
print("ok")
"""


def test_cpu_rejection_never_imports_the_kernel_module():
    result = subprocess.run(
        [sys.executable, "-c", _DEVICE_PROBE],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


# ------------------------------------------------------------------ rejections
_PACKED_512_1024 = 2 * 512 + 1024
_GEOM = {"q_dim": 512, "v_dim": 1024, "head_k_dim": HEAD_K, "head_v_dim": HEAD_V}


@pytest.mark.parametrize(
    "override,exc,match",
    [
        ({"head_k_dim": 96}, ValueError, "head_k_dim"),
        ({"head_v_dim": 96}, ValueError, "head_v_dim"),
        ({"q_dim": 0}, ValueError, "must be positive"),
        ({"v_dim": -8}, ValueError, "must be positive"),
        ({"head_k_dim": 0}, ValueError, "must be positive"),
        ({"q_dim": 512.0}, TypeError, "python int"),
        ({"q_dim": True}, TypeError, "python int"),
        ({"q_dim": None}, TypeError, "python int"),
    ],
)
def test_bad_dimension_arguments_are_rejected(override, exc, match):
    x = torch.zeros(2, _PACKED_512_1024, dtype=torch.bfloat16)
    with pytest.raises(exc, match=match):
        api._plan(x, block=BLOCK_SMALL, **{**_GEOM, **override})


@pytest.mark.parametrize(
    "k_dim,exc,match",
    [
        (0, ValueError, "must be positive"),
        (200, ValueError, "head_k_dim"),
        (3.0, TypeError, "python int"),
    ],
)
def test_k_dim_is_validated(k_dim, exc, match):
    width = 512 + (k_dim if isinstance(k_dim, int) else 512) + 1024
    x = torch.zeros(2, width, dtype=torch.bfloat16)
    with pytest.raises(exc, match=match):
        api._plan(x, k_dim=k_dim, block=BLOCK_SMALL, **_GEOM)


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float64, torch.uint8]
)
def test_dtype_not_supported_is_rejected(dtype):
    x = torch.zeros(2, _PACKED_512_1024, dtype=dtype)
    with pytest.raises(TypeError, match="bfloat16/float16/float32"):
        _plan_of(x, 512, 1024)


@pytest.mark.parametrize("rows", [(1536,), (1, 2, 512, 1024)])
def test_rank_other_than_two_is_rejected(rows):
    x = torch.zeros(*rows, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="2D"):
        _plan_of(x, 512, 1024)


@pytest.mark.parametrize("delta", [-1, 1])
def test_packed_last_dim_mismatch_is_rejected(delta):
    x = torch.zeros(2, _PACKED_512_1024 + delta, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="last dim"):
        _plan_of(x, 512, 1024)


def test_empty_seq_len_is_rejected_like_the_host_chain():
    x = torch.zeros(0, _PACKED_512_1024, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="seq_len == 0"):
        _plan_of(x, 512, 1024)
    with pytest.raises(RuntimeError):
        reference.rearrange_mixed_qkv_reference(
            x, q_dim=512, v_dim=1024, head_k_dim=HEAD_K, head_v_dim=HEAD_V
        )


def test_transposed_input_is_rejected():
    x = torch.zeros(_PACKED_512_1024, 2, dtype=torch.bfloat16).t()
    assert x.shape == (2, _PACKED_512_1024)
    assert x.stride(-1) == 2
    with pytest.raises(ValueError, match="last-dim stride"):
        _plan_of(x, 512, 1024)


def test_row_stride_shorter_than_the_packed_row_is_rejected():
    x = torch.as_strided(
        torch.zeros(2 * _PACKED_512_1024, dtype=torch.bfloat16),
        (2, _PACKED_512_1024),
        (_PACKED_512_1024 - 1, 1),
    )
    with pytest.raises(ValueError, match="row stride"):
        _plan_of(x, 512, 1024)


def test_fp32_mask_limit_is_rejected():
    too_wide = torch.zeros(1, 2 * 2**24 + 8, dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="fp32 column mask"):
        _plan_of(too_wide, 2**24, 8, HEAD_K, 8)


def test_output_span_over_int32_is_rejected():
    too_big = torch.zeros(2**21, _PACKED_512_1024, dtype=torch.bfloat16, device="meta")
    assert too_big.numel() >= 2**31
    with pytest.raises(ValueError, match="overflows int32"):
        _plan_of(too_big, 512, 1024)


def test_input_row_span_over_int32_is_rejected():
    """The flat output can fit while ``(T-1)*row_stride + width`` does not."""
    q_dim, n_rows = 128, 1000
    packed = 3 * q_dim
    stride0 = -(-(2**31 - packed) // (n_rows - 1))
    x = torch.empty_strided(
        (n_rows, packed), (stride0, 1), dtype=torch.bfloat16, device="meta"
    )
    assert n_rows * packed < 2**31 <= (n_rows - 1) * stride0 + packed
    with pytest.raises(ValueError, match="overflows int32"):
        _plan_of(x, q_dim, q_dim)


def test_grid_over_coredim_is_rejected_not_promised():
    q_dim, v_dim = 136, 280
    n_col_tiles = -(-max(q_dim, v_dim) // BLOCK_SMALL)
    over = 65535 // (3 * n_col_tiles) + 1
    x = torch.zeros(over, 2 * q_dim + v_dim, dtype=torch.bfloat16, device="meta")
    with pytest.raises(ValueError, match="coreDim"):
        _plan_of(x, q_dim, v_dim, 8, 8)

    at_cap = torch.zeros(
        over - 1, 2 * q_dim + v_dim, dtype=torch.bfloat16, device="meta"
    )
    plan = _plan_of(at_cap, q_dim, v_dim, 8, 8)
    assert plan.grid == (3, (over - 1) * n_col_tiles)
    assert plan.grid[0] * plan.grid[1] <= 65535


# ------------------------------------------------------------ import behaviour
_LAZY_PROBE = """
import sys
import vllm_ascend_split_batch.gdn_qkv_unpack as pkg

assert "torch" not in sys.modules
assert "triton" not in sys.modules
assert "torch_npu" not in sys.modules

fn = pkg.rearrange_mixed_qkv
ref = pkg.rearrange_mixed_qkv_reference
assert callable(fn) and callable(ref)
assert "vllm_ascend_split_batch.gdn_qkv_unpack.kernels" not in sys.modules
assert "kernels" not in vars(sys.modules[fn.__module__])
print("ok")
"""


def test_package_import_is_inert_and_the_kernel_stays_lazy():
    result = subprocess.run(
        [sys.executable, "-c", _LAZY_PROBE],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


def test_api_and_reference_do_not_import_the_kernel_module():
    assert "triton" not in vars(api)
    assert "triton" not in vars(reference)
    assert "kernels" not in vars(api)


def test_no_test_in_this_module_reached_the_real_kernel_module():
    """Runs last by definition order: the stub keeps the kernel module unloaded."""
    assert "vllm_ascend_split_batch.gdn_qkv_unpack.kernels" not in sys.modules
