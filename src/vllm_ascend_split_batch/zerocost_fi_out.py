# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Zero-cost wiring ①: the prefill FIA writes into the caller's output buffer.

The decode FULL-graph leg always called the ``.out`` overload of
``torch_npu.npu_fused_infer_attention_score``; the eager prefill leg calls the
functional form and then copies the result into ``output``.  This module makes
the eager leg do the same thing without the copy:

* the resolved device adaptor's FIA entry point gains a private ``_zc_fi_out=``
  kwarg (the destination region) and routes it to the ``out`` overload, falling
  back to the host call whenever the out path is not provably safe;
* both ``output[...] = attn_output[...]`` statements in the host attention class
  are guarded by an exact "src and dst are the same bytes" predicate, so a copy
  that cannot change memory is skipped and every other copy still happens.

Load-bearing rule: every path fails open.  A refused rewrite raises out of
:func:`install_fi_out` and the caller rolls the whole feature back; a runtime
surprise inside the wrapper falls back to the host call.
"""

from __future__ import annotations

import ast

from .zerocost_state import (
    ADAPTOR_ATTR,
    ATTENTION_CLASS,
    ATTENTION_MODULE,
    FI_OUT_KWARG,
    FIA_SYMBOL,
    IDENTITY_HELPER,
    LARGE_HEAD_FALLBACK,
    METHOD_FIA,
    METHOD_FORWARD,
    _import_host,
    _info_once,
    _orig,
    _rewrite_method,
    _stats,
    _tally_state,
    _tally_tokens,
    _warn_once,
)

#: marker set on the installed wrapper, so a repeated install cannot wrap the
#: wrapper it installed itself.
FIA_MARKER = "_zc_fi_out_wrapper"


def _identity_copy(attn_output, output, num_tokens) -> bool:
    """True when ``output[:n] = attn_output[:n]`` cannot change any byte.

    Exact, dtype/stride/address-level comparison of the two regions -- the only
    case in which skipping the copy is provably a no-op.  Never raises: any
    surprise means "not identity", i.e. the copy runs as it always did.
    """
    try:
        if attn_output is output:
            return True
        count = int(num_tokens)
        if count <= 0:
            return True
        src = attn_output[:count]
        dst = output[:count]
        return bool(
            src.shape == dst.shape
            and src.stride() == dst.stride()
            and src.data_ptr() == dst.data_ptr()
            and src.dtype == dst.dtype
            and src.device == dst.device
        )
    except Exception:  # noqa: BLE001 -- fail-open: keep the copy
        return False


def _is_tail_copy(node: ast.stmt) -> bool:
    """``output[:num_tokens] = attn_output[:num_tokens]`` (the guarded seam)."""
    if not isinstance(node, ast.Assign) or len(node.targets) != 1:
        return False
    target = node.targets[0]
    if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)):
        return False
    if target.value.id != "output":
        return False
    value = node.value
    if not (isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name)):
        return False
    return value.value.id == "attn_output"


def _guard_tail_copy(fn: ast.FunctionDef) -> None:
    """Wrap the trailing ``output[...] = attn_output[...]`` in the identity guard."""
    hits = [i for i, stmt in enumerate(fn.body) if _is_tail_copy(stmt)]
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one trailing `output[...] = attn_output[...]` "
            f"assignment in {fn.name}, found {len(hits)}"
        )
    stmt = fn.body[hits[0]]
    guard = ast.If(
        test=ast.UnaryOp(
            op=ast.Not(),
            operand=ast.Call(
                func=ast.Name(id=IDENTITY_HELPER, ctx=ast.Load()),
                args=[
                    ast.Name(id="attn_output", ctx=ast.Load()),
                    ast.Name(id="output", ctx=ast.Load()),
                    ast.Name(id="num_tokens", ctx=ast.Load()),
                ],
                keywords=[],
            ),
        ),
        body=[stmt],
        orelse=[],
    )
    ast.copy_location(guard, stmt)
    fn.body[hits[0]] = guard


def _inject_fi_out(fn: ast.FunctionDef) -> None:
    """Hand the destination region to the adaptor FIA call."""
    hits = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == FIA_SYMBOL
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == ADAPTOR_ATTR
        ):
            hits.append(node)
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly one {ADAPTOR_ATTR}.{FIA_SYMBOL} call in {fn.name}, "
            f"found {len(hits)}"
        )
    call = hits[0]
    if any(kw.arg == FI_OUT_KWARG for kw in call.keywords):
        return
    call.keywords.append(
        ast.keyword(
            arg=FI_OUT_KWARG,
            value=ast.Subscript(
                value=ast.Name(id="output", ctx=ast.Load()),
                slice=ast.Slice(upper=ast.Name(id="num_tokens", ctx=ast.Load())),
                ctx=ast.Load(),
            ),
        )
    )


def _transform_fia_method(fn: ast.FunctionDef) -> None:
    """``forward_fused_infer_attention``: hand the destination over + guard the copy."""
    _inject_fi_out(fn)
    _guard_tail_copy(fn)


def _fi_out_call(orig, query, key, value, attn_metadata, args, kwargs, dest):
    """Route one FIA call to the ``.out`` overload, writing into ``dest``.

    Falls back to the host's functional call (whose result the caller's identity
    guard then copies into ``dest`` as before) whenever the out path is not
    provably safe.  ``orig`` is the host's *bound* classmethod (cls already
    bound), so it is called without the class argument.
    """
    import torch  # noqa: PLC0415 -- only reachable with the feature on
    import torch_npu  # noqa: PLC0415

    call = dict(kwargs)
    if call.get("head_size") == LARGE_HEAD_FALLBACK:
        # The host opts into its large-head prefill kernel there and allocates
        # its own output; the caller's guard copies it back.
        _stats["fi_out_head_fallback"] += 1
        return orig(query, key, value, attn_metadata, *args, **kwargs)

    num_key_value_heads = call.pop("num_key_value_heads")
    num_heads = call.pop("num_heads")
    scale = call.pop("scale")
    for consumed in (
        "key_cache",
        "value_cache",
        "current_key",
        "current_value",
        "head_size",
        "is_prefill_no_cache",
    ):
        call.pop(consumed, None)

    lse = torch.empty(1, dtype=query.dtype, device=query.device)
    result = torch_npu.npu_fused_infer_attention_score.out(
        query=query,
        key=key.contiguous(),
        value=value.contiguous(),
        num_key_value_heads=num_key_value_heads,
        num_heads=num_heads,
        scale=scale,
        out=[dest, lse],
        **call,
    )
    first = result[0] if isinstance(result, (tuple, list)) and result else result
    if first is not dest and (
        first.data_ptr() != dest.data_ptr() or tuple(first.shape) != tuple(dest.shape)
    ):
        # The op did not land in the caller's region (e.g. it re-allocated):
        # keep native semantics by writing the result back explicitly.
        dest.copy_(first)
        _stats["fi_out_writeback"] += 1
    else:
        _stats["fi_out_applied"] += 1
        _info_once(
            "zero-cost wiring ①: prefill FIA now writes into the caller's "
            "output buffer through the .out overload (no TensorMove copy)."
        )
    return dest, lse


def _make_fia_wrapper(orig):
    """Signature-agnostic wrapper for the adaptor's FIA entry point."""

    def _npu_fused_infer_attention_score(
        cls, query, key, value, attn_metadata, *args, **kwargs
    ):
        dest = kwargs.pop(FI_OUT_KWARG, None)
        if dest is None:
            return orig(query, key, value, attn_metadata, *args, **kwargs)
        _stats["fi_out_calls"] += 1
        _tally_tokens(query)
        _tally_state(attn_metadata)
        try:
            return _fi_out_call(
                orig, query, key, value, attn_metadata, args, kwargs, dest
            )
        except Exception as exc:  # noqa: BLE001 -- never break the host call
            _stats["fi_out_fail_open"] += 1
            _warn_once(
                "zero-cost wiring ①: FIA .out forwarding unavailable (%s: %s); "
                "this process keeps the host's functional FIA call plus its "
                "copy (fail-open).",
                type(exc).__name__,
                exc,
            )
            return orig(query, key, value, attn_metadata, *args, **kwargs)

    return _npu_fused_infer_attention_score


def _identity_helper(attn_output, output, num_tokens) -> bool:  # pragma: no cover
    """Host-module shim: counts then defers to :func:`_identity_copy`."""
    same = _identity_copy(attn_output, output, num_tokens)
    if same:
        _stats["selfcopy_identity"] += 1
    else:
        _stats["selfcopy_copied"] += 1
    return same


def _patch_adaptor(adaptor) -> None:
    """Point the adaptor's FIA entry at the routing wrapper, recording rollback.

    The record is taken **before** the ``setattr``: reading the attribute
    afterwards would store the wrapper itself as "the original" and
    :func:`vllm_ascend_split_batch.zerocost_wiring.uninstall` would re-install
    the wrapper instead of restoring the host method.

    Two cases are told apart, because the rollback differs:

    * ``FIA_SYMBOL`` is in ``adaptor.__dict__``: the class owns the descriptor,
      so the record carries that descriptor and rollback writes it back;
    * ``FIA_SYMBOL`` is inherited from a base class: the patch *adds* a class
      attribute, so the record is ``None`` and rollback deletes the override,
      letting inheritance resolve the host method again.

    Idempotent: a class that already carries this wrapper is not wrapped twice
    (a second wrap would record the first wrapper as the original).
    """
    current = getattr(adaptor, FIA_SYMBOL, None)
    if not callable(current):
        raise RuntimeError(
            f"{ADAPTOR_ATTR}.{FIA_SYMBOL} is missing on host adaptor "
            f"{adaptor.__name__!r}"
        )
    if getattr(current, FIA_MARKER, False):
        return
    owned = adaptor.__dict__.get(FIA_SYMBOL)
    wrapper = _make_fia_wrapper(current)
    setattr(wrapper, FIA_MARKER, True)
    setattr(adaptor, FIA_SYMBOL, classmethod(wrapper))
    _orig[f"method:{adaptor.__name__}.{FIA_SYMBOL}"] = (adaptor, FIA_SYMBOL, owned)


def install_fi_out() -> None:
    """Install ①. Raises on a moved anchor; the caller rolls the feature back."""
    module = _import_host(ATTENTION_MODULE)
    cls = getattr(module, ATTENTION_CLASS)
    adaptor = getattr(module, ADAPTOR_ATTR)
    _patch_adaptor(adaptor)

    # the rewritten body calls this helper; inject it into the host module so
    # the ported source stays byte-equal to the host body plus the guard.
    if getattr(module, IDENTITY_HELPER, None) is None:
        module.__dict__[IDENTITY_HELPER] = _identity_helper
        _orig[f"global:{ATTENTION_MODULE}.{IDENTITY_HELPER}"] = (
            module,
            IDENTITY_HELPER,
            None,
        )

    _rewrite_method(module, cls, METHOD_FIA, _transform_fia_method)
    _rewrite_method(module, cls, METHOD_FORWARD, _guard_tail_copy)
