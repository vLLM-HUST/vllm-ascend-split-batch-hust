# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Cascade two-stage decode plugin (default-off).

This is the plugin-side port of the HUST fork cascade decode integration.  It
monkeypatches the official vllm/vllm-ascend at ``vllm.general_plugins`` load
time, so the host source trees stay untouched.

Provenance:
  - vllm-ascend-hust ``plan811/lo-dual-stream-0826`` @ ``1407d2a12``
  - vllm-hust ``plan811/pr-full-graph-parallel-0826`` @ ``09906fd22d``
The cascade methods below are adapted from ``vllm_ascend/attention/attention_v1.py``
(``_forward_cascade_decode`` / ``_forward_cascade_decode_inner``,
``use_cascade_attention``) and ``vllm/forward_context.py``
(``BatchDescriptor.cascade``).  The original code is Copyright (c) 2025 Huawei
Technologies Co., Ltd. and is licensed under the CANN Open Software License
Agreement Version 2.0 / Apache-2.0.

Default mode: nothing changes unless ``VLLM_ASCEND_ENABLE_CASCADE_DECODE=1``.
"""

import contextlib
import importlib
import logging
import math
import os
import sys
import types

import torch
import torch_npu  # noqa: F401
import vllm_ascend.envs as envs_mod

logger = logging.getLogger(__name__)

# Filled by load().
attn_mod = None
_EXTRA_CTX = None
_HAS_LSE_MERGE_OP = False
_HAS_FA_FP32_STAGE1_OP = False
_cascade_warning_once = False


def _cascade_stage2_kv_lens(seq_lens_list, num_tokens, shared_len):
    """Per-request suffix (private KV) lengths for cascade stage 2."""
    lens = []
    for seq_len in seq_lens_list[:num_tokens]:
        n = seq_len - shared_len
        lens.append(n if n > 0 else 1)
    return lens


def _cascade_has_short_real_request(seq_lens_list, num_tokens, shared_len):
    """True when a real request reached the shared prefix (must fail open)."""
    return any(
        seq_len > 1 and seq_len - shared_len <= 0
        for seq_len in seq_lens_list[:num_tokens]
    )


def _use_cascade_attention(
    self,
    common_prefix_len,
    query_lens,
    num_query_heads,
    num_kv_heads,
    use_alibi,
    use_sliding_window,
    use_local_attention,
    num_sms,
    dcp_world_size,
):
    """Cascade decode gate (see vllm_ascend/envs.py for thresholds)."""
    if envs_mod.VLLM_ASCEND_CASCADE_STRICT:
        if envs_mod.VLLM_ASCEND_ENABLE_CASCADE_DECODE:
            logger.warning(
                "VLLM_ASCEND_CASCADE_STRICT=1 overrides "
                "VLLM_ASCEND_ENABLE_CASCADE_DECODE=1: running the standard "
                "full-KV FIA path for bit-exact reproducibility."
            )
        return False
    if not envs_mod.VLLM_ASCEND_ENABLE_CASCADE_DECODE:
        return False
    precision = envs_mod.VLLM_ASCEND_CASCADE_PRECISION
    if precision not in ("bf16", "fp32"):
        logger.error(
            "Invalid VLLM_ASCEND_CASCADE_PRECISION=%r (expected 'bf16' or "
            "'fp32'); disabling the two-stage cascade decode path.",
            precision,
        )
        return False
    if precision == "fp32" and not _HAS_FA_FP32_STAGE1_OP:
        logger.warning(
            "VLLM_ASCEND_CASCADE_PRECISION=fp32 requires the custom "
            "fa_fp32_stage1 op (ascend_kernel wheel); falling back to the "
            "Tier-0 bf16 two-stage path."
        )
    if use_alibi or use_sliding_window or use_local_attention or dcp_world_size > 1:
        return False
    if common_prefix_len < envs_mod.VLLM_ASCEND_CASCADE_MIN_PREFIX:
        return False
    if len(query_lens) < envs_mod.VLLM_ASCEND_CASCADE_MIN_REQS:
        return False
    global _cascade_warning_once
    if not _cascade_warning_once:
        logger.warning(
            "Two-stage cascade decode is active for this batch: its output has "
            "a bf16-level difference versus the standard full-KV FIA path and "
            "is not bit-exact reproducible (see VLLM_ASCEND_CASCADE_STRICT)."
        )
        _cascade_warning_once = True
    return True


def _make_build_wrapper(orig_build):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = orig_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        # The official AscendMetadata dataclass has no cascade_shared_len field;
        # the attribute is added dynamically and is a no-op when cascade is off.
        with contextlib.suppress(AttributeError):
            metadata.cascade_shared_len = common_prefix_len or 0
        return metadata

    return build


def _cascade_stage1_fp32_supported(self, embed, block_size):
    """Tier-1 stage-1 op constraints (see ascend-kernel design.md)."""
    if not _HAS_FA_FP32_STAGE1_OP:
        return False
    if embed != 128 or block_size != 128:
        return False
    return math.isclose(self.scale, 1.0 / math.sqrt(embed), rel_tol=1e-9)


def _forward_cascade_decode(
    self, query, key, value, attn_metadata, output, kv_cache=None
):
    """Two-stage cascade decode; returns False to fail open to the FIA path."""
    try:
        return self._forward_cascade_decode_inner(
            query, key, value, attn_metadata, output, kv_cache
        )
    except Exception:
        logger.exception("cascade decode failed, failing open to the standard path")
        return False


def _forward_cascade_decode_inner(
    self, query, key, value, attn_metadata, output, kv_cache=None
):
    key, value, block_size, block_table, _ = self._get_fia_params(
        key, value, attn_metadata, kv_cache
    )
    shared_len = attn_metadata.cascade_shared_len
    actual_q = attn_metadata.actual_seq_lengths_q
    num_tokens = len(actual_q)
    seq_lens_list = attn_metadata.seq_lens_list
    if (
        block_table is None
        or shared_len < block_size
        or len(seq_lens_list) < num_tokens
    ):
        return False
    shared_blocks = shared_len // block_size
    indep_kv_lens = _cascade_stage2_kv_lens(seq_lens_list, num_tokens, shared_len)
    if _cascade_has_short_real_request(seq_lens_list, num_tokens, shared_len):
        return False
    bt_shared = block_table[:1, :shared_blocks].contiguous()
    bt_rest = block_table[:, shared_blocks:].contiguous()

    fp32_stage1 = (
        envs_mod.VLLM_ASCEND_CASCADE_PRECISION == "fp32"
        and self._cascade_stage1_fp32_supported(self.head_size, block_size)
    )
    if fp32_stage1:
        dev = query.device
        stage1_q_seqlens = torch.full((1,), num_tokens, dtype=torch.int64, device=dev)
        stage1_kv_seqlens = torch.full(
            (1,), shared_len, dtype=torch.int64, device=dev
        )
        o1_fp32, l1_fp32 = torch.ops.npu.fa_fp32_stage1(
            query,
            self.key_cache,
            self.value_cache,
            bt_shared,
            stage1_q_seqlens,
            stage1_kv_seqlens,
            num_tokens,
        )
        o1 = o1_fp32
        l1 = l1_fp32.reshape(-1)
    else:
        out_pre, lse_pre = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            block_table=bt_shared,
            block_size=block_size,
            actual_seq_lengths=[num_tokens],
            actual_seq_lengths_kv=[shared_len],
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            scale=self.scale,
            softmax_lse_flag=True,
        )
        o1 = out_pre.reshape(num_tokens, self.num_heads, self.head_size)
        l1 = lse_pre.reshape(num_tokens, self.num_heads).float()

    out_suf, lse_suf = torch_npu.npu_fused_infer_attention_score(
        query,
        key,
        value,
        block_table=bt_rest,
        block_size=block_size,
        actual_seq_lengths=actual_q,
        actual_seq_lengths_kv=indep_kv_lens,
        num_heads=self.num_heads,
        num_key_value_heads=self.num_kv_heads,
        input_layout="TND",
        scale=self.scale,
        softmax_lse_flag=True,
    )

    o2 = out_suf.reshape(num_tokens, self.num_heads, self.head_size)
    l2 = lse_suf.reshape(num_tokens, self.num_heads).float()
    merged = None
    if _HAS_LSE_MERGE_OP:
        try:
            merged = torch.ops.npu.lse_merge(
                o1, o2, l1, l2, 1 if fp32_stage1 else 0
            )
        except Exception:
            if not getattr(self, "_lse_merge_fallback_logged", False):
                logger.exception(
                    "lse_merge kernel failed, falling back to torch merge"
                )
                self._lse_merge_fallback_logged = True
            merged = None
    if merged is None:
        m = torch.maximum(l1, l2)
        w_pre = torch.exp(l1 - m)
        w_suf = torch.exp(l2 - m)
        merged = (
            (o1.float() * w_pre.unsqueeze(-1) + o2.float() * w_suf.unsqueeze(-1))
            / (w_pre + w_suf).unsqueeze(-1)
        ).to(o1.dtype)
    output[:num_tokens] = merged.reshape(num_tokens, self.num_heads, self.head_size)
    if not getattr(self, "_cascade_active_logged", False):
        print(
            f"[cascade-active] shared_len={shared_len} "
            f"shared_blocks={shared_blocks} num_tokens={num_tokens} "
            f"heads={self.num_heads} head_size={self.head_size}",
            flush=True,
        )
        self._cascade_active_logged = True
    return True


def _make_forward_wrapper(orig_forward_fused_infer_attention):
    def forward_fused_infer_attention(
        self, query, key, value, attn_metadata, output, kv_cache=None
    ):
        capturing = getattr(_EXTRA_CTX, "capturing", False)
        if (
            not capturing
            and envs_mod.VLLM_ASCEND_ENABLE_CASCADE_DECODE
            and attn_metadata.attn_state == attn_mod.AscendAttentionState.DecodeOnly
            and getattr(attn_metadata, "cascade_shared_len", 0) > 0
            and not getattr(self, "enable_hamming_sparse", False)
            and getattr(self, "sliding_window", None) is None
            and self._forward_cascade_decode(
                query, key, value, attn_metadata, output, kv_cache
            )
        ):
            return output
        return orig_forward_fused_infer_attention(
            self, query, key, value, attn_metadata, output, kv_cache
        )

    return forward_fused_infer_attention


def _inject_env_vars():
    envs_mod.env_variables.update(
        {
            "VLLM_ASCEND_ENABLE_CASCADE_DECODE": lambda: bool(
                int(os.getenv("VLLM_ASCEND_ENABLE_CASCADE_DECODE", "0"))
            ),
            "VLLM_ASCEND_CASCADE_MIN_PREFIX": lambda: int(
                os.getenv("VLLM_ASCEND_CASCADE_MIN_PREFIX", "8192")
            ),
            "VLLM_ASCEND_CASCADE_MIN_REQS": lambda: int(
                os.getenv("VLLM_ASCEND_CASCADE_MIN_REQS", "32")
            ),
            "VLLM_ASCEND_CASCADE_STRICT": lambda: bool(
                int(os.getenv("VLLM_ASCEND_CASCADE_STRICT", "0"))
            ),
            "VLLM_ASCEND_CASCADE_PRECISION": lambda: os.getenv(
                "VLLM_ASCEND_CASCADE_PRECISION", "bf16"
            ),
        }
    )


def _install_policy_factory_stub():
    """Avoid importing numba at vllm-ascend module import time.

    The upstream ``policy_factory`` imports ``policy_flashlb`` (and therefore
    ``numba``) at module load.  The HUST fork made this import lazy; we do the
    same via a synthetic module so the default (non-FlashLB) serving path does
    not require a numba/numpy-compatible environment.
    """
    mod_name = "vllm_ascend.eplb.core.policy.policy_factory"
    if mod_name in sys.modules:
        # Already importable (e.g. a compatible numba was present); leave it.
        return
    try:
        from vllm_ascend.eplb.core.policy.policy_default_eplb import DefaultEplb
        from vllm_ascend.eplb.core.policy.policy_random import RandomLoadBalance
        from vllm_ascend.eplb.core.policy.policy_swift_balancer import SwiftBalanceEplb
    except Exception:
        # If even the non-FlashLB policy modules cannot be imported, do not
        # mask the original failure; leave the import chain untouched.
        return

    class PolicyFactory:
        @staticmethod
        def generate_policy(policy_type):
            if policy_type == 3:
                from vllm_ascend.eplb.core.policy.policy_flashlb import FlashLB, warm_up

                warm_up()
                logger.info(
                    "[eplb/policy] Policy: %s (type=%s)", FlashLB.__name__, policy_type
                )
                return FlashLB()
            policy = {
                0: RandomLoadBalance,
                1: DefaultEplb,
                2: SwiftBalanceEplb,
            }
            policy_class = policy.get(policy_type)
            if policy_class is None:
                policy_class = RandomLoadBalance
                logger.warning(
                    "[eplb/policy] Unrecognized policy_type=%s, "
                    "falling back to %s",
                    policy_type,
                    policy_class.__name__,
                )
            return policy_class()

    stub = types.ModuleType(mod_name)
    stub.PolicyFactory = PolicyFactory
    sys.modules[mod_name] = stub


def _install_spec_decode_stub():
    """Make ``vllm_ascend.spec_decode`` import lazy (HUST parity).

    The upstream module imports ``AscendNgramProposer`` at module import time,
    which pulls in ``vllm.v1.spec_decode.ngram_proposer`` and therefore numba.
    The HUST fork moved these imports inside ``get_spec_decode_method``; we
    install a lightweight stand-in so the default (no spec decode) serving path
    never imports numba.
    """
    mod_name = "vllm_ascend.spec_decode"
    if mod_name in sys.modules:
        return

    def get_spec_decode_method(method, vllm_config, device, runner):
        if method == "ngram":
            from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer

            return AscendNgramProposer(vllm_config, runner)
        if method == "ngram_gpu":
            from vllm_ascend.spec_decode.ngram_proposer_npu import (
                AscendNgramProposerNPU,
            )

            return AscendNgramProposerNPU(vllm_config, device, runner)
        if method == "suffix":
            from vllm_ascend.spec_decode.suffix_proposer import (
                AscendSuffixDecodingProposer,
            )

            return AscendSuffixDecodingProposer(vllm_config, runner)
        if method == "medusa":
            from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer

            return AscendMedusaProposer(vllm_config, device)
        if method in ("eagle", "eagle3", "mtp"):
            from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
            from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer

            speculative_config = vllm_config.speculative_config
            if speculative_config is not None and speculative_config.use_step3p5_mtp():
                return AscendStep3p5MTPProposer(vllm_config, device, runner)
            return AscendEagleProposer(vllm_config, device, runner)
        if method == "dflash":
            from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

            return AscendDflashProposer(vllm_config, device, runner)
        if method == "draft_model":
            from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer

            return AscendDraftModelProposer(vllm_config, device, runner)
        if method == "extract_hidden_states":
            from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
                AscendExtractHiddenStatesProposer,
            )

            return AscendExtractHiddenStatesProposer(vllm_config, device, runner)
        raise ValueError(f"Unknown speculative decoding method: {method}")

    stub = types.ModuleType(mod_name)
    stub.__package__ = mod_name
    stub.get_spec_decode_method = get_spec_decode_method
    # Keep it a package so direct submodule imports (e.g.
    # ``vllm_ascend.spec_decode.dflash_proposer``) still resolve to the real
    # vllm-ascend files.
    import vllm_ascend

    stub.__path__ = [os.path.join(os.path.dirname(vllm_ascend.__file__), "spec_decode")]
    sys.modules[mod_name] = stub


def _install_ngram_proposer_stub():
    """Avoid importing numba via ``vllm_ascend.spec_decode.ngram_proposer``.

    The upstream ``model_runner_v1`` imports ``AscendNgramProposer`` from this
    module at module load time; the module itself imports numba.  For the
    default (no speculative decoding) serving path we install a lightweight
    placeholder so the worker import chain succeeds.  If ngram speculative
    decoding is actually requested, the placeholder raises and the real
    numba-based path remains a separate (non-default) configuration.
    """
    mod_name = "vllm_ascend.spec_decode.ngram_proposer"
    if mod_name in sys.modules:
        return

    class AscendNgramProposer:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "ngram speculative decoding requires a numba-compatible "
                "environment; use ngram_gpu instead"
            )

    stub = types.ModuleType(mod_name)
    stub.AscendNgramProposer = AscendNgramProposer
    sys.modules[mod_name] = stub


def load():
    """vllm.general_plugins entry point.

    Idempotent.  The runtime cascade gate stays closed unless the caller sets
    ``VLLM_ASCEND_ENABLE_CASCADE_DECODE=1``, so the patched methods are no-ops
    in the default (off) configuration.
    """
    global attn_mod, _EXTRA_CTX, _HAS_LSE_MERGE_OP, _HAS_FA_FP32_STAGE1_OP

    try:
        import ascend_kernel  # noqa: F401  registers torch.ops.npu.*

        _HAS_LSE_MERGE_OP = hasattr(torch.ops.npu, "lse_merge")
        _HAS_FA_FP32_STAGE1_OP = hasattr(torch.ops.npu, "fa_fp32_stage1")
    except Exception:
        _HAS_LSE_MERGE_OP = False
        _HAS_FA_FP32_STAGE1_OP = False

    _inject_env_vars()
    _install_policy_factory_stub()
    _install_spec_decode_stub()
    _install_ngram_proposer_stub()

    # Resolve the upstream device_op <-> ops circular import by fully importing
    # vllm_ascend.ops before attention_v1 (mirrors the normal vllm-ascend lazy
    # import order), then patch the attention modules.
    importlib.import_module("vllm_ascend.ops")
    attn_mod = importlib.import_module("vllm_ascend.attention.attention_v1")

    # Published on the official module as well so any HUST-derived code or
    # probe that imports the module globals sees the same flags.
    attn_mod._HAS_LSE_MERGE_OP = _HAS_LSE_MERGE_OP
    attn_mod._HAS_FA_FP32_STAGE1_OP = _HAS_FA_FP32_STAGE1_OP
    _EXTRA_CTX = getattr(attn_mod, "_EXTRA_CTX", None)

    builder_cls = attn_mod.AscendAttentionMetadataBuilder
    if not getattr(builder_cls, "_cascade_plugin_patched", False):
        builder_cls.use_cascade_attention = _use_cascade_attention
        builder_cls.build = _make_build_wrapper(builder_cls.build)
        builder_cls._cascade_plugin_patched = True

    impl_cls = attn_mod.AscendAttentionBackendImpl
    if not getattr(impl_cls, "_cascade_plugin_patched", False):
        impl_cls._cascade_stage1_fp32_supported = _cascade_stage1_fp32_supported
        impl_cls._forward_cascade_decode = _forward_cascade_decode
        impl_cls._forward_cascade_decode_inner = _forward_cascade_decode_inner
        impl_cls.forward_fused_infer_attention = _make_forward_wrapper(
            impl_cls.forward_fused_infer_attention
        )
        impl_cls._cascade_plugin_patched = True
