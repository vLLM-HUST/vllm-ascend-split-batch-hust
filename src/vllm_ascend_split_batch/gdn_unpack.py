# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

ENABLED = os.environ.get("VLLM_HUST_GDN_UNPACK_VIEWS", "1") == "1"


def configure(layer):
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    layer._hust_unpack_full_graph = (
        config is not None and str(config.compilation_config.cudagraph_mode) == "FULL"
    )


def eligible(layer, mixed, metadata):
    model = getattr(layer, "model_config", None)
    config = getattr(model, "hf_text_config", None)
    return (
        ENABLED
        and getattr(layer, "_hust_unpack_full_graph", False)
        and mixed is not None
        and metadata.num_prefills == 0
        and getattr(layer, "speculative_config", None) is None
        and getattr(layer, "quant_config", None) is None
        and getattr(model, "model", "").rstrip("/").split("/")[-1] == "Qwen3.5-35B-A3B"
        and getattr(config, "model_type", None) == "qwen3_5_moe_text"
        and getattr(config, "hidden_size", None) == 2048
        and getattr(config, "num_hidden_layers", None) == 40
        and getattr(config, "num_experts", None) == 256
        and layer.tp_size == 2
        and layer.key_dim == 2048
        and layer.value_dim == 4096
        and layer.head_k_dim == 128
        and layer.head_v_dim == 128
        and mixed.dtype == torch.bfloat16
        and mixed.device.type == "npu"
        and mixed.ndim == 2
        and mixed.shape[0] in (1, 16)
        and mixed.shape[1] == 4096
    )


def unpack(mixed, layer):
    qd = layer.key_dim // layer.tp_size
    s = mixed.shape[0]
    q = mixed[:, :qd].contiguous().view(1, s, -1, layer.head_k_dim)
    k = mixed[:, qd : 2 * qd].contiguous().view(1, s, -1, layer.head_k_dim)
    v = mixed[:, 2 * qd :].contiguous().view(1, s, -1, layer.head_v_dim)
    return q, k, v
