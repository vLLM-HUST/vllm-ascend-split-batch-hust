from types import SimpleNamespace

from vllm_ascend_split_batch.gdn_unpack import unpack as helper

layer = SimpleNamespace(
    key_dim=2048, value_dim=4096, tp_size=2, head_k_dim=128, head_v_dim=128
)


def unpack(mixed):
    return helper(mixed, layer)
