import ast
from types import SimpleNamespace

from vllm_ascend_split_batch.runtime_source_patches import _VIEWS_TRANSFORMS

source = next(
    new for _, new, _ in _VIEWS_TRANSFORMS if "def _unpack_mixed_qkv_views" in new
)
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), __file__, "exec"), namespace)
helper = namespace["_unpack_mixed_qkv_views"]
layer = SimpleNamespace(
    key_dim=2048, value_dim=4096, tp_size=2, head_k_dim=128, head_v_dim=128
)


def unpack(mixed):
    return helper(mixed, layer)
