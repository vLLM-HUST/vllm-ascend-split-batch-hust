# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Drift guard for the RoPE variant defect carrier (``rope_fix_plugin``).

The carrier exists because three defects live in the *fork* (see
``docs/design/rope-variant-defects.md``); it patches nothing until the host
tells it to and it must never become a silent duplicate of an upstream fix.

Therefore the core assertions here are **reverse assertions**: they assert that
the defects are STILL PRESENT upstream.

* ``test_llama3_is_still_unwired`` /
  ``test_triton_mrope_call_still_misses_is_neox_style`` /
  ``test_yarn_truncate_default_is_still_false`` assert the *bug* still exists.
  The day upstream fixes one of them these tests fail **by design**: the
  corresponding override has become dead code (worst case it shadows a fixed
  host class) and must be dropped from ``rope_fix_plugin``.

Every failure is worded to say which of the two things happened, so a red run
can be triaged at a glance:

* ``upstream fixed defect ①/②/③ → drop the corresponding override``
  -- the anchor is still there, the defect is gone: delete the override.
* ``anchor drifted → re-audit HOST_CONTRACT``
  -- the seam this carrier leans on was renamed/moved/re-shaped: re-derive the
  carrier (and the ``HOST_CONTRACT.md`` section that documents it) before
  anything else.

Also pinned here: the two seams the carrier *uses* (``op_registry_oot`` read at
instantiation time + the duplicate-name assert that forbids pre-planting a key),
the host signatures the subclasses must mirror (the llama3 10-argument
construction signature, ``AscendRotaryEmbedding.forward_oot``), the mrope/CustomOp
attributes the mirrored body touches, the host ``triton_mrope`` arity, and an
AST-level proof that the mirrored ``forward_triton`` is the host body plus
exactly the injected argument.

Note on naming: the task's "``head_dim``" attribute of ``AscendMRotaryEmbedding``
is spelled ``head_size`` on this baseline (``RotaryEmbeddingBase.__init__``);
the guard pins the real name.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest

from vllm_ascend_split_batch import rope_fix_plugin as plugin

#: both failure wordings, spelled once (see the module docstring).
FIXED = "upstream fixed defect {} → drop the corresponding override"
DRIFT = "anchor drifted → re-audit HOST_CONTRACT"

#: host seam names, imported from the carrier so both sides cannot drift apart.
HOST_UTILS_MODULE = plugin.HOST_UTILS_MODULE
HOST_REGISTRATION_FUNC = plugin.HOST_REGISTRATION_FUNC
HOST_ROPE_MODULE = plugin.HOST_ROPE_MODULE

PLUGIN_SRC = Path(plugin.__file__).resolve()


def _host_root() -> Path | None:
    """Locate the vllm-ascend source tree without importing it.

    ``importlib.util.find_spec`` reads the editable-install path hooks and does
    NOT execute the package, so the source-level guards stay import-free (and
    therefore runnable on a CPU-only box).  The last candidate is derived from
    this file's own location (deterministic -- no CWD dependency): the workspace
    layout puts the host checkout next to this repository.
    """
    env = __import__("os").environ.get("VLLM_ASCEND_HUST_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env) / "vllm_ascend")
        candidates.append(Path(env))
    try:
        spec = importlib.util.find_spec("vllm_ascend")
    except Exception:  # noqa: BLE001 -- no host tree on this box
        spec = None
    if spec is not None:
        for location in spec.submodule_search_locations or ():
            candidates.append(Path(location))
    # <workspace>/vllm-ascend-split-batch-hust/tests/x.py -> <workspace>
    workspace = Path(__file__).resolve().parents[2].parent
    candidates.append(workspace / "vllm-ascend-hust" / "vllm_ascend")
    for candidate in candidates:
        if (candidate / "utils.py").is_file() and (
            candidate / "ops" / "rotary_embedding.py"
        ).is_file():
            return candidate.resolve()
    return None


HOST_ROOT = _host_root()

#: This module cannot say anything useful without the host source tree.  It is
#: marked (not skipped) so the guard keeps failing loudly on the full run, while
#: the dependency-free CI job deselects it explicitly with
#: ``pytest -m "not host_tree"``.  See ``docs/release.md`` §8.
pytestmark = pytest.mark.host_tree


@pytest.fixture(scope="module", autouse=True)
def _require_host_tree():
    """Fail the module when the host checkout is missing (guard, not skip)."""
    if HOST_ROOT is None:
        # Deliberately a failure, not a skip: this guard is the only thing
        # standing between the carrier and a silent upstream fix (or a silently
        # moved seam).  Missing evidence must be red -- and on the org CI job the
        # markers are deselected before this fixture can run.  Point
        # VLLM_ASCEND_HUST_ROOT at the host checkout to run it.
        pytest.fail(
            "vllm-ascend source tree not found; the reverse defect guard (and the "
            "mirror-equivalence proof) cannot be evaluated.  Set "
            "VLLM_ASCEND_HUST_ROOT to the vllm-ascend checkout to run it.",
            pytrace=False,
        )


UTILS_PY = (HOST_ROOT / "utils.py") if HOST_ROOT else Path()
ROPE_PY = (HOST_ROOT / "ops" / "rotary_embedding.py") if HOST_ROOT else Path()
WORKER_PY = (HOST_ROOT / "worker" / "worker.py") if HOST_ROOT else Path()

LLAMA3_SIGNATURE = (
    "head_size",
    "rotary_dim",
    "max_position_embeddings",
    "base",
    "is_neox_style",
    "dtype",
    "scaling_factor",
    "low_freq_factor",
    "high_freq_factor",
    "orig_max_position",
)
FORWARD_OOT_SIGNATURE = (
    ("positions", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("query", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("key", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("offsets", "POSITIONAL_OR_KEYWORD", None),
    ("is_neox_style_override", "POSITIONAL_OR_KEYWORD", None),
    ("out_dtype", "POSITIONAL_OR_KEYWORD", None),
)


# ------------------------------------------------------------------ AST tools
def _parse(path: Path) -> ast.Module:
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def _class_def(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _func_def(node: ast.AST, name: str) -> ast.FunctionDef | None:
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef) and child.name == name:
            return child
    return None


def _name_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _called_name(node: ast.Call) -> str | None:
    return _name_chain(node.func)


def _dict_keys(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Dict):
        return set()
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def _registered_op_keys(func: ast.FunctionDef) -> set[str]:
    """Every op name ``register_ascend_customop`` writes into its registry dict."""
    keys: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "REGISTERED_ASCEND_OPS"
                ):
                    keys |= _dict_keys(node.value)
                elif (
                    isinstance(target, ast.Subscript)
                    and _name_chain(target.value) == "REGISTERED_ASCEND_OPS"
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    keys.add(target.slice.value)
        elif (
            isinstance(node, ast.Call)
            and _name_chain(node.func) == "REGISTERED_ASCEND_OPS.update"
        ):
            for arg in node.args:
                keys |= _dict_keys(arg)
    return keys


def _arg_named(func: ast.FunctionDef, name: str) -> ast.arg | None:
    args = func.args
    for candidate in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        if candidate.arg == name:
            return candidate
    return None


def _default_for(func: ast.FunctionDef, name: str) -> ast.AST | None:
    args = func.args
    positional = [*args.posonlyargs, *args.args]
    for index, candidate in enumerate(positional):
        if candidate.arg == name and index >= len(positional) - len(args.defaults):
            return args.defaults[index - (len(positional) - len(args.defaults))]
    for candidate, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        if candidate.arg == name:
            return default
    return None


def _calls_to(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _called_name(child) == name
    ]


# ------------------------------------------------------- reverse defect guard
def test_llama3_is_still_unwired() -> None:
    """Defect ①: ``REGISTERED_ASCEND_OPS`` must still lack llama3-scaling."""
    tree = _parse(UTILS_PY)
    func = _func_def(tree, "register_ascend_customop")
    if func is None:
        pytest.fail(
            f"{DRIFT}: register_ascend_customop is gone from {UTILS_PY}",
            pytrace=False,
        )
    keys = _registered_op_keys(func)
    for expected in (
        "RotaryEmbedding",
        "MRotaryEmbedding",
        "YaRNScalingRotaryEmbedding",
    ):
        if expected not in keys:
            pytest.fail(
                f"{DRIFT}: the registry extractor no longer sees {expected!r} in "
                f"register_ascend_customop ({sorted(keys)}); the dict shape changed.",
                pytrace=False,
            )
    if "Llama3RotaryEmbedding" in keys:
        pytest.fail(
            f"{FIXED.format('①')}: vllm_ascend/utils.py now registers "
            "'Llama3RotaryEmbedding' -> the fork is wired itself, so the "
            "llama3 override in rope_fix_plugin is dead code.",
            pytrace=False,
        )


def test_triton_mrope_call_still_misses_is_neox_style() -> None:
    """Defect ②: the fork's interleaved mrope path must still drop the argument."""
    tree = _parse(ROPE_PY)
    cls = _class_def(tree, "AscendMRotaryEmbedding")
    if cls is None:
        pytest.fail(
            f"{DRIFT}: class AscendMRotaryEmbedding is gone from {ROPE_PY}",
            pytrace=False,
        )
    func = _func_def(cls, "forward_triton")
    if func is None:
        pytest.fail(
            f"{DRIFT}: AscendMRotaryEmbedding.forward_triton is gone from {ROPE_PY}",
            pytrace=False,
        )
    calls = _calls_to(func, "triton_mrope")
    if not calls:
        pytest.fail(
            f"{DRIFT}: forward_triton no longer calls triton_mrope; the mirrored "
            "override in rope_fix_plugin is stale.",
            pytrace=False,
        )
    if len(calls) > 1:
        pytest.fail(
            f"{DRIFT}: forward_triton now has {len(calls)} triton_mrope call "
            "sites; the single-call mirror no longer holds.",
            pytrace=False,
        )
    call = calls[0]
    keywords = [keyword.arg for keyword in call.keywords]
    if len(call.args) >= 9 or "is_neox_style" in keywords:
        pytest.fail(
            f"{FIXED.format('②')}: forward_triton now passes is_neox_style "
            f"({len(call.args)} positional, keywords={keywords}) -> drop the "
            "mirrored forward_triton override.",
            pytrace=False,
        )
    if len(call.args) != 8:
        pytest.fail(
            f"{DRIFT}: the triton_mrope call site changed shape "
            f"({len(call.args)} positional args, keywords={keywords}); re-derive "
            "the mirrored override.",
            pytrace=False,
        )


def test_yarn_truncate_default_is_still_false() -> None:
    """Defect ③: the fork's YaRN default must still diverge from vLLM/HF."""
    tree = _parse(ROPE_PY)
    cls = _class_def(tree, "AscendYaRNRotaryEmbedding")
    if cls is None:
        pytest.fail(
            f"{DRIFT}: class AscendYaRNRotaryEmbedding is gone from {ROPE_PY}",
            pytrace=False,
        )
    func = _func_def(cls, "__init__")
    if func is None:
        pytest.fail(
            f"{DRIFT}: AscendYaRNRotaryEmbedding.__init__ is gone from {ROPE_PY}",
            pytrace=False,
        )
    if _arg_named(func, "truncate") is None:
        pytest.fail(
            f"{DRIFT}: AscendYaRNRotaryEmbedding.__init__ has no `truncate` "
            "parameter any more; re-derive the YaRN override.",
            pytrace=False,
        )
    default = _default_for(func, "truncate")
    if not (isinstance(default, ast.Constant) and default.value is False):
        rendered = ast.unparse(default) if default is not None else "<required>"
        pytest.fail(
            f"{FIXED.format('③')}: the fork's `truncate` default is now "
            f"{rendered} (it must still be False for this override to be "
            "needed) -> drop the YaRN override.",
            pytrace=False,
        )


# ------------------------------------------------- host-interface assertions
def _host_rope():
    importlib.import_module("vllm_ascend.ops")
    return pytest.importorskip(
        "vllm_ascend.ops.rotary_embedding",
        reason="the live host signatures need the full vllm-ascend stack",
    )


def test_llama3_construction_signature_is_unchanged() -> None:
    """The override is instantiated with the host's own argument list."""
    rope_layers = pytest.importorskip(
        "vllm.model_executor.layers.rotary_embedding",
        reason="the live host signatures need vllm",
    )
    params = list(
        inspect.signature(rope_layers.Llama3RotaryEmbedding.__init__).parameters
    )
    names = tuple(name for name in params if name != "self")
    if names != LLAMA3_SIGNATURE:
        pytest.fail(
            f"{DRIFT}: Llama3RotaryEmbedding.__init__ is {names}, the carrier's "
            f"override mirrors {LLAMA3_SIGNATURE}",
            pytrace=False,
        )
    tree = _parse(PLUGIN_SRC)
    cls = _class_def(tree, "AscendFixLlama3RotaryEmbedding")
    assert cls is not None, "the llama3 override disappeared from the carrier"
    own = tuple(
        arg.arg
        for arg in (*_func_def(cls, "__init__").args.args,)  # type: ignore[union-attr]
    )
    assert own == ("self", *LLAMA3_SIGNATURE), own


def test_ascend_rotary_forward_oot_signature_is_unchanged() -> None:
    """The llama3/YaRN overrides delegate to this exact method."""
    fork_rope = _host_rope()
    signature = inspect.signature(fork_rope.AscendRotaryEmbedding.forward_oot)
    actual = tuple(
        (name, param.kind.name, param.default)
        for name, param in signature.parameters.items()
        if name != "self"
    )
    if actual != FORWARD_OOT_SIGNATURE:
        pytest.fail(
            f"{DRIFT}: AscendRotaryEmbedding.forward_oot is {actual}, the "
            f"delegating overrides call {FORWARD_OOT_SIGNATURE}",
            pytrace=False,
        )


def test_ascend_mrope_attributes_the_mirror_touches_still_exist() -> None:
    """``forward_triton`` reads mrope_interleaved / is_neox_style / head_size."""
    fork_rope = _host_rope()
    sources = []
    for cls in fork_rope.AscendMRotaryEmbedding.__mro__:
        try:
            sources.append(inspect.getsource(cls.__init__))
        except (AttributeError, OSError, TypeError):
            continue
    joined = "\n".join(sources)
    missing = [
        attribute
        for attribute in ("mrope_interleaved", "is_neox_style", "head_size")
        if f"self.{attribute} =" not in joined
    ]
    if missing:
        pytest.fail(
            f"{DRIFT}: {missing} are no longer assigned by the "
            "AscendMRotaryEmbedding MRO __init__; the mirrored forward_triton "
            "reads them (`head_dim` is spelled `head_size` on this baseline).",
            pytrace=False,
        )


def test_triton_mrope_arity_is_still_nine() -> None:
    """The mirrored call appends argument #9; the host must still want nine.

    Read from the *fork module's own* binding -- the object the mirrored call site
    resolves -- so this cannot pass against a different symbol.  When the fork
    reports ``HAS_TRITON`` the check is mandatory (a missing binding here means
    the carrier silently stops carrying defect ②, so it must be red, not skipped);
    only a triton-less build may skip, where nothing carries ② either.
    """
    fork_rope = _host_rope()
    if not getattr(fork_rope, "HAS_TRITON", False):
        pytest.skip("build without triton: neither the fork nor the carrier has ②")

    kernel = getattr(fork_rope, "triton_mrope", None)
    if kernel is None:
        pytest.fail(
            f"{DRIFT}: the fork reports HAS_TRITON but {HOST_ROPE_MODULE}"
            ".triton_mrope is missing; defect ② has nothing to be carried with.",
            pytrace=False,
        )
    params = list(inspect.signature(kernel).parameters)
    if len(params) != 9 or params[-1] != "is_neox_style":
        pytest.fail(
            f"{DRIFT}: triton_mrope takes {params} (the fork's call site passes "
            "9 positional arguments, the last one is_neox_style)",
            pytrace=False,
        )


def test_worker_module_still_binds_the_registration_function_directly() -> None:
    """The wrapper rebinding list is not hypothetical.

    ``vllm_ascend/worker/worker.py`` does ``from vllm_ascend.utils import
    register_ascend_customop`` and calls it at ``NPUWorker.__init__`` time, so a
    carrier that only patched the ``vllm_ascend.utils`` attribute would miss the
    live call site entirely.  If upstream ever switches to
    ``vllm_ascend.utils.register_ascend_customop(...)``, the rebinding rationale
    (and HOST_CONTRACT.md's anchor note) has to be re-audited.
    """
    tree = _parse(WORKER_PY)
    imported = [
        alias
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == HOST_UTILS_MODULE
        for alias in node.names
        if alias.name == HOST_REGISTRATION_FUNC
    ]
    if not imported:
        pytest.fail(
            f"{DRIFT}: {WORKER_PY} no longer imports "
            f"{HOST_REGISTRATION_FUNC} from {HOST_UTILS_MODULE} directly; "
            "re-audit the wrapper's call-site rebinding list.",
            pytrace=False,
        )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) == HOST_REGISTRATION_FUNC
    ]
    if not calls:
        pytest.fail(
            f"{DRIFT}: {WORKER_PY} imports {HOST_REGISTRATION_FUNC} but never "
            "calls it; re-audit the registration-order contract.",
            pytrace=False,
        )


def test_registry_lever_still_behaves_as_documented() -> None:
    """``op_registry_oot`` is read at instantiation and refuses duplicates.

    Both halves matter: the duplicate assert is why the carrier refuses to
    pre-plant the two replacement keys, and the instantiation-time read is why
    replacing the entries *after* the host pass is effective.
    """
    custom_op = pytest.importorskip(
        "vllm.model_executor.custom_op",
        reason="the live registry needs vllm",
    )
    assert isinstance(getattr(custom_op, "op_registry_oot", None), dict)
    new_source = inspect.getsource(custom_op.CustomOp.__new__)
    if "op_registry_oot" not in new_source:
        pytest.fail(
            f"{DRIFT}: CustomOp.__new__ no longer consults op_registry_oot",
            pytrace=False,
        )

    probe_name = "_rope_fix_drift_probe"

    class _Probe(custom_op.CustomOp):
        pass

    custom_op.op_registry_oot.pop(probe_name, None)
    try:
        custom_op.CustomOp.register_oot(_decorated_op_cls=_Probe, name=probe_name)
        # The lever the carrier uses: the *entry*, not a second registration.
        assert custom_op.op_registry_oot[probe_name] is _Probe
        assert _Probe.name == probe_name
        with pytest.raises(AssertionError, match="Duplicate op name"):
            custom_op.CustomOp.register_oot(_decorated_op_cls=_Probe, name=probe_name)
    finally:
        custom_op.op_registry_oot.pop(probe_name, None)


# --------------------------------------------------------- mirrored-body proof
def _normalised_dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _strip_injected_argument(func: ast.FunctionDef) -> None:
    """Drop the extra positional the carrier appends to the kernel call."""
    for call in _calls_to(func, "triton_mrope"):
        call.args = call.args[:-1]


def test_mirrored_forward_triton_is_the_host_body_plus_one_argument() -> None:
    """The mirror must stay the host method, not a fork of it.

    Any *other* change upstream (new guard, new cache handling, reordered
    statement) invalidates the mirror silently, so the two bodies are compared
    as ASTs after removing the injected argument.

    A body mismatch has two possible causes, and the wording says which:

    * the host call site already supplies the 9th argument -> upstream fixed
      defect ② itself, the override is now dead code;
    * anything else -> the seam moved, re-audit HOST_CONTRACT.
    """
    host_tree = _parse(ROPE_PY)
    host_cls = _class_def(host_tree, "AscendMRotaryEmbedding")
    if host_cls is None:
        pytest.fail(
            f"{DRIFT}: class AscendMRotaryEmbedding is gone from {ROPE_PY}, so "
            "the mirrored override has nothing to mirror.",
            pytrace=False,
        )
    host_func = _func_def(host_cls, "forward_triton")
    if host_func is None:
        pytest.fail(
            f"{DRIFT}: AscendMRotaryEmbedding.forward_triton is gone from {ROPE_PY}.",
            pytrace=False,
        )

    plugin_tree = _parse(PLUGIN_SRC)
    plugin_cls = _class_def(plugin_tree, "AscendFixMRotaryEmbedding")
    assert plugin_cls is not None, "the mrope override disappeared from the carrier"
    plugin_func = _func_def(plugin_cls, "forward_triton")
    assert plugin_func is not None

    calls = _calls_to(plugin_func, "triton_mrope")
    assert len(calls) == 1, "the mirror must keep exactly one kernel call"
    last = calls[0].args[-1]
    assert isinstance(last, ast.Attribute), ast.dump(last)
    assert _name_chain(last) == "self.is_neox_style", ast.dump(last)

    mirrored = ast.parse(ast.unparse(plugin_func)).body[0]
    assert isinstance(mirrored, ast.FunctionDef)
    _strip_injected_argument(mirrored)
    if _normalised_dump(mirrored) != _normalised_dump(host_func):
        host_calls = _calls_to(host_func, "triton_mrope")
        host_fixed = bool(host_calls) and _host_call_passes_neox_style(host_calls[0])
        reason = FIXED.format("②") if host_fixed else DRIFT
        pytest.fail(
            f"{reason}: the mirrored forward_triton no longer equals the host "
            f"body plus the injected `self.is_neox_style`"
            f"{' (the host call site now passes it)' if host_fixed else ''}.\n"
            f"host  : {ast.unparse(host_func)}\n"
            f"mirror: {ast.unparse(plugin_func)}",
            pytrace=False,
        )


def _host_call_passes_neox_style(call: ast.Call) -> bool:
    """True when the host's ``triton_mrope`` call already supplies argument #9."""
    if any(keyword.arg == "is_neox_style" for keyword in call.keywords):
        return True
    return len(call.args) >= 9


def test_yarn_override_stays_signature_agnostic_with_truncate_true() -> None:
    """The YaRN override must only change the default, nothing else."""
    tree = _parse(PLUGIN_SRC)
    cls = _class_def(tree, "AscendFixYaRNRotaryEmbedding")
    assert cls is not None, "the YaRN override disappeared from the carrier"
    func = _func_def(cls, "__init__")
    assert func is not None
    assert func.args.vararg is not None, (
        "the YaRN override must accept *args (host call shape may drift)"
    )
    assert func.args.kwarg is not None, (
        "the YaRN override must accept **kwargs (host call shape may drift)"
    )
    truncate = _arg_named(func, "truncate")
    assert truncate is not None and truncate.arg == "truncate"
    default = _default_for(func, "truncate")
    assert isinstance(default, ast.Constant) and default.value is True, (
        f"the YaRN override must default truncate=True (vLLM/HF), found "
        f"{ast.unparse(default) if default is not None else '<required>'}"
    )
    forwards = _calls_to(func, "super")
    assert forwards, "the YaRN override must delegate to the fork class"
