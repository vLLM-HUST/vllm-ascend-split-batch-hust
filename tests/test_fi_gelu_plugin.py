from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_ascend_split_batch import fi_gelu_plugin as plugin


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    plugin.uninstall()
    plugin._reset_for_tests()
    monkeypatch.delenv(plugin.ENV_ENABLE, raising=False)
    yield
    plugin.uninstall()
    plugin._reset_for_tests()


@pytest.fixture
def host(monkeypatch):
    registry = {}

    class CustomOp:
        @classmethod
        def register_oot(cls, *, name):
            def register(provider):
                assert name not in registry
                provider.name = name
                registry[name] = provider
                return provider

            return register

    class GeluAndMul(CustomOp):
        name = "gelu_and_mul"

        def __new__(cls, *args, **kwargs):
            return object.__new__(registry.get(cls.__name__, cls))

        def __init__(self, approximate="none"):
            self.approximate = approximate

        def forward_oot(self, x):
            return "host"

    modules = {}
    for name in (
        "torch",
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        plugin.HOST_MODULE,
        "vllm.model_executor.custom_op",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    modules["torch"].bfloat16 = "bf16"
    modules[plugin.HOST_MODULE].GeluAndMul = GeluAndMul
    custom = modules["vllm.model_executor.custom_op"]
    custom.CustomOp = CustomOp
    custom.op_registry_oot = registry
    monkeypatch.setattr(
        plugin, "_load_fi_api", lambda: SimpleNamespace(gelu_and_mul=lambda x: "kernel")
    )
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    return SimpleNamespace(cls=GeluAndMul, registry=registry, custom=custom)


def tensor(dtype="bf16", shape=(4, 8), device="npu"):
    return SimpleNamespace(
        dtype=dtype,
        shape=shape,
        device=SimpleNamespace(type=device),
        dim=lambda: len(shape),
    )


def test_default_off_has_no_heavy_imports():
    code = """
import sys
from vllm_ascend_split_batch import fi_gelu_plugin as p
assert not p.load()
assert not p.install()
assert p._registered_cls is None
assert not any(n.split('.')[0] in ('torch', 'torch_npu', 'triton', 'vllm',
                                  'vllm_ascend') for n in sys.modules)
assert not any(n.startswith('vllm_ascend_split_batch.fi_gelu.')
               for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd="/tmp", capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", ["", "0", "true", "yes", " 1", "1 "])
def test_exact_enable_token(monkeypatch, value):
    monkeypatch.setenv(plugin.ENV_ENABLE, value)
    assert not plugin.load()


def test_registration_preserves_host_and_controls_new_instances(host):
    original = dict(host.cls.__dict__)
    before = host.cls()
    assert plugin.load()
    provider = plugin._registered_cls
    assert host.registry == {"GeluAndMul": provider}
    assert provider.name == host.cls.name == "gelu_and_mul"
    assert host.cls.__dict__ == original
    after = host.cls()
    assert type(before) is host.cls
    assert type(after) is provider
    assert before.forward_oot(tensor()) == "host"
    assert after.forward_oot(tensor()) == "kernel"
    assert plugin.load()
    assert plugin._registered_cls is provider
    plugin.uninstall()
    assert not host.registry
    assert type(host.cls()) is host.cls
    assert after.forward_oot(tensor()) == "kernel"
    plugin.uninstall()


def test_foreign_provider_is_never_overwritten_or_removed(host, monkeypatch):
    foreign = type("Foreign", (host.cls,), {})
    host.registry["GeluAndMul"] = foreign
    monkeypatch.setattr(plugin, "_load_fi_api", lambda: pytest.fail("kernel import"))
    assert not plugin.load()
    plugin.uninstall()
    assert host.registry["GeluAndMul"] is foreign


def test_uninstall_preserves_later_foreign_provider(host):
    assert plugin.load()
    foreign = object()
    host.registry["GeluAndMul"] = foreign
    assert not plugin.load()
    plugin.uninstall()
    assert host.registry["GeluAndMul"] is foreign


@pytest.mark.parametrize("partial", [False, True])
def test_registration_failure_cleans_only_owned_entry(host, monkeypatch, partial):
    def register(*, name):
        def fail(provider):
            if partial:
                host.registry[name] = provider
            raise RuntimeError("registration unavailable")

        return fail

    monkeypatch.setattr(host.custom.CustomOp, "register_oot", register)
    assert not plugin.load()
    assert not host.registry
    assert plugin._registered_cls is None


def test_missing_host_is_fail_open(monkeypatch):
    monkeypatch.setenv(plugin.ENV_ENABLE, "1")
    monkeypatch.setitem(sys.modules, "torch", None)
    assert not plugin.load()
    assert plugin._registered_cls is None


@pytest.mark.parametrize(
    "api", [None, SimpleNamespace(), SimpleNamespace(gelu_and_mul=42)]
)
def test_missing_api_does_not_register(host, monkeypatch, api):
    monkeypatch.setattr(plugin, "_load_fi_api", lambda: api)
    assert not plugin.load()
    assert not host.registry


@pytest.mark.parametrize(
    "x",
    [
        tensor(dtype="fp32"),
        tensor(shape=(8,)),
        tensor(shape=(4, 0)),
        tensor(shape=(4, 7)),
        tensor(device="cpu"),
    ],
)
def test_ineligible_inputs_keep_host_fallback(host, x):
    assert plugin.load()
    assert host.cls().forward_oot(x) == "host"


def test_tanh_and_kernel_failure_keep_host_fallback(host, monkeypatch, caplog):
    def fail(x):
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(
        plugin, "_load_fi_api", lambda: SimpleNamespace(gelu_and_mul=fail)
    )
    assert plugin.load()
    assert host.cls("tanh").forward_oot(tensor()) == "host"
    for _ in range(2):
        assert host.cls().forward_oot(tensor()) == "host"
    assert sum("kernel call failed" in r.message for r in caplog.records) == 1
