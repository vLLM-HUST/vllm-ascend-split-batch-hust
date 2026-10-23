"""Resolve KV-cache helpers across the 2026-10-13 vLLM module split.

Upstream vLLM moved KV-cache planning out of ``vllm.v1.core.kv_cache_utils``
into ``vllm.v1.core.kv_cache_planning``.  Import and monkey-patch through this
module so one plugin build works on both layouts: a name is always resolved to
the module that currently owns it, and a patch is written back to that same
module.  Writing to a fixed module would silently stop applying once a symbol
moves, which is worse than failing.
"""

from vllm.v1.core import kv_cache_planning, kv_cache_utils

# Order matters: the original module wins, so names that never moved keep their
# pre-refactor owner; only names absent there fall through to the new module.
_SEARCH = (kv_cache_utils, kv_cache_planning)


def _owner(name):
    for module in _SEARCH:
        if hasattr(module, name):
            return module
    raise AttributeError(f"no kv-cache module provides {name!r}")


def __getattr__(name):
    return getattr(_owner(name), name)


class _PatchTarget:
    """Read/write proxy that acts on whichever module owns the name."""

    def __getattr__(self, name):
        return getattr(_owner(name), name)

    def __setattr__(self, name, value):
        setattr(_owner(name), name, value)


kc = _PatchTarget()
