"""Tests for the third-party shims in src/compat.py.

These patch someone else's module, so the risk is that the patch silently stops
applying — mflux renames a call site, MLX moves the function — and the failure
resurfaces as a confusing runtime error. Each test pins one observable effect
rather than the implementation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import compat  # noqa: E402


class _FakeArray:
    """Stands in for the 0-d mx.array mflux passes as a scale."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return self.value


def _install_fake_mlx(monkeypatch, recorder: list) -> types.ModuleType:
    """Put a minimal fake mlx.core / mlx.core.fast into sys.modules."""

    def strict_attention(q, k, v, *, scale, **kwargs):
        # Mirrors MLX >= 0.32: a non-float scale is a hard error.
        if not isinstance(scale, float):
            raise TypeError(f"scale must be float, got {type(scale).__name__}")
        recorder.append(scale)
        return "attended"

    fast = types.ModuleType("mlx.core.fast")
    fast.scaled_dot_product_attention = strict_attention

    core = types.ModuleType("mlx.core")
    core.fast = fast

    root = types.ModuleType("mlx")
    root.core = core

    monkeypatch.setitem(sys.modules, "mlx", root)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx.core.fast", fast)
    return fast


def test_an_array_scale_is_coerced_to_float(monkeypatch) -> None:
    """The actual mflux 0.18.0 bug: 1 / mx.sqrt(int) is an array, not a float."""
    calls: list = []
    fast = _install_fake_mlx(monkeypatch, calls)

    assert compat._patch_attention_scale() is True
    result = fast.scaled_dot_product_attention(1, 2, 3, scale=_FakeArray(0.125))

    assert result == "attended"
    assert calls == [0.125]


def test_a_float_scale_is_passed_through_untouched(monkeypatch) -> None:
    calls: list = []
    fast = _install_fake_mlx(monkeypatch, calls)
    compat._patch_attention_scale()

    fast.scaled_dot_product_attention(1, 2, 3, scale=0.5)
    assert calls == [0.5]


def test_modules_that_imported_the_function_directly_are_rebound(monkeypatch) -> None:
    """mflux does `from mlx.core.fast import scaled_dot_product_attention`.

    Reassigning the attribute on the module does not reach a name already bound
    into an importer's namespace, so the shim has to rebind those too — this is
    the part most likely to rot silently.
    """
    calls: list = []
    fast = _install_fake_mlx(monkeypatch, calls)

    borrower = types.ModuleType("fake_mflux_attention")
    borrower.scaled_dot_product_attention = fast.scaled_dot_product_attention
    monkeypatch.setitem(sys.modules, "fake_mflux_attention", borrower)

    compat._patch_attention_scale()

    borrower.scaled_dot_product_attention(1, 2, 3, scale=_FakeArray(0.25))
    assert calls == [0.25]


def test_patching_twice_does_not_stack_wrappers(monkeypatch) -> None:
    calls: list = []
    fast = _install_fake_mlx(monkeypatch, calls)

    assert compat._patch_attention_scale() is True
    once = fast.scaled_dot_product_attention
    assert compat._patch_attention_scale() is False
    assert fast.scaled_dot_product_attention is once


def test_apply_survives_a_broken_shim(monkeypatch) -> None:
    """A failed shim must not stop the app from starting."""
    def boom() -> bool:
        raise RuntimeError("no mlx here")

    monkeypatch.setattr(compat, "_patch_attention_scale", boom)
    compat.apply_compatibility_patches()  # must not raise
