"""Compatibility shims for the mflux/MLX pair this app pins.

Everything here patches a third-party module, so each shim states what it
fixes, which versions need it, and how you would know it is safe to delete.
"""

from __future__ import annotations

import functools
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def _patch_attention_scale() -> bool:
    """Coerce mflux's ``mx.array`` attention scale to the float MLX declares.

    mflux 0.18.0 computes the scale as ``1 / mx.sqrt(q.shape[-1])`` in a dozen
    places (the FLUX.2 VAE mid-block, the Qwen VAE, FLUX/Fibo/Z-Image
    attention). ``mx.sqrt`` of a Python int returns a 0-d ``mx.array``, not a
    float, while MLX 0.32.0 declares ``scale: float``. Users have hit:

        scaled_dot_product_attention(): incompatible function arguments
        Invoked with ... kwargs = { scale: mlx.core.array }

    This is defensive rather than a fix for a reproduced failure. On a
    consistent mlx/mlx-cuda-13 0.32.0 install, nanobind implicitly converts a
    0-d array of any dtype and every one of those call sites works — verified
    directly, including instantiating the FLUX.2 VAE attention block. The
    error appears when that implicit conversion is unavailable, which in
    practice means a skewed install: ``mlx`` (Python bindings) and
    ``mlx-cuda-13`` (native library) at different versions, as a partial or
    interrupted reinstall can leave them. requirements.txt on its own resolves
    mlx down to mflux's ceiling, so that skew is reachable if the second
    install from requirements-cuda.txt does not complete.

    Keeping the coercion costs one isinstance check per attention call and
    removes the whole class of failure. Delete it once mflux passes a float —
    the wrapper then finds nothing to convert and is inert either way.

    The real repair for a skewed install is to reinstall both packages at the
    same version; this shim only stops it from presenting as an unreadable
    signature error mid-edit.

    Returns True if a patch was applied.
    """
    try:
        import mlx.core as mx
        import mlx.core.fast as fast
    except ImportError:
        return False

    original = getattr(fast, "scaled_dot_product_attention", None)
    if original is None or getattr(original, "_sanitos_scale_shim", False):
        return False

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        scale = kwargs.get("scale")
        if scale is not None and not isinstance(scale, (int, float)):
            try:
                kwargs["scale"] = float(scale)
            except (TypeError, ValueError):
                pass  # not something we can convert; let MLX report it
        return original(*args, **kwargs)

    wrapper._sanitos_scale_shim = True  # type: ignore[attr-defined]

    fast.scaled_dot_product_attention = wrapper
    if getattr(mx, "fast", None) is not None:
        mx.fast.scaled_dot_product_attention = wrapper  # same module object

    # Modules that already ran `from mlx.core.fast import
    # scaled_dot_product_attention` hold their own reference to the original,
    # which reassigning the attribute above does not reach.
    rebound = 0
    for module in list(sys.modules.values()):
        if module is None or module is fast:
            continue
        if getattr(module, "scaled_dot_product_attention", None) is original:
            module.scaled_dot_product_attention = wrapper
            rebound += 1

    logger.debug(
        "Patched scaled_dot_product_attention for a float scale (%d module(s) rebound).",
        rebound,
    )
    return True


def apply_compatibility_patches() -> None:
    """Apply every shim. Safe to call more than once.

    Call this before importing any mflux pipeline: modules that bind
    ``scaled_dot_product_attention`` at import time are rebound here, but doing
    it up front keeps that path from mattering.
    """
    try:
        _patch_attention_scale()
    except Exception as exc:  # noqa: BLE001 - a failed shim must not stop startup
        logger.warning("Could not apply the MLX attention-scale shim: %s", exc)
