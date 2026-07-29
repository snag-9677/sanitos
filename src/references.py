"""Making reference images cheaper than the image being edited.

mflux encodes every image — the source and each reference — at the *output*
resolution, so shrinking a reference file changes nothing: ``LatentCreator.
encode_image`` resizes it straight back up first. What a reference actually
costs is the tokens it adds to the joint attention stream, which every one of
the 32 layers then processes on every denoise step.

The lever that works is encoding references at a smaller grid than the source.
``prepare_reference_image_conditioning`` builds each image's grid ids from its
own encoded shape and concatenates along the token axis, so mixed sizes are
structurally fine — a reference at half scale costs a quarter of the pixels,
while the edited image keeps its full detail.

Peak memory is linear in that stream length. The single coefficient is
empirical and carries the measurements it came from; two tidier theories were
tried and disproven first, recorded at MIB_PER_STREAM_TOKEN.

This module owns the arithmetic (pure, tested) and the patch that applies it
(FLUX.2 only — Qwen conditions images through a different path).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Latent grid stride: the VAE downsamples by 8, then 2x2 patching halves again.
PIXELS_PER_TOKEN = 16

# Scales tried by "auto", largest first. Below a quarter scale a reference
# stops carrying usable identity, so that is the floor.
AUTO_SCALES = (1.0, 0.75, 0.5, 0.375, 0.25)

# Smallest edge we will shrink a reference to, in pixels.
MIN_REFERENCE_EDGE = 256

# Text tokens in the joint stream. mflux pads the prompt to max_sequence_length
# unconditionally, so this is a constant, not a function of the prompt.
TEXT_TOKENS = 512

# Peak memory per token of the joint attention stream, in MiB.
#
# The law is LINEAR in total stream length. Two tidier theories were measured
# wrong first; both are recorded so they are not reinvented:
#
#   "attention is quadratic, so cost is queries x keys" — no. On this GPU MLX
#   dispatches to a fused cuDNN flash-attention kernel, which never
#   materialises the score matrix. Measured: quadrupling the key/value tokens
#   (2048x1024 -> 2048x8192) costs exactly zero extra bytes. MLX only falls
#   back to the quadratic path when head_dim > 128 or head_dim % 8, or on
#   compute capability < 8; Klein 9B is head_dim 128 on sm_86, so it never
#   does.
#
#   "reference tokens live in the KV cache" — no. Klein 9B reports
#   supports_kv_cache = False, so nothing is cached between steps and the full
#   stream is recomputed every step.
#
# What remains is per-layer activations across 32 attention layers, which are
# linear in stream length. The exact coefficient is not derivable from the
# architecture — MLX frees intermediates by refcount during graph
# construction, so how many per-token buffers are live at once is decided by
# the scheduler. This is the one genuinely empirical constant.
#
# Bracketed by measurement on a 24 GB RTX A5000 (FLUX.2 Klein 9B, low mode,
# ~12.3 GB free beyond 9.8 GB of weights). Each row a separate process, so the
# low-mode reload cannot leak between them:
#
#     704x704,  0 refs          S = 4384   ran
#     704x704,  2 refs @50%     S = 5352   ran
#     512x512,  3 refs          S = 5632   ran
#     768x768,  1 ref  @50%     S = 5696   ran      <- largest success,
#                                                      peaked 21.99 GB against
#                                                      a 22.07 GB ceiling
#     896x672,  2 refs @37.5%   S = 5856   failed   <- smallest failure
#     896x672,  2 refs @50%     S = 6336   failed
#     768x768,  1 ref           S = 7424   failed
#     896x672,  2 refs          S = 9920   failed
#
# Success and failure are under 3% apart in S, which is a good sign for the
# linear form. The coefficient is bracketed to 12.28 GB / 5856 = 2.147 up to
# 12.28 GB / 5696 = 2.208 MiB per token; 2.2 sits inside and puts the budget
# at ~5715 tokens here, admitting the largest success and rejecting the
# smallest failure. The 768x768 row is the strongest evidence for the whole
# model: predicted to fit with 19 tokens to spare, it peaked 0.08 GB under the
# ceiling. That same configuration is an out-of-memory failure at full scale
# (S = 7424) — shrinking the reference is what makes it run. Biased toward the safe side, because overflowing is not
# always a catchable error — CUDA graph instantiation aborts the process, and
# a failed allocation corrupts the context so everything after it in the same
# process is unreliable.
MIB_PER_STREAM_TOKEN = 2.2


def tokens_for(width: int, height: int) -> int:
    """Conditioning tokens one image contributes at this output size."""
    return max(1, (width // PIXELS_PER_TOKEN) * (height // PIXELS_PER_TOKEN))


def total_tokens(width: int, height: int, reference_count: int, scale: float) -> int:
    """Tokens for the edited image plus ``reference_count`` scaled references."""
    base = tokens_for(width, height)
    if reference_count <= 0:
        return base
    ref_w, ref_h = scaled_dimensions(width, height, scale)
    return base + reference_count * tokens_for(ref_w, ref_h)


def scaled_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    """Reference dimensions at ``scale``, snapped to the latent grid.

    Snapped to a multiple of 32 rather than 16: the encoder crops to an even
    spatial size before patching, so an odd grid silently loses a row.
    """
    if scale >= 1.0:
        return width, height

    def snap(value: int) -> int:
        scaled = int(round(value * scale / 32)) * 32
        return max(MIN_REFERENCE_EDGE, min(value, scaled))

    return snap(width), snap(height)


def stream_tokens(width: int, height: int, reference_count: int, scale: float) -> int:
    """Total tokens the transformer attends over, per denoise step.

    The joint stream is [text, output latents, conditioning images]. The image
    being edited appears twice: once as the latents being denoised, and again
    as image_paths[0] in the conditioning block at full resolution. Only the
    extra references shrink with ``scale``.
    """
    latents = tokens_for(width, height)
    conditioning = total_tokens(width, height, reference_count, scale)
    return TEXT_TOKENS + latents + conditioning


def token_budget(free_memory_gb: float) -> int:
    """Stream tokens this much spare memory carries. 0 when unbounded."""
    if free_memory_gb <= 0:
        return 0
    return int(free_memory_gb * 1024 / MIB_PER_STREAM_TOKEN)


def fits(
    width: int, height: int, reference_count: int, scale: float, free_memory_gb: float
) -> bool:
    """True when this combination is expected to fit. Unbounded always fits."""
    budget = token_budget(free_memory_gb)
    return budget <= 0 or stream_tokens(
        width, height, reference_count, scale
    ) <= budget


def largest_fitting_square(reference_count: int, free_memory_gb: float) -> int:
    """Biggest square output where ``reference_count`` references still fit.

    Used to make the "this will not fit" error actionable rather than a bare
    refusal. Returns 0 when unbounded.
    """
    if token_budget(free_memory_gb) <= 0:
        return 0

    edge = 0
    for candidate in range(256, 1537, 64):
        if fits(candidate, candidate, reference_count, AUTO_SCALES[-1], free_memory_gb):
            edge = candidate
    return edge


def choose_scale(
    width: int,
    height: int,
    reference_count: int,
    free_memory_gb: float,
) -> float:
    """Largest reference scale that fits the memory budget.

    Returns 1.0 when there is no budget to fit into (unified memory, or an
    unknown ceiling), because there the cost of guessing wrong is slowness
    rather than an aborted process. When even the smallest scale overflows it
    returns that smallest scale — the caller decides whether to refuse.
    """
    if reference_count <= 0:
        return 1.0
    if token_budget(free_memory_gb) <= 0:
        return 1.0

    for scale in AUTO_SCALES:
        if fits(width, height, reference_count, scale, free_memory_gb):
            return scale

    return AUTO_SCALES[-1]


def resolve_scale(
    configured: float | str,
    width: int,
    height: int,
    reference_count: int,
    free_memory_gb: float,
) -> float:
    """Turn the configured ``reference_scale`` into a concrete factor."""
    if isinstance(configured, str):
        if configured != "auto":
            raise ValueError(
                f"memory.reference_scale must be a number or 'auto', got {configured!r}."
            )
        return choose_scale(width, height, reference_count, free_memory_gb)
    return max(0.1, min(1.0, float(configured)))


def apply_reference_scale(scale: float) -> bool:
    """Make FLUX.2 encode references at ``scale`` of the output size.

    Patches ``_Flux2KleinEditHelpers.prepare_reference_image_conditioning``,
    reproducing mflux 0.18.0's loop with a per-image target size. Index 0 is
    the image being edited and always keeps full resolution; the rest shrink.

    A no-op at scale 1.0, and safe to call repeatedly. Returns True if the
    patch is now installed.
    """
    if scale >= 1.0:
        return _restore()

    try:
        from mflux.models.common.latent_creator.latent_creator import LatentCreator
        from mflux.models.flux2.latent_creator.flux2_latent_creator import (
            Flux2LatentCreator,
        )
        from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
            _Flux2KleinEditHelpers as helpers,
        )
        import mlx.core as mx
    except ImportError as exc:
        logger.warning("Cannot scale reference images on this mflux build: %s", exc)
        return False

    original = getattr(helpers, "_sanitos_original_prepare", None)
    if original is None:
        original = helpers.prepare_reference_image_conditioning
        helpers._sanitos_original_prepare = staticmethod(original)

    def patched(*, vae, tiling_config, image_paths=None, height, width, batch_size=1):
        if not image_paths:
            return None, None

        ref_w, ref_h = scaled_dimensions(width, height, scale)
        packed_latents_list = []
        ids_list = []

        for i, path in enumerate(image_paths):
            # Index 0 is the image being edited; it keeps full detail.
            target_w, target_h = (width, height) if i == 0 else (ref_w, ref_h)
            encoded = LatentCreator.encode_image(
                vae=vae,
                image_path=path,
                height=target_h,
                width=target_w,
                tiling_config=tiling_config,
            )
            encoded = helpers.ensure_4d_latents(encoded)
            encoded = helpers.crop_to_even_spatial(encoded)
            encoded = Flux2LatentCreator.patchify_latents(encoded)
            encoded = helpers.bn_normalize_vae_encoded_latents(encoded, vae=vae)

            packed_latents_list.append(Flux2LatentCreator.pack_latents(encoded))
            ids_list.append(Flux2LatentCreator.prepare_grid_ids(encoded, t_coord=10 + 10 * i))

        image_latents = mx.concatenate(packed_latents_list, axis=1)
        image_latent_ids = mx.concatenate(ids_list, axis=1)

        if image_latents.shape[0] != batch_size:
            image_latents = mx.broadcast_to(
                image_latents, (batch_size, image_latents.shape[1], image_latents.shape[2])
            )
        if image_latent_ids.shape[0] != batch_size:
            image_latent_ids = mx.broadcast_to(
                image_latent_ids,
                (batch_size, image_latent_ids.shape[1], image_latent_ids.shape[2]),
            )

        return image_latents, image_latent_ids

    helpers.prepare_reference_image_conditioning = staticmethod(patched)
    logger.info(
        "References encode at %.0f%% of the output size (%dx%d).",
        scale * 100,
        *scaled_dimensions(1024, 1024, scale),
    )
    return True


def _restore() -> bool:
    """Put mflux's own implementation back, if we replaced it."""
    try:
        from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
            _Flux2KleinEditHelpers as helpers,
        )
    except ImportError:
        return False

    original = getattr(helpers, "_sanitos_original_prepare", None)
    if original is not None:
        helpers.prepare_reference_image_conditioning = original
    return False
