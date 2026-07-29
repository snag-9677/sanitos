"""Reference-image token arithmetic.

The thing worth guarding is that the cost model matches how mflux actually
prices conditioning — per token at the output resolution, not per pixel of the
supplied file. Getting that backwards produces a "compressor" that saves
nothing, which is exactly the trap this replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.references import (  # noqa: E402
    MIN_REFERENCE_EDGE,
    stream_tokens,
    token_budget,
    choose_scale,
    fits,
    largest_fitting_square,
    resolve_scale,
    scaled_dimensions,
    tokens_for,
    total_tokens,
)

# Free memory beyond FLUX.2's 9.8 GB denoise weights on a 24 GB card.
A5000_FREE_GB = 24.0 * 0.92 - 9.8


def test_tokens_follow_the_latent_grid() -> None:
    assert tokens_for(512, 512) == 32 * 32
    assert tokens_for(896, 672) == 56 * 42


def test_halving_a_reference_quarters_its_cost() -> None:
    """The whole point: tokens scale with area, so half scale is a 4x saving."""
    full = total_tokens(512, 512, reference_count=1, scale=1.0)
    half = total_tokens(512, 512, reference_count=1, scale=0.5)

    source_only = tokens_for(512, 512)
    assert full - source_only == 1024
    assert half - source_only == 256


def test_three_small_references_beat_one_large_one() -> None:
    """Cost is driven by resolution, which is unintuitive and worth pinning."""
    three_small = total_tokens(512, 512, reference_count=3, scale=1.0)
    one_large = total_tokens(768, 768, reference_count=1, scale=1.0)
    assert three_small < one_large


def test_the_cost_model_matches_every_measurement() -> None:
    """Calibration guard: a token-count model passed these and was still wrong.

    Each row was actually run on a 24 GB RTX A5000. The model must admit the
    ones that ran and reject the ones that aborted.
    """
    measured = [
        # (width, height, refs, scale, did it run) — each a separate process
        (512, 512, 0, 1.0, True),
        (512, 512, 2, 1.0, True),
        (704, 704, 0, 1.0, True),
        (704, 704, 2, 0.5, True),
        (512, 512, 3, 1.0, True),     # S=5632
        (768, 768, 1, 0.5, True),     # largest success, S=5696, peaked 21.99/22.07 GB
        (896, 672, 2, 0.375, False),  # smallest failure, S=5856
        (896, 672, 2, 0.5, False),
        (768, 768, 1, 1.0, False),    # same edit at full scale — this is what
                                      # auto-scaling rescues
        (896, 672, 2, 1.0, False),
    ]
    for width, height, refs, scale, ran in measured:
        assert fits(width, height, refs, scale, A5000_FREE_GB) is ran, (
            f"{width}x{height} with {refs} refs at {scale} "
            f"(S={stream_tokens(width, height, refs, scale)}) "
            f"vs budget {token_budget(A5000_FREE_GB)}"
        )


def test_the_stream_counts_text_and_the_source_twice() -> None:
    """The edited image is both the latents and image_paths[0].

    Omitting either that second copy or the padded text block is what made a
    linear model look wrong and sent this down a quadratic dead end.
    """
    # 512x512 -> 1024 latent tokens, same again as conditioning, plus text.
    assert stream_tokens(512, 512, 0, 1.0) == 512 + 1024 + 1024
    assert stream_tokens(512, 512, 3, 1.0) == 512 + 1024 + (1024 + 3 * 1024)


def test_auto_leaves_a_cheap_edit_alone() -> None:
    assert choose_scale(512, 512, reference_count=1, free_memory_gb=A5000_FREE_GB) == 1.0


def test_no_hard_ceiling_means_no_shrinking() -> None:
    """On unified memory an overshoot is slow, not fatal — don't degrade."""
    assert choose_scale(1024, 1024, reference_count=3, free_memory_gb=0.0) == 1.0


def test_a_suggested_resolution_actually_fits() -> None:
    """The refusal message names a size; it must be true."""
    for refs in (1, 2, 3):
        edge = largest_fitting_square(refs, A5000_FREE_GB)
        assert edge > 0
        assert fits(edge, edge, refs, 0.25, A5000_FREE_GB)
        assert not fits(edge + 64, edge + 64, refs, 0.25, A5000_FREE_GB)


def test_scaling_snaps_to_the_latent_grid_and_has_a_floor() -> None:
    width, height = scaled_dimensions(1024, 768, 0.5)
    assert width % 32 == 0 and height % 32 == 0

    tiny_w, tiny_h = scaled_dimensions(512, 512, 0.1)
    assert tiny_w >= MIN_REFERENCE_EDGE and tiny_h >= MIN_REFERENCE_EDGE


def test_scaling_never_enlarges_a_reference() -> None:
    assert scaled_dimensions(320, 320, 0.9) <= (320, 320)


def test_explicit_scale_overrides_auto() -> None:
    assert resolve_scale(0.5, 896, 672, 2, A5000_FREE_GB) == 0.5
    assert resolve_scale(1.0, 896, 672, 2, A5000_FREE_GB) == 1.0


def test_a_nonsense_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_scale("aggressive", 512, 512, 1, A5000_FREE_GB)


def test_zero_references_cost_only_the_source() -> None:
    assert total_tokens(512, 512, reference_count=0, scale=0.25) == tokens_for(512, 512)
