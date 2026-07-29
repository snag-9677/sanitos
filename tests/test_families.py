"""Model-family adapter tests.

The failure mode these guard is unforgiving: passing a kwarg the pipeline does
not accept (a negative prompt to FLUX.2) is a hard error inside mflux, and the
memory numbers here drive whether the app decides it needs to evict anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.families import FAMILIES, UnknownFamilyError, get_family  # noqa: E402


def test_default_family_fits_a_24gb_machine() -> None:
    """The whole reason for switching off Qwen."""
    family = get_family("flux2-klein-edit")
    assert family.working_set_gb < 22, "default model must fit in 22 GB"
    assert not family.text_encoder_unquantised


def test_qwen_is_recorded_as_not_fitting() -> None:
    """Qwen stays selectable but must be honestly labelled as too large."""
    family = get_family("qwen-image-edit")
    assert family.working_set_gb > 30
    assert family.text_encoder_unquantised, (
        "the bf16 text encoder is precisely why Qwen does not fit"
    )
    assert "skip_quantization" in family.notes


def test_unknown_family_is_rejected_clearly() -> None:
    with pytest.raises(UnknownFamilyError) as excinfo:
        get_family("stable-diffusion")
    assert "flux2-klein-edit" in str(excinfo.value), "error should list valid options"


def test_flux2_never_receives_a_negative_prompt() -> None:
    """mflux errors out if FLUX.2 is handed one, so it must be filtered here."""
    family = get_family("flux2-klein-edit")
    kwargs = family.build_generate_kwargs(
        seed=1, prompt="make it blue", image_paths=["/tmp/a.png"],
        steps=28, guidance=1.0, width=768, height=512,
        negative_prompt="blurry, ugly",
    )

    assert "negative_prompt" not in kwargs
    assert "image_path" not in kwargs
    assert kwargs["scheduler"] == "flow_match_euler_discrete"
    assert kwargs["width"] == 768 and kwargs["height"] == 512


def test_flux2_gets_concrete_dimensions() -> None:
    """FLUX.2 has no auto-size mode; None would blow up downstream."""
    kwargs = get_family("flux2-klein-edit").build_generate_kwargs(
        seed=1, prompt="x", image_paths=["/tmp/a.png"],
        steps=28, guidance=1.0, width=None, height=None,
    )
    assert isinstance(kwargs["width"], int) and kwargs["width"] > 0
    assert isinstance(kwargs["height"], int) and kwargs["height"] > 0


def test_qwen_keeps_its_negative_prompt_and_metadata_path() -> None:
    kwargs = get_family("qwen-image-edit").build_generate_kwargs(
        seed=1, prompt="make it blue", image_paths=["/tmp/a.png"],
        steps=25, guidance=4.0, width=None, height=None,
        negative_prompt="blurry",
    )

    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["image_path"] == "/tmp/a.png"
    assert kwargs["scheduler"] == "linear"
    # Qwen derives its own dimensions when these are None.
    assert kwargs["width"] is None and kwargs["height"] is None


def test_blank_negative_prompt_becomes_none_for_qwen() -> None:
    kwargs = get_family("qwen-image-edit").build_generate_kwargs(
        seed=1, prompt="x", image_paths=["/tmp/a.png"],
        steps=25, guidance=4.0, width=None, height=None, negative_prompt="",
    )
    assert kwargs["negative_prompt"] is None


def test_explicit_scheduler_overrides_the_family_default() -> None:
    kwargs = get_family("flux2-klein-edit").build_generate_kwargs(
        seed=1, prompt="x", image_paths=["/tmp/a.png"], steps=4,
        guidance=1.0, width=512, height=512, scheduler="custom",
    )
    assert kwargs["scheduler"] == "custom"


def test_denoise_peak_excludes_the_text_encoder() -> None:
    family = get_family("qwen-image-edit")
    assert family.denoise_peak_gb < family.working_set_gb
    assert family.denoise_peak_gb == pytest.approx(
        family.transformer_gb + family.vae_gb
    )


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_every_family_resolves_its_mflux_classes(key: str) -> None:
    """Catches a renamed mflux class before it becomes a runtime crash."""
    family = get_family(key)
    assert isinstance(family.load_pipeline_class(), type)
    weight_definition = family.load_weight_definition()
    assert weight_definition.get_download_patterns()
    assert family.load_model_config() is not None


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_every_family_names_a_real_encode_method(key: str) -> None:
    """The memory fix depends on this attribute existing on the pipeline."""
    family = get_family(key)
    assert hasattr(family.load_pipeline_class(), family.encode_method), (
        f"{key}: {family.encode_method} is missing — the encode barrier would "
        f"silently do nothing and memory would spike"
    )


@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_every_family_has_sane_defaults(key: str) -> None:
    family = get_family(key)
    low, high = family.guidance_range
    assert low <= family.default_guidance <= high
    assert family.default_steps >= 1
    assert family.decode_kind in {"qwen", "flux2"}
