"""Model-family adapters.

Different editing models expose meaningfully different mflux APIs — FLUX.2
rejects a negative prompt outright, wants ``guidance=1.0`` and the
``flow_match_euler_discrete`` scheduler, and names its encode step
``_encode_prompt_pair``; Qwen-Image-Edit takes a negative prompt, likes
``guidance≈4`` with the ``linear`` scheduler, and calls its encode step
``_encode_prompts_with_images``.

Keeping those differences in one table means the rest of the app — inference,
UI, memory management — stays model-agnostic, and adding a family later is a
data change rather than a refactor.

Memory footprints below are measured from the published shard sizes and drive
the startup advice, so they are per-family rather than hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class UnknownFamilyError(ValueError):
    """Raised when config.yaml names a family that does not exist."""


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """Everything that varies between editing models."""

    key: str
    label: str
    # Where the mflux pipeline class lives.
    module: str
    class_name: str
    # ModelConfig factory method name, e.g. "flux2_klein_9b".
    model_config_factory: str
    weight_definition_module: str
    weight_definition_class: str

    # Generation defaults and capabilities.
    supports_negative_prompt: bool
    default_guidance: float
    default_scheduler: str
    default_steps: int
    guidance_range: tuple[float, float]

    # The method whose lazy output must be forced before the text encoder can
    # be freed. See ImageEditor._install_encode_barrier.
    encode_method: str

    # Approximate resident size in GB, for startup advice.
    transformer_gb: float
    text_encoder_gb: float
    vae_gb: float
    # True when the text encoder stays bf16 (Qwen), which is what makes its
    # footprint so much larger than the quantised families.
    text_encoder_unquantised: bool = False

    # How packed latents map back to an image; see inference.decode_latents.
    decode_kind: str = "flux2"
    notes: str = ""
    extra_generate_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def working_set_gb(self) -> float:
        return self.transformer_gb + self.text_encoder_gb + self.vae_gb

    @property
    def denoise_peak_gb(self) -> float:
        """Peak once the text encoder has been released."""
        return self.transformer_gb + self.vae_gb

    def load_pipeline_class(self) -> type:
        module = __import__(self.module, fromlist=[self.class_name])
        return getattr(module, self.class_name)

    def load_weight_definition(self) -> Any:
        module = __import__(
            self.weight_definition_module, fromlist=[self.weight_definition_class]
        )
        return getattr(module, self.weight_definition_class)

    def load_model_config(self) -> Any:
        from mflux.models.common.config import ModelConfig

        factory: Callable[[], Any] = getattr(ModelConfig, self.model_config_factory)
        return factory()

    def build_generate_kwargs(
        self,
        *,
        seed: int,
        prompt: str,
        image_paths: list[str],
        steps: int,
        guidance: float,
        width: int | None,
        height: int | None,
        negative_prompt: str = "",
        scheduler: str | None = None,
    ) -> dict[str, Any]:
        """Assemble the keyword arguments this family's generate_image accepts.

        Passing an unsupported kwarg (a negative prompt to FLUX.2, say) is a
        hard error inside mflux, so the filtering happens here rather than at
        the call site.
        """
        kwargs: dict[str, Any] = {
            "seed": seed,
            "prompt": prompt,
            "image_paths": image_paths,
            "num_inference_steps": steps,
            "guidance": guidance,
            "scheduler": scheduler or self.default_scheduler,
        }

        if self.supports_negative_prompt:
            kwargs["negative_prompt"] = negative_prompt or None
            # Qwen reads the primary path back out for metadata.
            kwargs["image_path"] = image_paths[0] if image_paths else None
            kwargs["width"] = width
            kwargs["height"] = height
        else:
            # FLUX.2 requires concrete dimensions; it has no auto mode.
            kwargs["width"] = width or 1024
            kwargs["height"] = height or 1024

        kwargs.update(self.extra_generate_kwargs)
        return kwargs


FAMILIES: dict[str, ModelFamily] = {
    "flux2-klein-edit": ModelFamily(
        key="flux2-klein-edit",
        label="FLUX.2 Klein — image editing",
        module="mflux.models.flux2.variants.edit.flux2_klein_edit",
        class_name="Flux2KleinEdit",
        model_config_factory="flux2_klein_9b",
        weight_definition_module="mflux.models.flux2.weights.flux2_weight_definition",
        weight_definition_class="Flux2KleinWeightDefinition",
        supports_negative_prompt=False,
        default_guidance=1.0,
        default_scheduler="flow_match_euler_discrete",
        default_steps=28,
        guidance_range=(1.0, 6.0),
        encode_method="_encode_prompt_pair",
        transformer_gb=9.6,
        text_encoder_gb=8.0,
        vae_gb=0.2,
        notes=(
            "Every component is quantised, including the text encoder, so the "
            "whole model stays resident on a 24 GB machine."
        ),
    ),
    "flux2-klein-edit-4b": ModelFamily(
        key="flux2-klein-edit-4b",
        label="FLUX.2 Klein 4B — image editing",
        module="mflux.models.flux2.variants.edit.flux2_klein_edit",
        class_name="Flux2KleinEdit",
        model_config_factory="flux2_klein_4b",
        weight_definition_module="mflux.models.flux2.weights.flux2_weight_definition",
        weight_definition_class="Flux2KleinWeightDefinition",
        supports_negative_prompt=False,
        default_guidance=1.0,
        default_scheduler="flow_match_euler_discrete",
        default_steps=28,
        guidance_range=(1.0, 6.0),
        encode_method="_encode_prompt_pair",
        transformer_gb=3.2,
        text_encoder_gb=3.3,
        vae_gb=0.2,
        notes="Smallest and fastest; lower fidelity than the 9B.",
    ),
    "qwen-image-edit": ModelFamily(
        key="qwen-image-edit",
        label="Qwen-Image-Edit — image editing",
        module="mflux.models.qwen.variants.edit.qwen_image_edit",
        class_name="QwenImageEdit",
        model_config_factory="qwen_image_edit",
        weight_definition_module="mflux.models.qwen.weights.qwen_weight_definition",
        weight_definition_class="QwenWeightDefinition",
        supports_negative_prompt=True,
        default_guidance=4.0,
        default_scheduler="linear",
        default_steps=25,
        guidance_range=(1.0, 10.0),
        encode_method="_encode_prompts_with_images",
        transformer_gb=16.6,
        text_encoder_gb=15.5,
        vae_gb=0.25,
        text_encoder_unquantised=True,
        decode_kind="qwen",
        notes=(
            "mflux keeps this text encoder at bf16 (skip_quantization) because "
            "quantising it degrades prompt understanding — so the working set "
            "is ~32 GB regardless of the transformer's bit width. Needs 40 GB+ "
            "of memory to run without swapping."
        ),
    ),
}


def get_family(key: str) -> ModelFamily:
    try:
        return FAMILIES[key]
    except KeyError as exc:
        raise UnknownFamilyError(
            f"Unknown model family {key!r}. Available: {', '.join(sorted(FAMILIES))}"
        ) from exc
