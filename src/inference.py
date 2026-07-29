"""Text-guided image editing: the generation loop, previews, and cancellation.

One edit is 30-90 seconds of blocking GPU work, so this module exposes three
things the UI needs to stay honest during that window:

* **progress** — step count plus an ETA derived from measured per-step time
* **previews** — the latents decoded at a few early steps, so a misunderstood
  instruction is visible ~60 seconds before the final image would be
* **cancellation** — a stop that keeps the half-denoised result instead of
  throwing the work away
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from .device import active_memory_gb, peak_memory_gb, reset_peak_memory
from .model_loader import EditModel, ModelLoadError
from .utils import coerce_seed, estimate_remaining, format_duration, snap_dimension

logger = logging.getLogger(__name__)

# Fractions of the run at which to decode a preview. Front-loaded on purpose:
# an early look is what lets someone abandon a bad prompt, and a preview at 95%
# tells you nothing you won't see for free in a moment.
PREVIEW_FRACTIONS = (0.20, 0.45, 0.75)
# Below this many steps the decode overhead isn't worth it.
MIN_STEPS_FOR_PREVIEW = 8


class InferenceError(RuntimeError):
    """Raised when generation fails for a reason worth showing the user."""


@dataclass(slots=True)
class EditRequest:
    """Everything needed to run one edit."""

    images: list[Image.Image]
    instruction: str
    seed: int
    steps: int = 25
    guidance: float = 4.0
    width: int | None = None
    height: int | None = None
    negative_prompt: str = ""
    scheduler: str = ""

    def __post_init__(self) -> None:
        if not self.images:
            raise InferenceError("An image is required before editing.")
        if not self.instruction.strip():
            raise InferenceError("Enter an editing instruction first.")
        self.steps = max(1, int(self.steps))
        self.guidance = float(self.guidance)
        self.seed = coerce_seed(self.seed)
        self.width = snap_dimension(self.width)
        self.height = snap_dimension(self.height)


@dataclass(slots=True)
class EditProgress:
    """A single progress tick handed to the UI."""

    step: int
    total: int
    elapsed: float
    eta: str = ""
    preview: Image.Image | None = None
    message: str = ""

    @property
    def fraction(self) -> float:
        return min(1.0, self.step / self.total) if self.total else 0.0

    def describe(self) -> str:
        if self.message:
            return self.message
        base = f"Step {self.step}/{self.total}"
        return f"{base} · {self.eta}" if self.eta else base


@dataclass(slots=True)
class EditResult:
    """The outcome of one edit."""

    image: Image.Image
    seed: int
    steps: int
    guidance: float
    width: int
    height: int
    duration: float
    instruction: str
    negative_prompt: str = ""
    interrupted: bool = False
    completed_steps: int | None = None
    peak_memory_gb: float = 0.0

    def caption(self) -> str:
        parts = [f"{self.width}x{self.height}", f"{self.steps} steps"]
        if self.interrupted and self.completed_steps is not None:
            parts[-1] = f"stopped at {self.completed_steps}/{self.steps} steps"
        parts += [f"guidance {self.guidance:g}", f"seed {self.seed}"]
        parts.append(format_duration(self.duration))
        return " · ".join(parts)


ProgressFn = Callable[[EditProgress], None]


class _GenerationCallback:
    """Bridges mflux's callback protocol to progress, previews, and stop.

    mflux converts a ``KeyboardInterrupt`` raised inside the denoise loop into
    a ``StopImageGenerationException`` after handing the current latents to
    ``call_interrupt``. Raising it from here is therefore the supported way to
    cancel while keeping the partial result.
    """

    def __init__(
        self,
        model_handle: EditModel,
        total_steps: int,
        on_progress: ProgressFn | None,
        cancel_event: threading.Event | None,
        enable_previews: bool,
    ) -> None:
        self.model_handle = model_handle
        self.total_steps = total_steps
        self.on_progress = on_progress
        self.cancel_event = cancel_event
        self.enable_previews = enable_previews and total_steps >= MIN_STEPS_FOR_PREVIEW

        self.started = time.perf_counter()
        self.interrupted_latents: Any = None
        self.interrupted_config: Any = None
        self.completed_steps = 0
        self.preview_seconds = 0.0

        self._preview_steps = self._plan_previews(total_steps)

    @staticmethod
    def _plan_previews(total: int) -> set[int]:
        if total < MIN_STEPS_FOR_PREVIEW:
            return set()
        steps = {max(1, int(round(total * f))) for f in PREVIEW_FRACTIONS}
        return {s for s in steps if 0 < s < total}

    # ------------------------------------------------------- mflux protocol

    def call_before_loop(
        self, seed, prompt, latents, config, canny_image=None, depth_image=None
    ) -> None:
        """Fires after prompt encoding, before the first denoise step.

        This is exactly the moment the text encoder becomes dead weight for the
        rest of the run, so ``low`` mode releases its 15.5 GB here.
        """
        # Normally a no-op: the encode barrier in ImageEditor already evicted
        # after forcing the embeddings. This is the fallback if that patch
        # could not be installed against this mflux version.
        if self.model_handle.memory_mode == "low":
            self.model_handle.evict_text_encoder()
        logger.debug("Entering denoise loop at %.1f GB active", active_memory_gb())
        self._emit(EditProgress(0, self.total_steps, 0.0, message="Denoising…"))

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise KeyboardInterrupt

        self.completed_steps = t + 1
        # Exclude preview decoding from the ETA, or the estimate jumps around.
        elapsed = time.perf_counter() - self.started - self.preview_seconds

        preview = None
        if self.completed_steps in self._preview_steps:
            preview = self._decode_preview(latents, config)

        self._emit(
            EditProgress(
                step=self.completed_steps,
                total=self.total_steps,
                elapsed=elapsed,
                eta=estimate_remaining(elapsed, self.completed_steps, self.total_steps),
                preview=preview,
            )
        )

    def call_interrupt(self, t, seed, prompt, latents, config, time_steps) -> None:
        """Keep the latents so a stopped run still yields an image."""
        self.interrupted_latents = latents
        self.interrupted_config = config
        self.completed_steps = t

    # ---------------------------------------------------------------- helpers

    def _decode_preview(self, latents: Any, config: Any) -> Image.Image | None:
        started = time.perf_counter()
        try:
            image = decode_latents(self.model_handle, latents, config)
        except Exception as exc:  # noqa: BLE001 - a preview must never fail a run
            logger.debug("Preview decode failed at step %s: %s", self.completed_steps, exc)
            return None
        finally:
            self.preview_seconds += time.perf_counter() - started
        return image

    def _emit(self, progress: EditProgress) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress(progress)
        except Exception as exc:  # noqa: BLE001 - UI errors must not kill the run
            logger.debug("Progress callback raised: %s", exc)


def decode_latents(model_handle: EditModel, latents: Any, config: Any) -> Image.Image:
    """Turn packed latents into a PIL image using the model's VAE.

    Used for mid-run previews and for salvaging a stopped generation. Latent
    packing differs by family, so the unpack step is dispatched on
    ``family.decode_kind``.

    Uses the already-resident model rather than ``ensure_loaded()``. In ``low``
    mode the text encoder may have just been evicted on purpose, and going
    through ensure_loaded() would drag it straight back in — on every preview
    decode, mid-generation. Only the VAE is needed here.
    """
    from mflux.utils.image_util import ImageUtil

    model = model_handle.live_model
    kind = model_handle.family.decode_kind

    if kind == "qwen":
        from mflux.models.common.vae.vae_util import VAEUtil
        from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator

        unpacked = QwenLatentCreator.unpack_latents(
            latents=latents, height=config.height, width=config.width
        )
        decoded = VAEUtil.decode(
            vae=model.vae,
            latent=unpacked,
            tiling_config=getattr(model, "tiling_config", None),
        )
    else:
        # FLUX.2 keeps latents as (batch, patches, channels); the patch grid is
        # height/16 x width/16 (VAE stride 8, then 2x2 patching).
        latent_height = config.height // 16
        latent_width = config.width // 16
        if latent_height * latent_width != latents.shape[1]:
            raise ValueError(
                f"Latent grid {latent_height}x{latent_width} does not match "
                f"{latents.shape[1]} patches; cannot decode."
            )
        packed = latents.reshape(
            latents.shape[0], latent_height, latent_width, latents.shape[-1]
        ).transpose(0, 3, 1, 2)
        decoded = model.vae.decode_packed_latents(packed)

    normalised = ImageUtil._denormalize(decoded)
    return ImageUtil._numpy_to_pil(ImageUtil._to_numpy(normalised))


class ImageEditor:
    """Runs text-guided edits against the loaded model."""

    def __init__(self, model_handle: EditModel, *, enable_previews: bool = True) -> None:
        self.model_handle = model_handle
        self.enable_previews = enable_previews
        self._lock = threading.Lock()
        self.last_duration: float | None = None

    @property
    def is_busy(self) -> bool:
        return self._lock.locked()

    def edit(
        self,
        request: EditRequest,
        *,
        on_progress: ProgressFn | None = None,
        cancel_event: threading.Event | None = None,
    ) -> EditResult:
        """Run one edit. Blocks for the duration of the generation.

        Raises:
            InferenceError: on any failure worth showing the user.
        """
        # One GPU, one job: a second concurrent edit would thrash unified memory.
        if not self._lock.acquire(blocking=False):
            raise InferenceError("An edit is already running. Wait for it to finish.")

        try:
            return self._run(request, on_progress, cancel_event)
        finally:
            self._lock.release()

    def _run(
        self,
        request: EditRequest,
        on_progress: ProgressFn | None,
        cancel_event: threading.Event | None,
    ) -> EditResult:
        if on_progress:
            on_progress(
                EditProgress(0, request.steps, 0.0, message="Preparing model…")
            )

        try:
            model = self.model_handle.ensure_loaded(
                progress=lambda p: on_progress(
                    EditProgress(
                        0,
                        request.steps,
                        0.0,
                        message=f"{p.stage}{f' — {p.detail}' if p.detail else ''}",
                    )
                )
                if on_progress
                else None
            )
        except ModelLoadError as exc:
            raise InferenceError(str(exc)) from exc

        reset_peak_memory()
        callback = _GenerationCallback(
            model_handle=self.model_handle,
            total_steps=request.steps,
            on_progress=on_progress,
            cancel_event=cancel_event,
            enable_previews=self.enable_previews,
        )

        registry = getattr(model, "callbacks", None)
        if registry is None:
            raise InferenceError("Model is missing its callback registry.")
        self._reset_registry(registry)
        registry.register(callback)

        started = time.perf_counter()
        restore = self._install_encode_barrier(model)
        try:
            with tempfile.TemporaryDirectory(prefix="qwen-edit-") as tmp:
                # mflux takes image paths, not PIL objects.
                image_paths = self._write_inputs(request.images, Path(tmp))
                result = self._generate(model, request, image_paths, callback, started)
        finally:
            restore()
            self._reset_registry(registry)
            self.model_handle.after_generation()

        self.last_duration = result.duration
        logger.info(
            "Edit finished in %s (%dx%d, %d steps, peak %.1f GB)",
            format_duration(result.duration),
            result.width,
            result.height,
            result.steps,
            result.peak_memory_gb,
        )
        return result

    def _generate(
        self,
        model: Any,
        request: EditRequest,
        image_paths: list[str],
        callback: _GenerationCallback,
        started: float,
    ) -> EditResult:
        try:
            from mflux.utils.exceptions import StopImageGenerationException
        except ImportError:  # pragma: no cover - mflux always ships this
            StopImageGenerationException = RuntimeError  # type: ignore[assignment]

        # Families differ in which kwargs they accept — FLUX.2 errors on a
        # negative prompt, Qwen requires one path for metadata — so the family
        # assembles the call.
        kwargs = self.model_handle.family.build_generate_kwargs(
            seed=request.seed,
            prompt=request.instruction,
            image_paths=image_paths,
            steps=request.steps,
            guidance=request.guidance,
            width=request.width,
            height=request.height,
            negative_prompt=request.negative_prompt,
            scheduler=request.scheduler or None,
        )

        try:
            generated = model.generate_image(**kwargs)
        except StopImageGenerationException:
            return self._salvage(request, callback, started)
        except KeyboardInterrupt:
            # Interrupted outside the guarded loop (e.g. during VAE decode).
            return self._salvage(request, callback, started)
        except MemoryError as exc:
            raise InferenceError(
                "Ran out of memory. Try a smaller resolution (512 or 768), fewer "
                "steps, or set memory.mode: low in config.yaml."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(self._explain(exc)) from exc

        image = getattr(generated, "image", None)
        if image is None:
            raise InferenceError("Generation returned no image.")

        duration = time.perf_counter() - started
        return EditResult(
            image=image,
            seed=request.seed,
            steps=request.steps,
            guidance=request.guidance,
            width=image.width,
            height=image.height,
            duration=duration,
            instruction=request.instruction,
            negative_prompt=request.negative_prompt,
            peak_memory_gb=peak_memory_gb(),
        )

    def _salvage(
        self, request: EditRequest, callback: _GenerationCallback, started: float
    ) -> EditResult:
        """Decode whatever the stopped run had reached.

        A partially denoised image is usually good enough to judge the
        direction of an edit, and occasionally good enough to keep.
        """
        if callback.interrupted_latents is None:
            raise InferenceError("Edit stopped before an image could be produced.")

        try:
            image = decode_latents(
                self.model_handle, callback.interrupted_latents, callback.interrupted_config
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Edit stopped and the partial image failed to decode: {exc}") from exc

        duration = time.perf_counter() - started
        logger.info("Edit stopped at step %d/%d", callback.completed_steps, request.steps)
        return EditResult(
            image=image,
            seed=request.seed,
            steps=request.steps,
            guidance=request.guidance,
            width=image.width,
            height=image.height,
            duration=duration,
            instruction=request.instruction,
            negative_prompt=request.negative_prompt,
            interrupted=True,
            completed_steps=callback.completed_steps,
            peak_memory_gb=peak_memory_gb(),
        )

    def _install_encode_barrier(self, model: Any) -> Callable[[], None]:
        """Force the text encoder to run, then free it, before denoising starts.

        This is the single most important optimisation in the app on a machine
        with less memory than the model.

        MLX is lazily evaluated. ``_encode_prompts_with_images`` returns an
        *unevaluated graph*, not embeddings — and that graph holds a reference
        to every one of the text encoder's 15.5 GB of weights. Dropping the
        module afterwards therefore frees nothing: the weights stay alive until
        the graph is finally forced, which happens at the first ``mx.eval()``
        inside the denoise loop, exactly when the 16.6 GB transformer is also
        materialising. Both are resident at once, peak hits ~32 GB, and a 24 GB
        machine swaps for the entire run.

        Measured on an M5/24 GB: 116 s/step without this barrier.

        Inserting an explicit ``mx.eval`` on the embeddings collapses the graph
        while the transformer is still untouched, so the encoder's weights can
        actually be released. The embeddings themselves are a few MB.

        Returns a callable that restores the original method.
        """
        method = self.model_handle.family.encode_method
        original = getattr(model, method, None)
        if original is None or self.model_handle.memory_mode == "off":
            return lambda: None

        try:
            import mlx.core as mx
        except ImportError:
            return lambda: None

        handle = self.model_handle

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            try:
                arrays = [a for a in result if a is not None]
                mx.eval(*arrays)  # collapse the graph; weights become droppable
            except Exception as exc:  # noqa: BLE001 - never break generation
                logger.debug("Could not force prompt-embedding evaluation: %s", exc)
                return result

            if handle.memory_mode == "low":
                handle.evict_text_encoder()
            return result

        setattr(model, method, wrapped)

        def restore() -> None:
            try:
                setattr(model, method, original)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not restore the encode method: %s", exc)

        return restore

    @staticmethod
    def _write_inputs(images: Sequence[Image.Image], directory: Path) -> list[str]:
        paths: list[str] = []
        for index, image in enumerate(images):
            path = directory / f"input-{index}.png"
            image.convert("RGB").save(path, format="PNG")
            paths.append(str(path))
        return paths

    @staticmethod
    def _reset_registry(registry: Any) -> None:
        """Drop callbacks from previous runs so they don't accumulate."""
        for bucket in ("before_loop", "in_loop", "after_loop", "interrupt"):
            listeners = getattr(registry, bucket, None)
            if isinstance(listeners, list):
                listeners.clear()

    @staticmethod
    def _explain(exc: Exception) -> str:
        text = str(exc)
        lowered = text.lower()
        if "out of memory" in lowered or "insufficient" in lowered:
            return (
                f"Out of memory: {text}\n"
                "Try 512 or 768 resolution, fewer steps, or memory.mode: low."
            )
        if "shape" in lowered or "broadcast" in lowered:
            return (
                f"Tensor shape error: {text}\n"
                "This usually means an unsupported resolution — width and height "
                "must be multiples of 16."
            )
        return f"Edit failed: {text}"


def memory_snapshot() -> str:
    """One-line memory readout for the UI footer."""
    return f"MLX {active_memory_gb():.1f} GB active · {peak_memory_gb():.1f} GB peak"
