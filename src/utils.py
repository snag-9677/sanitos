"""Shared helpers: image IO, output naming, seeds, and formatting."""

from __future__ import annotations

import json
import logging
import random
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

MAX_SEED = 2**32 - 1
# Qwen's VAE downsamples by 8 and the transformer patches 2x2, so both edges
# must be multiples of 16 or the latent grid will not tile evenly.
DIMENSION_MULTIPLE = 16


class ImageError(RuntimeError):
    """Raised when an image cannot be read, written, or is unusable."""


def random_seed() -> int:
    return random.randint(0, MAX_SEED)


def coerce_seed(value: Any) -> int:
    """Turn UI input into a valid seed, generating one for blank/invalid input."""
    if value is None or value == "":
        return random_seed()
    try:
        seed = int(value)
    except (TypeError, ValueError):
        logger.debug("Invalid seed %r, generating a random one.", value)
        return random_seed()
    if seed < 0:
        return random_seed()
    return seed % (MAX_SEED + 1)


def snap_dimension(value: int | None) -> int | None:
    """Round a dimension down to a multiple of 16 (minimum 256)."""
    if value is None:
        return None
    return max(256, int(value) // DIMENSION_MULTIPLE * DIMENSION_MULTIPLE)


def slugify(text: str, max_length: int = 48) -> str:
    """Make a filesystem-safe fragment out of a prompt."""
    normalised = unicodedata.normalize("NFKD", text)
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^\w\s-]", "", ascii_only).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:max_length].strip("-") or "edit"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def format_duration(seconds: float) -> str:
    """Render a duration the way a person would say it."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s"


def load_image(source: str | Path | Image.Image) -> Image.Image:
    """Load an image as RGB, honouring EXIF orientation.

    Phone photos carry an EXIF rotation flag that PIL ignores by default; not
    applying it silently sends a sideways image to the model.
    """
    if isinstance(source, Image.Image):
        image = source
    else:
        path = Path(source)
        if not path.is_file():
            raise ImageError(f"Image not found: {path}")
        try:
            image = Image.open(path)
        except (OSError, ValueError) as exc:
            raise ImageError(f"Could not read image {path}: {exc}") from exc

    try:
        image = ImageOps.exif_transpose(image)
    except Exception as exc:  # noqa: BLE001 - malformed EXIF must not be fatal
        logger.debug("EXIF transpose failed, using image as-is: %s", exc)

    return image.convert("RGB")


def save_image(
    image: Image.Image,
    directory: Path,
    prompt: str,
    *,
    fmt: str = "png",
    metadata: dict[str, Any] | None = None,
    prefix: str = "edit",
) -> Path:
    """Write an image (and optional sidecar JSON) to the output directory."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp()}-{prefix}-{slugify(prompt)}"
    suffix = "jpg" if fmt in {"jpg", "jpeg"} else "png"
    path = directory / f"{stem}.{suffix}"

    # Never clobber: an identical timestamp+prompt within the same second is
    # unlikely but a silent overwrite would lose a minute of GPU time.
    counter = 1
    while path.exists():
        path = directory / f"{stem}-{counter}.{suffix}"
        counter += 1

    try:
        if suffix == "jpg":
            image.save(path, format="JPEG", quality=95, subsampling=0)
        else:
            image.save(path, format="PNG")
    except (OSError, ValueError) as exc:
        raise ImageError(f"Could not write image to {path}: {exc}") from exc

    if metadata:
        try:
            path.with_suffix(".json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:  # non-fatal: the image itself is already safe
            logger.warning("Could not write metadata sidecar for %s: %s", path, exc)

    return path


def thumbnail(image: Image.Image, size: int = 256) -> Image.Image:
    """Downscale a copy of the image to fit a square box."""
    copy = image.copy()
    copy.thumbnail((size, size), Image.Resampling.LANCZOS)
    return copy


def dimensions_for_megapixels(image: Image.Image, megapixels: float) -> tuple[int, int]:
    """Size to a pixel budget while preserving the source aspect ratio.

    Forcing a square output on a non-square photo stretches it, so aspect-
    preserving sizing is the sane default; both edges snap to the 16-pixel
    latent grid.
    """
    budget = max(0.05, float(megapixels)) * 1_000_000
    ratio = image.width / image.height if image.height else 1.0
    width = (budget * ratio) ** 0.5
    height = width / ratio if ratio else width
    return snap_dimension(int(width)) or 512, snap_dimension(int(height)) or 512


def describe_dimensions(width: int | None, height: int | None, image: Image.Image) -> str:
    """Describe the resolution an edit will actually run at."""
    if width and height:
        return f"{width}x{height}"
    auto_w, auto_h = dimensions_for_megapixels(image, 1.0)
    return f"{auto_w}x{auto_h} (auto)"


def estimate_remaining(elapsed: float, done: int, total: int) -> str:
    """ETA from the mean per-step time so far.

    MLX step times are very stable after the first couple of steps, so a simple
    mean is accurate enough to be worth showing.
    """
    if done <= 0 or done >= total:
        return ""
    per_step = elapsed / done
    return f"~{format_duration(per_step * (total - done))} left"
