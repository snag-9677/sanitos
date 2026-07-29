"""Loading and memory management for the editing model.

Memory is the whole problem when the model is larger than the machine. The
per-family footprints live in ``families.py``; the two that matter here:

    FLUX.2 Klein 9B (8-bit)   transformer 9.6 + text encoder 8.0 + vae 0.2
                              = ~17.9 GB, everything quantised, fully resident
                              on a 24 GB machine.

    Qwen-Image-Edit (6-bit)   transformer 16.6 + text encoder 15.5 (bf16, mflux
                              marks it skip_quantization) + vae 0.25 = ~32.4 GB.
                              Needs 40 GB+ or it swaps.

When the working set does not fit, ``low`` mode evicts the text encoder once
the prompt is encoded and lazily reloads it next turn, bounding the denoise
phase to the transformer alone. Crucially the eviction only works if the
encoder's lazily-evaluated output graph is forced first — see
``ImageEditor._install_encode_barrier``.

mflux's own MemorySaver does the eviction but never reloads, so a second edit
with a new prompt would crash. This module owns that lifecycle instead.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .families import ModelFamily, get_family

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the model cannot be downloaded, found, or initialised."""


@dataclass(slots=True)
class LoadProgress:
    """Coarse load-stage reporting, so the UI can say something useful."""

    stage: str
    detail: str = ""


ProgressFn = Callable[[LoadProgress], None]


def configure_hf_cache(cache_dir: Path) -> None:
    """Point HuggingFace at the project-local model directory.

    Must run before ``huggingface_hub`` is imported anywhere, because the
    library reads these into module-level constants at import time.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    # High-throughput Xet transfer; ~32 GB is painful without it. (This
    # replaced HF_HUB_ENABLE_HF_TRANSFER, which now warns on newer hub versions.)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    # Tokenizers warns loudly about forking under Gradio's thread pool.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class EditModel:
    """Lazily-loaded, thread-safe handle to an mflux editing pipeline.

    Which pipeline is decided by the configured :class:`ModelFamily`.
    """

    def __init__(
        self,
        repo_id: str,
        cache_dir: Path,
        *,
        family: str | ModelFamily = "flux2-klein-edit",
        label: str | None = None,
        memory_mode: str = "low",
        cache_limit_bytes: int | None = 1024**3,
        vae_tiling: bool = True,
        quantize: int | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.family = family if isinstance(family, ModelFamily) else get_family(family)
        self._label = label
        self.memory_mode = memory_mode
        self.cache_limit_bytes = cache_limit_bytes
        self.vae_tiling = vae_tiling
        self.quantize = quantize
        self.lora_paths = list(lora_paths or [])
        self.lora_scales = list(lora_scales or [])

        self._model: Any | None = None
        self._model_root: Path | None = None
        self._lock = threading.RLock()
        self._load_seconds: float | None = None
        self._encoder_evicted = False

        configure_hf_cache(cache_dir)

    # ------------------------------------------------------------ properties

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def live_model(self) -> Any:
        """The resident model, without triggering a load or an encoder reload.

        Callers mid-generation (preview decoding, salvaging a stopped run) need
        the VAE only. Routing them through :meth:`ensure_loaded` would undo the
        text-encoder eviction that ``low`` mode just performed.
        """
        model = self._model
        if model is None:
            raise ModelLoadError("The model is not loaded.")
        return model

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    @property
    def label(self) -> str:
        return self._label or self.family.label

    # --------------------------------------------------------------- loading

    def ensure_loaded(self, progress: ProgressFn | None = None) -> Any:
        """Return the live model, loading it on first use.

        Loading takes minutes on a cold cache (~32 GB download) and tens of
        seconds warm, so it is deliberately deferred until the first edit.
        """
        with self._lock:
            if self._model is not None:
                self._ensure_text_encoder(progress)
                return self._model

            self._report(progress, "Locating model", self.repo_id)
            started = time.perf_counter()

            try:
                pipeline_class = self.family.load_pipeline_class()
                model_config = self.family.load_model_config()
            except ImportError as exc:
                raise ModelLoadError(
                    f"Could not load the {self.family.label} pipeline. Is mflux "
                    f"installed and recent enough? Run: pip install -r requirements.txt\n{exc}"
                ) from exc

            self._apply_mlx_limits()
            self._report(
                progress,
                "Loading weights",
                f"~{self.family.working_set_gb:.0f} GB on first run — "
                "this downloads once, then caches",
            )

            try:
                model = pipeline_class(
                    quantize=self.quantize,
                    model_path=self.repo_id,
                    lora_paths=self.lora_paths or None,
                    lora_scales=self.lora_scales or None,
                    model_config=model_config,
                )
            except Exception as exc:  # noqa: BLE001 - surface a usable message
                raise ModelLoadError(self._explain_load_failure(exc, self.repo_id)) from exc

            if self.vae_tiling:
                self._enable_vae_tiling(model)

            self._model = model
            self._encoder_evicted = False
            self._load_seconds = time.perf_counter() - started
            self._model_root = self._resolve_root()

            bits = getattr(model, "bits", None)
            logger.info(
                "Model ready in %.1fs (quantisation: %s-bit)", self._load_seconds, bits
            )
            if bits is None:
                logger.warning(
                    "The weights report no quantisation level — this may be a "
                    "bf16 export, which will need far more memory than expected."
                )

            self._report(progress, "Ready", f"{self._load_seconds:.1f}s")
            return model

    def _resolve_root(self) -> Path | None:
        """Local directory the weights were resolved from (for partial reloads)."""
        try:
            from mflux.models.common.resolution.path_resolution import PathResolution

            return PathResolution.resolve(
                path=self.repo_id,
                patterns=self.family.load_weight_definition().get_download_patterns(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not resolve model root: %s", exc)
            return None

    def _apply_mlx_limits(self) -> None:
        if self.memory_mode == "off" or self.cache_limit_bytes is None:
            return
        from .device import apply_memory_settings

        apply_memory_settings(self.cache_limit_bytes)

    def _enable_vae_tiling(self, model: Any) -> None:
        """Tile VAE encode/decode so activations don't spike at high resolution."""
        try:
            from mflux.models.common.vae.tiling_config import TilingConfig

            if getattr(model, "tiling_config", None) is None:
                model.tiling_config = TilingConfig()
        except Exception as exc:  # noqa: BLE001 - tiling is an optimisation
            logger.debug("Could not enable VAE tiling: %s", exc)

    @staticmethod
    def _explain_load_failure(exc: Exception, repo_id: str = "") -> str:
        """Turn a deep MLX/HF traceback into something actionable."""
        text = str(exc)
        lowered = text.lower()

        # MLX reports a short shard as "invalid data offsets", which reads like
        # an internal bug. Name the actual bad file instead.
        if "data offsets" in lowered or "incomplete download" in lowered or "corrupt" in lowered:
            detail = ""
            try:
                from mflux.models.common.resolution.path_resolution import PathResolution

                root = PathResolution.resolve(
                    path=repo_id,
                    patterns=["*/*.safetensors", "*/*.json"],
                )
                if root is not None:
                    problems = verify_weights(Path(root))
                    if problems:
                        detail = "\n".join(f"  - {p}" for p in problems)
            except Exception:  # noqa: BLE001
                pass

            return (
                "The model weights are incomplete or corrupt.\n"
                + (f"{detail}\n" if detail else f"  {text}\n")
                + "Delete the cached snapshot in model.cache_dir and let it "
                "re-download. If it fails again the upstream repo itself is "
                "truncated — pick a different model.repo_id."
            )

        if "connection" in lowered or "resolve" in lowered or "timed out" in lowered:
            return (
                f"Could not download the model: {text}\n"
                "The first run needs ~32 GB from HuggingFace. Check your network "
                "and retry — partial downloads resume."
            )
        if "no space" in lowered or "disk" in lowered:
            return (
                f"Ran out of disk space downloading the model: {text}\n"
                "About 30 GB is needed in model.cache_dir."
            )
        if "not found" in lowered or "404" in lowered:
            return (
                f"Model not found: {text}\n"
                "Check model.repo_id in config.yaml. It must be a HuggingFace "
                "repo id ('org/name') or a local directory of mflux-format weights."
            )
        if "metal" in lowered or "gpu" in lowered:
            return (
                f"GPU initialisation failed: {text}\n"
                "See the Metal/MPS troubleshooting section in README.md."
            )
        return f"Could not load the model: {text}"

    # ----------------------------------------------------- memory management

    def evict_text_encoder(self) -> None:
        """Free the 15.5 GB bf16 text encoder after the prompt is encoded.

        Only meaningful in ``low`` mode. :meth:`_ensure_text_encoder` reloads it
        before the next edit needs it.
        """
        model = self._model
        if model is None or self._encoder_evicted:
            return

        # qwen_vl_encoder / the qwen_vl tokenizer only exist on Qwen-Image-Edit;
        # every family has `text_encoder`.
        for attr in ("text_encoder", "qwen_vl_encoder"):
            if getattr(model, attr, None) is not None:
                setattr(model, attr, None)
        tokenizers = getattr(model, "tokenizers", None)
        if isinstance(tokenizers, dict):
            tokenizers.pop("qwen_vl", None)

        self._encoder_evicted = True
        self._collect()
        logger.debug("Text encoder evicted (low memory mode).")

    def _ensure_text_encoder(self, progress: ProgressFn | None = None) -> None:
        """Rebuild the text encoder if a previous edit evicted it."""
        if not self._encoder_evicted or self._model is None:
            return

        self._report(progress, "Reloading text encoder", "freed after the last edit")
        started = time.perf_counter()

        try:
            self._reload_text_encoder()
        except Exception as exc:  # noqa: BLE001
            # A partial reload leaves the model unusable, so drop it entirely
            # and let the next call rebuild from scratch.
            logger.warning("Text-encoder reload failed (%s); reloading whole model.", exc)
            self._model = None
            self._encoder_evicted = False
            self._collect()
            self.ensure_loaded(progress)
            return

        self._encoder_evicted = False
        logger.info("Text encoder reloaded in %.1fs", time.perf_counter() - started)

    def _reload_text_encoder(self) -> None:
        """Load only the text_encoder component back onto the live model.

        Implemented for Qwen-Image-Edit, whose 15.5 GB bf16 encoder makes a
        partial reload worth the special-casing. Other families raise, and the
        caller falls back to reconstructing the whole pipeline — cheap for them,
        since their encoders are quantised and small.
        """
        if self.family.key != "qwen-image-edit":
            raise ModelLoadError(
                f"No partial text-encoder reload for {self.family.key}; "
                "reloading the full pipeline instead."
            )

        from mflux.models.common.weights.loading.weight_applier import WeightApplier
        from mflux.models.common.weights.loading.weight_loader import WeightLoader
        from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import (
            QwenTextEncoder,
        )
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import (
            QwenVisionLanguageEncoder,
        )
        from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import (
            VisionTransformer,
        )
        from mflux.models.qwen.tokenizer.qwen_vision_language_processor import (
            QwenVisionLanguageProcessor,
        )
        from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import (
            QwenVisionLanguageTokenizer,
        )
        from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

        root = self._model_root or self._resolve_root()
        if root is None:
            raise ModelLoadError("Could not locate the model directory for reload.")
        self._model_root = root

        component = next(
            c for c in QwenWeightDefinition.get_components() if c.name == "text_encoder"
        )
        # Load this one component only — a full WeightLoader.load() would also
        # pull the 16.6 GB transformer we already have resident.
        weights, q_level, version = WeightLoader._load_component(Path(root), component)

        from mflux.models.common.weights.loading.loaded_weights import (
            LoadedWeights,
            MetaData,
        )

        loaded = LoadedWeights(
            components={component.name: weights},
            meta_data=MetaData(quantization_level=q_level, mflux_version=version),
        )

        encoder = QwenTextEncoder()
        encoder.encoder.visual = VisionTransformer()
        WeightApplier.apply_and_quantize_single(
            weights=loaded, model=encoder, component=component, quantize_arg=self.quantize
        )

        model = self._model
        model.text_encoder = encoder
        model.qwen_vl_encoder = QwenVisionLanguageEncoder(encoder=encoder.encoder)
        model.tokenizers["qwen_vl"] = QwenVisionLanguageTokenizer(
            processor=QwenVisionLanguageProcessor(
                tokenizer=model.tokenizers["qwen"].tokenizer
            ),
            max_length=1024,
            use_picture_prefix=False,
        )

    def after_generation(self) -> None:
        """Housekeeping between edits: drop MLX's freed-buffer cache."""
        self._collect()

    def unload(self) -> None:
        """Release the whole model. Used on shutdown or a hard reset."""
        with self._lock:
            self._model = None
            self._encoder_evicted = False
            self._collect()
            logger.info("Model unloaded.")

    @staticmethod
    def _collect() -> None:
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _report(progress: ProgressFn | None, stage: str, detail: str = "") -> None:
        logger.info("%s%s", stage, f" — {detail}" if detail else "")
        if progress is not None:
            try:
                progress(LoadProgress(stage=stage, detail=detail))
            except Exception as exc:  # noqa: BLE001 - UI must never break loading
                logger.debug("Progress callback failed: %s", exc)


def fetch_remote_sizes(repo_id: str) -> dict[str, int] | None:
    """Published file sizes for a repo, or None if the Hub is unreachable."""
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{repo_id}?blobs=true", timeout=30
        ) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001 - offline is a normal state here
        logger.debug("Could not reach the Hub for size metadata: %s", exc)
        return None

    return {
        entry["rfilename"]: entry.get("size") or 0
        for entry in payload.get("siblings", [])
        if entry["rfilename"].endswith((".safetensors", ".json"))
    }


def download_status(
    repo_id: str, cache_dir: Path, remote: dict[str, int] | None = None
) -> dict[str, Any]:
    """Report how much of the model is on disk, per component.

    Compares the local snapshot against the repo's published file sizes, and
    counts bytes sitting in ``*.incomplete`` files as in-flight. Falls back to a
    local-only view when the Hub can't be reached, so this still works offline.

    Args:
        remote: Pre-fetched sizes from :func:`fetch_remote_sizes`. Pass this
            when polling in a loop so the Hub is queried once, not per tick.
    """
    configure_hf_cache(cache_dir)

    components: dict[str, dict[str, float]] = {}
    if remote is None:
        remote = fetch_remote_sizes(repo_id)
    online = remote is not None
    remote = remote or {}

    snapshot_root = Path(repo_id).expanduser()
    if not snapshot_root.is_dir():
        cache_name = f"models--{repo_id.replace('/', '--')}"
        snapshots = cache_dir / "hub" / cache_name / "snapshots"
        candidates = sorted(snapshots.iterdir()) if snapshots.is_dir() else []
        snapshot_root = candidates[0] if candidates else Path()

    for name, size in remote.items():
        component = name.split("/")[0] if "/" in name else "root"
        bucket = components.setdefault(component, {"have": 0.0, "want": 0.0})
        bucket["want"] += size
        candidate = snapshot_root / name
        if candidate.exists():
            try:
                bucket["have"] += candidate.stat().st_size
            except OSError:
                pass

    if not online and snapshot_root.exists():
        # Offline: report what is present without a completion percentage.
        for shard in snapshot_root.glob("*/*"):
            if shard.suffix not in {".safetensors", ".json"}:
                continue
            bucket = components.setdefault(shard.parent.name, {"have": 0.0, "want": 0.0})
            try:
                bucket["have"] += shard.stat().st_size
            except OSError:
                pass

    in_flight = 0.0
    hub_dir = cache_dir / "hub"
    if hub_dir.is_dir():
        for partial in hub_dir.rglob("*.incomplete"):
            try:
                in_flight += partial.stat().st_size
            except OSError:
                pass

    have = sum(c["have"] for c in components.values())
    want = sum(c["want"] for c in components.values())

    return {
        "repo_id": repo_id,
        "online": online,
        "components": components,
        "have_bytes": have,
        "want_bytes": want,
        "in_flight_bytes": in_flight,
        "complete": bool(want) and have >= want,
        "snapshot": snapshot_root if snapshot_root.exists() else None,
    }


def format_download_status(status: dict[str, Any], live: bool = False) -> str:
    """Render :func:`download_status` as a progress report.

    Args:
        live: True when redrawing in a watch loop, which drops the
            "re-run to refresh" hint that only applies to one-shot use.
    """
    lines: list[str] = []
    want = status["want_bytes"]

    for name in sorted(status["components"]):
        bucket = status["components"][name]
        if bucket["want"]:
            pct = 100 * bucket["have"] / bucket["want"]
            filled = int(pct / 5)
            bar = "█" * filled + "░" * (20 - filled)
            lines.append(
                f"  {name:13s} {bar} {pct:5.1f}%   "
                f"{bucket['have'] / 1e9:5.2f} / {bucket['want'] / 1e9:5.2f} GB"
            )
        else:
            lines.append(f"  {name:13s} {bucket['have'] / 1e9:5.2f} GB on disk")

    if want:
        pct = 100 * status["have_bytes"] / want
        lines.append("")
        lines.append(
            f"  Total         {status['have_bytes'] / 1e9:.2f} / {want / 1e9:.2f} GB  ({pct:.1f}%)"
        )
    if status["in_flight_bytes"]:
        lines.append(f"  In flight     {status['in_flight_bytes'] / 1e9:.2f} GB partially written")
    if not status["online"]:
        lines.append("  (offline — showing local files only)")
    if status["complete"]:
        lines.append("\n  ✅ Download complete.")
    elif want and not live:
        lines.append("\n  Still downloading. Re-run this command to refresh.")

    return "\n".join(lines) if lines else "  Nothing downloaded yet."


def download_model(
    repo_id: str,
    cache_dir: Path,
    *,
    family: str | ModelFamily = "flux2-klein-edit",
    max_attempts: int = 6,
    workers: int = 4,
    disable_xet: bool | None = None,
    endpoint: str | None = None,
) -> Path:
    """Fetch the weights, retrying through the flaky bits of a real network.

    Downloads resume from whatever is already on disk, so a retry after a
    dropped connection costs only the missing bytes. Between attempts the
    backoff grows, and Xet is toggled off if it looks like the transport is
    stalling — HuggingFace's Xet backend has been observed throttling to a few
    tens of KB/s where plain HTTP runs at full speed.

    Args:
        endpoint: Alternative Hub mirror (sets ``HF_ENDPOINT``), for networks
            where huggingface.co is slow or blocked.

    Raises:
        ModelLoadError: If every attempt fails.
    """
    import time

    configure_hf_cache(cache_dir)

    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelLoadError(
            "Dependencies are not installed. Run: pip install -r requirements.txt"
        ) from exc

    resolved = family if isinstance(family, ModelFamily) else get_family(family)
    patterns = [*resolved.load_weight_definition().get_download_patterns(), "tokenizer/*"]
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        before = download_status(repo_id, cache_dir)
        try:
            logger.info(
                "Downloading %s (attempt %d/%d)%s",
                repo_id,
                attempt,
                max_attempts,
                " without Xet" if os.environ.get("HF_HUB_DISABLE_XET") else "",
            )
            path = Path(
                snapshot_download(
                    repo_id=repo_id, allow_patterns=patterns, max_workers=workers
                )
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_error = exc
            after = download_status(repo_id, cache_dir)
            gained = after["have_bytes"] - before["have_bytes"]
            logger.warning(
                "Attempt %d failed after %.2f GB: %s", attempt, gained / 1e9, exc
            )

            if attempt >= max_attempts:
                break

            # If an attempt made no headway, the transport is the suspect —
            # flip Xet off (once) before simply waiting longer.
            if gained <= 0 and not os.environ.get("HF_HUB_DISABLE_XET"):
                logger.warning("No progress made; retrying with Xet disabled.")
                os.environ["HF_HUB_DISABLE_XET"] = "1"

            delay = min(60, 2**attempt)
            logger.info("Retrying in %ds…", delay)
            time.sleep(delay)
            continue

        problems = verify_weights(path)
        if problems:
            last_error = ModelLoadError("; ".join(problems))
            logger.warning(
                "Download completed but %d file(s) are incomplete; retrying.",
                len(problems),
            )
            # Remove the short blobs so the next pass re-fetches them rather
            # than trusting the cache and failing again at load time.
            _drop_truncated_blobs(path, problems)
            if attempt >= max_attempts:
                break
            continue

        logger.info("Model downloaded and verified: %s", path)
        return path

    raise ModelLoadError(
        f"Could not download {repo_id} after {max_attempts} attempts.\n"
        f"Last error: {last_error}\n"
        "Downloads resume, so re-running picks up where this stopped. On a "
        "restricted network try --mirror, or copy the model directory from "
        "another machine with --import-from."
    )


def _drop_truncated_blobs(snapshot: Path, problems: list[str]) -> None:
    """Delete short shards (and their blobs) so a retry re-fetches them."""
    for problem in problems:
        name = problem.split(":", 1)[0].strip()
        target = snapshot / name
        try:
            resolved = target.resolve() if target.is_symlink() else target
            for path in {target, resolved}:
                if path.exists():
                    path.unlink()
                    logger.info("Removed incomplete file: %s", path.name)
        except OSError as exc:
            logger.debug("Could not remove %s: %s", name, exc)


def import_model(source: Path, cache_dir: Path, repo_id: str) -> Path:
    """Adopt a weight directory copied from another machine.

    Sneakernet path for networks where a 32 GB download isn't practical: copy
    ``models/`` (or just the snapshot directory) onto a drive, then point this
    at it. The files are copied into the local cache layout and verified.
    """
    import shutil

    source = Path(source).expanduser().resolve()
    if not source.is_dir():
        raise ModelLoadError(f"Not a directory: {source}")

    # Accept either a snapshot directory or a whole HF cache tree.
    candidate = source
    if not any((source / part).is_dir() for part in ("transformer", "text_encoder")):
        matches = sorted(source.rglob("*/transformer"))
        if not matches:
            raise ModelLoadError(
                f"No model components found under {source}. Expected directories "
                f"named transformer/, text_encoder/, and vae/."
            )
        candidate = matches[0].parent

    problems = verify_weights(candidate)
    if problems:
        raise ModelLoadError(
            "The source copy is incomplete:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    configure_hf_cache(cache_dir)
    destination = (
        cache_dir / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "imported"
    )
    destination.mkdir(parents=True, exist_ok=True)

    for item in candidate.iterdir():
        if not item.is_dir():
            continue
        target = destination / item.name
        if target.exists():
            shutil.rmtree(target)
        # copy2 through the symlinks so the import is self-contained.
        shutil.copytree(item, target, symlinks=False)
        logger.info("Imported %s", item.name)

    remaining = verify_weights(destination)
    if remaining:
        raise ModelLoadError(
            "Import finished but the copy is not intact:\n"
            + "\n".join(f"  - {p}" for p in remaining)
        )

    logger.info("Model imported to %s", destination)
    return destination


def watch_download(
    repo_id: str,
    cache_dir: Path,
    interval: float = 5.0,
    stall_after: float = 120.0,
) -> int:
    """Live-render download progress until it completes or is interrupted.

    Returns 0 when the download finishes, 130 on Ctrl-C, 1 if it is not
    progressing. The Hub is queried once for file sizes; every tick after that
    is a local filesystem scan, so a short interval costs nothing.
    """
    import shutil
    import time

    remote = fetch_remote_sizes(repo_id)
    if remote is None:
        print("  Could not reach HuggingFace — showing local files only.\n")

    def measured(status: dict[str, Any]) -> float:
        # Count in-flight bytes: a 2 GB shard sits at 0% "complete" for minutes
        # while actively downloading, and a bar that never moves reads as a hang.
        return status["have_bytes"] + status["in_flight_bytes"]

    start = time.monotonic()
    status = download_status(repo_id, cache_dir, remote)
    baseline = measured(status)
    rate = 0.0                      # bytes/sec, exponentially smoothed
    last_bytes = baseline
    last_growth = time.monotonic()
    lines_drawn = 0
    columns = shutil.get_terminal_size((80, 24)).columns

    try:
        while True:
            status = download_status(repo_id, cache_dir, remote)
            now = time.monotonic()
            current = measured(status)

            delta = current - last_bytes
            if delta > 0:
                sample = delta / max(interval, 0.1)
                # Smooth hard: HuggingFace throughput is spiky and a raw rate
                # produces an ETA that swings by tens of minutes each tick.
                rate = sample if rate == 0 else (0.7 * rate + 0.3 * sample)
                last_growth = now
            last_bytes = current

            body = [format_download_status(status, live=True)]
            elapsed = now - start
            stalled = (now - last_growth) > stall_after

            if status["want_bytes"] and not status["complete"]:
                remaining = max(0.0, status["want_bytes"] - current)
                if stalled:
                    body.append(
                        f"  ⚠️  No progress for {(now - last_growth) / 60:.0f} min — "
                        f"the transfer may have stalled.\n"
                        f"     Interrupt the download and re-run it; it resumes. If it "
                        f"keeps stalling, retry with HF_HUB_DISABLE_XET=1."
                    )
                elif rate > 0:
                    eta = remaining / rate
                    body.append(
                        f"  {rate / 1e6:.1f} MB/s · ETA ~{eta / 60:.0f} min · "
                        f"elapsed {elapsed / 60:.0f} min"
                    )
                else:
                    body.append(f"  measuring… · elapsed {elapsed / 60:.0f} min")

            text = "\n".join(body)

            # Redraw in place rather than scrolling the terminal.
            if lines_drawn:
                print(f"\033[{lines_drawn}A\033[J", end="")
            print(text)
            lines_drawn = sum(
                max(1, -(-len(line) // max(columns, 1))) for line in text.split("\n")
            ) + 1

            if status["complete"]:
                print("\n  Weights are ready. Start the app with:  python app.py\n")
                return 0

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n  Stopped watching. The download itself is unaffected.\n")
        return 130


def verify_weights(root: Path) -> list[str]:
    """Check every safetensors shard is as long as its own header claims.

    A truncated shard is a real hazard: HuggingFace repos can publish short
    files, and the resulting failure surfaces from deep inside MLX as
    "invalid data offsets", which reads like a bug in this app rather than a
    bad download. Reading the 8-byte length prefix plus the JSON header is
    cheap — no tensor data is touched.

    Returns a list of human-readable problems; empty means the weights are
    structurally sound.
    """
    import json
    import struct

    problems: list[str] = []
    shards = sorted(Path(root).glob("*/*.safetensors"))

    if not shards:
        return [f"No .safetensors files found under {root}"]

    for shard in shards:
        try:
            actual = shard.stat().st_size
            with shard.open("rb") as handle:
                raw_len = handle.read(8)
                if len(raw_len) < 8:
                    problems.append(f"{shard.parent.name}/{shard.name}: file is truncated")
                    continue
                header_len = struct.unpack("<Q", raw_len)[0]
                header = json.loads(handle.read(header_len))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"{shard.parent.name}/{shard.name}: unreadable ({exc})")
            continue

        declared = max(
            (
                entry["data_offsets"][1]
                for key, entry in header.items()
                if key != "__metadata__" and isinstance(entry, dict) and "data_offsets" in entry
            ),
            default=0,
        )
        needed = 8 + header_len + declared
        if actual < needed:
            problems.append(
                f"{shard.parent.name}/{shard.name}: truncated — {actual:,} bytes on "
                f"disk but the header declares {needed:,}"
            )

    return problems


def is_model_cached(
    repo_id: str, cache_dir: Path, family: str | ModelFamily = "flux2-klein-edit"
) -> bool:
    """True if the weights are already on disk, so the UI can warn about a download."""
    configure_hf_cache(cache_dir)
    if Path(repo_id).expanduser().is_dir():
        return True
    try:
        from mflux.models.common.resolution.path_resolution import PathResolution

        resolved = family if isinstance(family, ModelFamily) else get_family(family)
        return (
            PathResolution._find_complete_cached_snapshot(
                repo_id, resolved.load_weight_definition().get_download_patterns()
            )
            is not None
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cache probe failed: %s", exc)
        return False
