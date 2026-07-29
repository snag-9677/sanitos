"""Configuration loading for the Qwen Image Edit M5 app.

Everything the app needs is declared in ``config.yaml``. This module turns that
YAML into typed dataclasses, resolves paths (``~``, ``${ENV}``, relative-to-
project-root), and validates the values that would otherwise fail deep inside
MLX with an unhelpful message.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_ENV_PATTERN = re.compile(r"\$\{([^}^{]+)\}")


class ConfigError(RuntimeError):
    """Raised when config.yaml is missing, malformed, or internally invalid."""


def _expand(value: str) -> str:
    """Expand ``${VAR}`` and ``~`` inside a config string."""

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = ""
        if ":-" in name:
            name, default = name.split(":-", 1)
        return os.environ.get(name, default)

    return os.path.expanduser(_ENV_PATTERN.sub(replace, value))


def resolve_path(value: str | os.PathLike[str], *, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path against the project root.

    Absolute paths are respected as-is; relative ones are anchored to the
    project root rather than the process CWD, so ``python app.py`` behaves the
    same from any directory.
    """
    path = Path(_expand(str(value)))
    return path if path.is_absolute() else (root / path).resolve()


@dataclass(slots=True)
class ResolutionPreset:
    """A resolution choice.

    Either ``megapixels`` (preserve the source aspect ratio at this pixel
    budget) or an explicit ``width``/``height`` pair (reframe to a fixed shape).
    """

    label: str
    width: int | None = None
    height: int | None = None
    megapixels: float | None = None

    @property
    def is_auto(self) -> bool:
        return self.megapixels is not None and not (self.width and self.height)


@dataclass(slots=True)
class StylePreset:
    label: str
    prompt: str


@dataclass(slots=True)
class ModelEntry:
    """One selectable model: a weights repo paired with the pipeline for it."""

    id: str
    label: str
    repo_id: str
    family: str
    notes: str = ""

    def __post_init__(self) -> None:
        from .families import FAMILIES

        if not self.repo_id:
            raise ConfigError(f"Model {self.id!r} has an empty repo_id.")
        if self.family not in FAMILIES:
            raise ConfigError(
                f"Model {self.id!r} has family {self.family!r}; "
                f"expected one of {sorted(FAMILIES)}."
            )

    @property
    def size_gb(self) -> float:
        from .families import get_family

        return get_family(self.family).working_set_gb

    def describe(self, memory_gb: float = 0.0) -> str:
        """Dropdown label: name, size, and whether it fits this machine."""
        text = f"{self.label} · {self.size_gb:.0f} GB"
        if memory_gb and self.size_gb > memory_gb - 2:
            text += "  \u26a0 larger than this machine"
        return text


@dataclass(slots=True)
class ModelConfig:
    repo_id: str
    cache_dir: Path
    # Which mflux pipeline drives this repo; see src/families.py.
    family: str = "flux2-klein-edit"
    # Selectable models, offered in the UI and via --model.
    catalog: list[ModelEntry] = field(default_factory=list)
    active: str = ""
    quantize: int | None = None
    preload: bool = False
    lora_paths: list[str] = field(default_factory=list)
    lora_scales: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        from .families import FAMILIES

        if not self.repo_id:
            raise ConfigError("model.repo_id must not be empty.")
        if self.family not in FAMILIES:
            raise ConfigError(
                f"model.family must be one of {sorted(FAMILIES)}, got {self.family!r}."
            )
        if len(self.lora_paths) != len(self.lora_scales):
            raise ConfigError(
                f"model.lora_paths has {len(self.lora_paths)} entries but "
                f"model.lora_scales has {len(self.lora_scales)}; they must match."
            )

        ids = [entry.id for entry in self.catalog]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ConfigError(f"Duplicate model ids in model.catalog: {sorted(duplicates)}")

        # Whatever repo_id/family point at stays selectable, so a config with no
        # catalog still works and a hand-edited repo_id still appears in the UI.
        if not any(e.repo_id == self.repo_id and e.family == self.family for e in self.catalog):
            self.catalog.insert(
                0,
                ModelEntry(
                    id="configured",
                    label=self.repo_id.split("/")[-1],
                    repo_id=self.repo_id,
                    family=self.family,
                    notes="From model.repo_id in config.yaml.",
                ),
            )
            ids.insert(0, "configured")

        if self.active and self.active not in ids:
            raise ConfigError(
                f"model.active is {self.active!r}, which is not in model.catalog "
                f"({sorted(ids)})."
            )
        if not self.active:
            self.active = self.entry_for_repo(self.repo_id).id

    def entry(self, model_id: str) -> ModelEntry:
        for candidate in self.catalog:
            if candidate.id == model_id:
                return candidate
        raise ConfigError(
            f"No model with id {model_id!r}. "
            f"Available: {sorted(e.id for e in self.catalog)}"
        )

    def entry_for_repo(self, repo_id: str) -> ModelEntry:
        for candidate in self.catalog:
            if candidate.repo_id == repo_id:
                return candidate
        return self.catalog[0]

    @property
    def active_entry(self) -> ModelEntry:
        return self.entry(self.active)


@dataclass(slots=True)
class MemoryConfig:
    mode: str = "auto"
    cache_limit_bytes: int | None = 1024**3
    vae_tiling: bool = True

    VALID_MODES = ("auto", "balanced", "low", "off")

    def resolve(self, working_set_gb: float, available_gb: float) -> str:
        """Turn ``auto`` into a concrete mode for the selected model.

        Eviction costs a text-encoder reload every turn, so it is only worth
        paying when the model genuinely does not fit. Models are switchable at
        runtime now, so this is decided per model rather than once in the file.
        """
        if self.mode != "auto":
            return self.mode
        if available_gb <= 0:
            return "balanced"
        return "balanced" if working_set_gb <= available_gb - 2 else "low"

    def __post_init__(self) -> None:
        if self.mode not in self.VALID_MODES:
            raise ConfigError(
                f"memory.mode must be one of {self.VALID_MODES}, got {self.mode!r}."
            )


@dataclass(slots=True)
class GenerationConfig:
    steps: int = 25
    guidance: float = 4.0
    width: int | None = None
    height: int | None = None
    negative_prompt: str = ""
    scheduler: str = "linear"
    resolution_presets: list[ResolutionPreset] = field(default_factory=list)
    m5_recommended: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ConfigError("generation.steps must be >= 1.")
        if self.guidance < 0:
            raise ConfigError("generation.guidance must be >= 0.")


@dataclass(slots=True)
class OutputConfig:
    dir: Path
    save_metadata: bool = True
    save_sessions: bool = True
    format: str = "png"

    @property
    def sessions_dir(self) -> Path:
        return self.dir / "sessions"


@dataclass(slots=True)
class UIConfig:
    title: str = "Qwen Image Edit — M5"
    server_name: str = "127.0.0.1"
    server_port: int = 7860
    share: bool = False
    inbrowser: bool = True
    style_presets: list[StylePreset] = field(default_factory=list)


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    file: Path | None = None


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig
    memory: MemoryConfig
    generation: GenerationConfig
    output: OutputConfig
    ui: UIConfig
    logging: LoggingConfig
    api_enabled: bool = True
    source_path: Path | None = None

    def ensure_directories(self) -> None:
        """Create the writable directories the app depends on."""
        self.model.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output.dir.mkdir(parents=True, exist_ok=True)
        if self.output.save_sessions:
            self.output.sessions_dir.mkdir(parents=True, exist_ok=True)
        if self.logging.file is not None:
            self.logging.file.parent.mkdir(parents=True, exist_ok=True)


def _preset_list(raw: Any) -> list[ResolutionPreset]:
    presets: list[ResolutionPreset] = []
    for item in raw or []:
        megapixels = item.get("megapixels")
        presets.append(
            ResolutionPreset(
                label=str(item["label"]),
                width=item.get("width"),
                height=item.get("height"),
                megapixels=float(megapixels) if megapixels is not None else None,
            )
        )
    if not presets:
        presets.append(ResolutionPreset("Match input, 1.0 MP", megapixels=1.0))
    return presets


def _catalog(raw: Any) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    for item in raw or []:
        entries.append(
            ModelEntry(
                id=str(item["id"]),
                label=str(item.get("label") or item["id"]),
                repo_id=str(item["repo_id"]),
                family=str(item["family"]),
                notes=str(item.get("notes") or ""),
            )
        )
    return entries


def _style_list(raw: Any) -> list[StylePreset]:
    return [
        StylePreset(label=str(item["label"]), prompt=str(item["prompt"]))
        for item in raw or []
    ]


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    """Load and validate ``config.yaml``.

    Args:
        path: Optional explicit config path. Defaults to ``config.yaml`` beside
            ``app.py``. May also be set via ``QWEN_EDIT_CONFIG``.

    Raises:
        ConfigError: If the file is missing or any value fails validation.
    """
    config_path = Path(
        path or os.environ.get("QWEN_EDIT_CONFIG") or DEFAULT_CONFIG_PATH
    ).expanduser()

    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}\n"
            f"Expected config.yaml beside app.py, or set QWEN_EDIT_CONFIG."
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level.")

    root = config_path.parent
    model_raw = raw.get("model") or {}
    memory_raw = raw.get("memory") or {}
    gen_raw = raw.get("generation") or {}
    out_raw = raw.get("output") or {}
    ui_raw = raw.get("ui") or {}
    log_raw = raw.get("logging") or {}

    log_file = log_raw.get("file")

    config = AppConfig(
        model=ModelConfig(
            repo_id=str(model_raw.get("repo_id", "")).strip(),
            cache_dir=resolve_path(model_raw.get("cache_dir", "./models/hf"), root=root),
            family=str(model_raw.get("family", "flux2-klein-edit")).strip(),
            catalog=_catalog(model_raw.get("catalog")),
            active=str(model_raw.get("active", "")).strip(),
            quantize=model_raw.get("quantize"),
            preload=bool(model_raw.get("preload", False)),
            lora_paths=list(model_raw.get("lora_paths") or []),
            lora_scales=[float(s) for s in (model_raw.get("lora_scales") or [])],
        ),
        memory=MemoryConfig(
            mode=str(memory_raw.get("mode", "low")),
            cache_limit_bytes=memory_raw.get("cache_limit_bytes", 1024**3),
            vae_tiling=bool(memory_raw.get("vae_tiling", True)),
        ),
        generation=GenerationConfig(
            steps=int(gen_raw.get("steps", 25)),
            guidance=float(gen_raw.get("guidance", 4.0)),
            width=gen_raw.get("width"),
            height=gen_raw.get("height"),
            negative_prompt=str(gen_raw.get("negative_prompt") or ""),
            scheduler=str(gen_raw.get("scheduler", "linear")),
            resolution_presets=_preset_list(gen_raw.get("resolution_presets")),
            m5_recommended=dict(gen_raw.get("m5_recommended") or {}),
        ),
        output=OutputConfig(
            dir=resolve_path(out_raw.get("dir", "./outputs"), root=root),
            save_metadata=bool(out_raw.get("save_metadata", True)),
            save_sessions=bool(out_raw.get("save_sessions", True)),
            format=str(out_raw.get("format", "png")).lower(),
        ),
        ui=UIConfig(
            title=str(ui_raw.get("title", "Qwen Image Edit — M5")),
            server_name=str(ui_raw.get("server_name", "127.0.0.1")),
            server_port=int(ui_raw.get("server_port", 7860)),
            share=bool(ui_raw.get("share", False)),
            inbrowser=bool(ui_raw.get("inbrowser", True)),
            style_presets=_style_list(ui_raw.get("style_presets")),
        ),
        logging=LoggingConfig(
            level=str(log_raw.get("level", "INFO")).upper(),
            file=resolve_path(log_file, root=root) if log_file else None,
        ),
        api_enabled=bool((raw.get("api") or {}).get("enabled", True)),
        source_path=config_path,
    )

    config.ensure_directories()
    return config


def configure_logging(cfg: LoggingConfig) -> None:
    """Install root logging handlers (console + optional rotating file)."""
    level = getattr(logging, cfg.level, logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if cfg.file is not None:
        try:
            handlers.append(logging.FileHandler(cfg.file, encoding="utf-8"))
        except OSError as exc:  # non-fatal: keep console logging
            logging.getLogger(__name__).warning(
                "Could not open log file %s: %s", cfg.file, exc
            )

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # These are chatty at INFO and drown out our own progress output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
