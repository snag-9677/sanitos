"""Model selection: the catalog, runtime switching, and memory guards.

The guards here exist because getting this wrong is not a graceful failure —
exceeding GPU memory aborts the process from inside the CUDA allocator
("cudaMallocAsync ... out of memory"), which no Python handler can catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigError, MemoryConfig, ModelEntry, load_config  # noqa: E402
from src.families import get_family  # noqa: E402
from src.model_loader import EditModel, ModelLoadError  # noqa: E402


# ------------------------------------------------------------------ catalog


def test_catalog_loads_and_has_an_active_entry() -> None:
    cfg = load_config()
    assert cfg.model.catalog, "config.yaml should offer selectable models"
    assert cfg.model.active_entry.repo_id
    assert cfg.model.active_entry in cfg.model.catalog


def test_configured_repo_is_always_selectable() -> None:
    """A hand-edited repo_id must still appear in the picker."""
    from src.config import ModelConfig

    cfg = ModelConfig(
        repo_id="someone/custom-export",
        cache_dir=Path("/tmp"),
        family="flux2-klein-edit",
        catalog=[
            ModelEntry("a", "A", "mlx-community/flux2-klein-9b-8bit", "flux2-klein-edit")
        ],
    )
    assert any(e.repo_id == "someone/custom-export" for e in cfg.catalog)
    assert cfg.active_entry.repo_id == "someone/custom-export"


def test_mismatched_family_is_rejected() -> None:
    with pytest.raises(ConfigError):
        ModelEntry("x", "X", "some/repo", "not-a-real-family")


def test_duplicate_ids_are_rejected() -> None:
    from src.config import ModelConfig

    with pytest.raises(ConfigError, match="Duplicate"):
        ModelConfig(
            repo_id="mlx-community/flux2-klein-9b-8bit",
            cache_dir=Path("/tmp"),
            family="flux2-klein-edit",
            catalog=[
                ModelEntry("dup", "A", "mlx-community/flux2-klein-9b-8bit", "flux2-klein-edit"),
                ModelEntry("dup", "B", "mlx-community/FLUX.2-Klein-4B-6bit", "flux2-klein-edit-4b"),
            ],
        )


def test_unknown_active_id_is_rejected() -> None:
    from src.config import ModelConfig

    with pytest.raises(ConfigError, match="model.active"):
        ModelConfig(
            repo_id="mlx-community/flux2-klein-9b-8bit",
            cache_dir=Path("/tmp"),
            family="flux2-klein-edit",
            catalog=[
                ModelEntry("real", "A", "mlx-community/flux2-klein-9b-8bit", "flux2-klein-edit")
            ],
            active="imaginary",
        )


def test_entry_description_flags_models_that_do_not_fit() -> None:
    big = ModelEntry("q", "Qwen", "OsaurusAI/Qwen-Image-Edit-mflux-q6", "qwen-image-edit")
    small = ModelEntry("k", "Klein", "mlx-community/FLUX.2-Klein-4B-6bit", "flux2-klein-edit-4b")

    assert "larger than this machine" in big.describe(24.0)
    assert "larger than this machine" not in small.describe(24.0)
    # A machine big enough should carry no warning.
    assert "larger than this machine" not in big.describe(64.0)


# -------------------------------------------------------------- memory mode


def test_auto_mode_keeps_a_fitting_model_resident() -> None:
    memory = MemoryConfig(mode="auto")
    assert memory.resolve(working_set_gb=17.9, available_gb=24.0) == "balanced"


def test_auto_mode_evicts_when_the_model_does_not_fit() -> None:
    memory = MemoryConfig(mode="auto")
    assert memory.resolve(working_set_gb=32.4, available_gb=24.0) == "low"


def test_explicit_mode_is_never_overridden() -> None:
    assert MemoryConfig(mode="balanced").resolve(32.4, 24.0) == "balanced"
    assert MemoryConfig(mode="low").resolve(6.6, 64.0) == "low"


def test_unknown_memory_falls_back_to_balanced() -> None:
    assert MemoryConfig(mode="auto").resolve(17.9, 0.0) == "balanced"


# ----------------------------------------------------------------- switching


@pytest.fixture()
def handle(tmp_path: Path) -> EditModel:
    return EditModel(
        repo_id="mlx-community/flux2-klein-9b-8bit",
        cache_dir=tmp_path,
        family="flux2-klein-edit",
        label="Klein 9B",
    )


def test_switch_changes_repo_family_and_label(handle: EditModel) -> None:
    handle.switch_to(
        "mlx-community/FLUX.2-Klein-4B-6bit", "flux2-klein-edit-4b", label="Klein 4B"
    )

    assert handle.repo_id == "mlx-community/FLUX.2-Klein-4B-6bit"
    assert handle.family.key == "flux2-klein-edit-4b"
    assert handle.label == "Klein 4B"
    assert not handle.is_loaded, "the new model must load lazily, not eagerly"


def test_switch_unloads_the_previous_model(handle: EditModel) -> None:
    """Holding two models at once would need the sum of both working sets."""
    handle._model = object()
    assert handle.is_loaded

    handle.switch_to("mlx-community/FLUX.2-Klein-4B-6bit", "flux2-klein-edit-4b")

    assert not handle.is_loaded
    assert handle.load_seconds is None


def test_reselecting_the_same_model_is_a_no_op(handle: EditModel) -> None:
    sentinel = object()
    handle._model = sentinel

    handle.switch_to("mlx-community/flux2-klein-9b-8bit", "flux2-klein-edit")

    assert handle._model is sentinel, "re-selecting must not pay a reload"


def test_switch_can_change_the_memory_mode(handle: EditModel) -> None:
    handle.switch_to(
        "OsaurusAI/Qwen-Image-Edit-mflux-q6", "qwen-image-edit", memory_mode="low"
    )
    assert handle.memory_mode == "low"


# -------------------------------------------------------- gpu memory guard


def test_preflight_blocks_a_model_larger_than_vram(tmp_path: Path) -> None:
    """Better a clear error than an uncatchable CUDA abort mid-generation."""
    model = EditModel(
        repo_id="OsaurusAI/Qwen-Image-Edit-mflux-q6",
        cache_dir=tmp_path,
        family="qwen-image-edit",
        memory_mode="balanced",
        memory_budget_gb=24.0,
    )
    with pytest.raises(ModelLoadError) as excinfo:
        model._preflight()

    message = str(excinfo.value)
    assert "32" in message and "24" in message
    assert "memory.mode: low" in message
    assert "smaller model" in message


def test_preflight_accounts_for_low_mode_eviction(tmp_path: Path) -> None:
    """In low mode only the transformer stays resident, so the bar is lower."""
    model = EditModel(
        repo_id="OsaurusAI/Qwen-Image-Edit-mflux-q6",
        cache_dir=tmp_path,
        family="qwen-image-edit",
        memory_mode="low",
        memory_budget_gb=24.0,
    )
    model._preflight()  # 16.85 GB denoise peak + 2 GB headroom fits in 24 GB

    tight = EditModel(
        repo_id="OsaurusAI/Qwen-Image-Edit-mflux-q6",
        cache_dir=tmp_path,
        family="qwen-image-edit",
        memory_mode="low",
        memory_budget_gb=16.0,
    )
    with pytest.raises(ModelLoadError):
        tight._preflight()


def test_preflight_passes_for_a_model_that_fits(tmp_path: Path) -> None:
    model = EditModel(
        repo_id="mlx-community/flux2-klein-9b-8bit",
        cache_dir=tmp_path,
        family="flux2-klein-edit",
        memory_budget_gb=24.0,
    )
    model._preflight()


def test_no_budget_means_no_preflight_block(tmp_path: Path) -> None:
    """Unified memory swaps rather than dying, so oversize is the user's call."""
    model = EditModel(
        repo_id="OsaurusAI/Qwen-Image-Edit-mflux-q6",
        cache_dir=tmp_path,
        family="qwen-image-edit",
        memory_mode="balanced",
        memory_budget_gb=None,
    )
    model._preflight()


def test_every_catalog_entry_resolves(tmp_path: Path) -> None:
    """A typo in config.yaml should fail here, not on first edit."""
    cfg = load_config()
    for entry in cfg.model.catalog:
        family = get_family(entry.family)
        assert family.load_pipeline_class()
        assert entry.size_gb > 0
