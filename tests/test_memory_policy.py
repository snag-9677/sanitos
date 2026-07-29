"""Memory-policy tests: eviction and reload behaviour, without loading weights.

The failure these guard against is invisible in output — the images come out
identical either way — but it silently costs 15.5 GB and tens of seconds per
preview, which is the whole point of low mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_loader import ModelLoadError, QwenEditModel, verify_weights  # noqa: E402


class FakeModel:
    """Minimal stand-in with the attributes the eviction path touches."""

    def __init__(self) -> None:
        self.text_encoder = object()
        self.qwen_vl_encoder = object()
        self.vae = object()
        self.transformer = object()
        self.tokenizers = {"qwen": object(), "qwen_vl": object()}
        self.tiling_config = None


@pytest.fixture()
def handle(tmp_path: Path) -> QwenEditModel:
    model = QwenEditModel(repo_id="fake/repo", cache_dir=tmp_path, memory_mode="low")
    model._model = FakeModel()
    return model


def test_evict_frees_only_the_text_encoder(handle: QwenEditModel) -> None:
    handle.evict_text_encoder()

    live = handle._model
    assert live.text_encoder is None
    assert live.qwen_vl_encoder is None
    assert "qwen_vl" not in live.tokenizers
    # The parts the denoise loop and VAE decode still need must survive.
    assert live.transformer is not None
    assert live.vae is not None
    assert live.tokenizers["qwen"] is not None


def test_evict_is_idempotent(handle: QwenEditModel) -> None:
    handle.evict_text_encoder()
    handle.evict_text_encoder()
    assert handle._encoder_evicted is True


def test_live_model_does_not_trigger_a_reload(handle: QwenEditModel) -> None:
    """Preview decoding must not resurrect the evicted text encoder.

    ``ensure_loaded()`` deliberately reloads the encoder; ``live_model`` must
    not, or every preview decode would pull 15.5 GB back into memory in the
    middle of a generation.
    """
    handle.evict_text_encoder()
    reloads: list[int] = []
    handle._reload_text_encoder = lambda: reloads.append(1)  # type: ignore[method-assign]

    model = handle.live_model

    assert model is handle._model
    assert reloads == [], "live_model must not reload the text encoder"
    assert handle._encoder_evicted is True

    # ensure_loaded, by contrast, is expected to restore it.
    handle.ensure_loaded()
    assert reloads == [1]
    assert handle._encoder_evicted is False


def test_live_model_raises_when_nothing_is_loaded(tmp_path: Path) -> None:
    model = QwenEditModel(repo_id="fake/repo", cache_dir=tmp_path)
    with pytest.raises(ModelLoadError):
        _ = model.live_model


def test_balanced_mode_is_not_evicted_by_the_callback(handle: QwenEditModel) -> None:
    """Only low mode should release the encoder."""
    from src.inference import _GenerationCallback

    handle.memory_mode = "balanced"
    callback = _GenerationCallback(handle, 25, None, None, enable_previews=False)
    callback.call_before_loop(seed=1, prompt="x", latents=None, config=None)

    assert handle._model.text_encoder is not None


def test_low_mode_evicts_via_the_callback(handle: QwenEditModel) -> None:
    from src.inference import _GenerationCallback

    callback = _GenerationCallback(handle, 25, None, None, enable_previews=False)
    callback.call_before_loop(seed=1, prompt="x", latents=None, config=None)

    assert handle._model.text_encoder is None


def test_cancel_flag_raises_keyboard_interrupt(handle: QwenEditModel) -> None:
    """mflux turns KeyboardInterrupt into a stop that preserves the latents."""
    import threading

    from src.inference import _GenerationCallback

    cancel = threading.Event()
    callback = _GenerationCallback(handle, 25, None, cancel, enable_previews=False)

    callback.call_in_loop(0, 1, "x", None, None, None)  # not cancelled: fine
    cancel.set()
    with pytest.raises(KeyboardInterrupt):
        callback.call_in_loop(1, 1, "x", None, None, None)


def test_verify_weights_flags_a_truncated_shard(tmp_path: Path) -> None:
    """Build a shard whose header claims more data than the file holds."""
    import json
    import struct

    component = tmp_path / "text_encoder"
    component.mkdir()

    header = {
        "__metadata__": {"quantization_level": "6"},
        "a.weight": {"dtype": "BF16", "shape": [1024, 1024], "data_offsets": [0, 2_097_152]},
    }
    blob = json.dumps(header).encode()

    good = component / "0.safetensors"
    good.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 2_097_152)
    bad = component / "1.safetensors"
    bad.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * 64)  # short

    problems = verify_weights(tmp_path)
    assert len(problems) == 1
    assert "1.safetensors" in problems[0] and "truncated" in problems[0]


def test_verify_weights_accepts_complete_shards(tmp_path: Path) -> None:
    import json
    import struct

    component = tmp_path / "vae"
    component.mkdir()
    header = {"a.weight": {"dtype": "BF16", "shape": [8], "data_offsets": [0, 16]}}
    blob = json.dumps(header).encode()
    (component / "0.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\0" * 16
    )

    assert verify_weights(tmp_path) == []


def test_verify_weights_reports_an_empty_directory(tmp_path: Path) -> None:
    assert verify_weights(tmp_path) != []


# -------------------------------------------------------- download status


def test_download_status_is_offline_safe(tmp_path: Path, monkeypatch) -> None:
    """No network: report local files without pretending to know the total."""
    from src import model_loader

    def boom(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr("urllib.request.urlopen", boom)

    snapshot = tmp_path / "hub" / "models--org--model" / "snapshots" / "abc"
    (snapshot / "vae").mkdir(parents=True)
    (snapshot / "vae" / "0.safetensors").write_bytes(b"\0" * 2048)

    status = model_loader.download_status("org/model", tmp_path)

    assert status["online"] is False
    assert status["components"]["vae"]["have"] == 2048
    assert "offline" in model_loader.format_download_status(status)


def test_download_status_counts_in_flight_bytes(tmp_path: Path, monkeypatch) -> None:
    from src import model_loader

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    blobs = tmp_path / "hub" / "models--org--model" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "deadbeef.incomplete").write_bytes(b"\0" * 4096)

    status = model_loader.download_status("org/model", tmp_path)
    assert status["in_flight_bytes"] == 4096
    assert "In flight" in model_loader.format_download_status(status)


def test_download_status_reports_completion(tmp_path: Path) -> None:
    """A local directory of weights counts as fully present."""
    from src import model_loader

    local = tmp_path / "weights"
    (local / "vae").mkdir(parents=True)
    (local / "vae" / "0.safetensors").write_bytes(b"\0" * 512)

    status = model_loader.download_status(str(local), tmp_path)
    rendered = model_loader.format_download_status(status)
    assert isinstance(rendered, str) and rendered.strip()


def test_format_suppresses_the_refresh_hint_in_live_mode(tmp_path: Path, monkeypatch) -> None:
    """The watch loop redraws itself; telling the user to re-run is wrong there."""
    from src import model_loader

    monkeypatch.setattr(
        model_loader, "fetch_remote_sizes", lambda repo: {"vae/0.safetensors": 4096}
    )
    status = model_loader.download_status("org/model", tmp_path)

    assert "Re-run this command" in model_loader.format_download_status(status)
    assert "Re-run this command" not in model_loader.format_download_status(status, live=True)


def test_fetch_remote_sizes_returns_none_when_offline(monkeypatch) -> None:
    from src import model_loader

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    assert model_loader.fetch_remote_sizes("org/model") is None


def test_download_status_accepts_prefetched_sizes(tmp_path: Path, monkeypatch) -> None:
    """Polling must not re-query the Hub on every tick."""
    from src import model_loader

    calls: list[str] = []
    monkeypatch.setattr(
        model_loader, "fetch_remote_sizes", lambda repo: calls.append(repo) or {}
    )
    model_loader.download_status("org/model", tmp_path, remote={"vae/0.safetensors": 10})
    assert calls == [], "prefetched sizes should short-circuit the network call"
