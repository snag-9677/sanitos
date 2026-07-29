"""Device-detection tests, including the Linux/CUDA path.

The CUDA branch cannot be exercised on the development machine, so these tests
drive it through fakes. They verify the decisions that differ by platform:
which backend wins, which memory number bounds the model, and that the banner
renders without macOS-only assumptions.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.device import DeviceInfo, startup_banner  # noqa: E402


def make(**overrides) -> DeviceInfo:
    base = dict(
        chip="Apple M5",
        backend="mlx-metal",
        metal_available=True,
        total_memory_gb=24.0,
        cpu_cores=10,
        gpu_cores=10,
        macos_version="26.5.2",
        python_version="3.12.13",
        mlx_version="0.31.2",
        torch_mps_available=True,
        os_name="Darwin",
    )
    base.update(overrides)
    return DeviceInfo(**base)


# ------------------------------------------------------------ apple silicon


def test_apple_reports_unified_memory() -> None:
    info = make()
    assert info.is_unified_memory
    assert info.usable_memory_gb == 24.0
    assert info.backend_label == "MLX / Metal (Apple GPU)"


def test_apple_24gb_is_told_to_use_low_mode() -> None:
    advice = make(total_memory_gb=24.0).memory_advice()
    assert "low" in advice and "RAM" in advice


def test_large_machine_is_comfortable() -> None:
    assert "Comfortable" in make(total_memory_gb=64.0).memory_advice()


def test_small_machine_warns_about_swapping() -> None:
    advice = make(total_memory_gb=8.0).memory_advice()
    assert "Constrained" in advice and "swapping" in advice


# --------------------------------------------------------------- linux/cuda


def test_cuda_is_bounded_by_vram_not_system_ram() -> None:
    """A 256 GB server with a 24 GB card is still a 24 GB budget."""
    info = make(
        chip="AMD EPYC 7763",
        backend="mlx-cuda",
        metal_available=False,
        os_name="Linux",
        total_memory_gb=256.0,
        gpu_name="NVIDIA RTX 4090",
        gpu_memory_gb=24.0,
        gpu_cores=None,
        torch_mps_available=False,
    )

    assert not info.is_unified_memory
    assert info.usable_memory_gb == 24.0
    advice = info.memory_advice()
    assert "VRAM" in advice and "low" in advice


def test_cuda_out_of_memory_wording_not_swapping() -> None:
    """Discrete GPUs OOM rather than swap; the advice must say so."""
    info = make(
        backend="mlx-cuda", os_name="Linux", gpu_memory_gb=12.0, total_memory_gb=128.0
    )
    advice = info.memory_advice()
    assert "OOM" in advice and "swapping" not in advice


def test_big_card_is_comfortable() -> None:
    info = make(backend="mlx-cuda", os_name="Linux", gpu_memory_gb=80.0)
    assert "Comfortable" in info.memory_advice()
    assert "VRAM" in info.memory_advice()


def test_a_model_that_only_just_fits_a_card_is_not_called_comfortable() -> None:
    """FLUX.2 Klein 9B on a 24 GB card: 17.9 GB of weights, no room to denoise.

    Unified memory can call this comfortable — it pages if wrong. VRAM cannot.
    """
    card = make(backend="mlx-cuda", os_name="Linux", gpu_memory_gb=24.0)
    mac = make(total_memory_gb=24.0)

    advice = card.memory_advice(working_set_gb=17.9, denoise_peak_gb=9.8)
    assert "Comfortable" not in advice
    assert "low" in advice
    assert "Comfortable" in mac.memory_advice(working_set_gb=17.9, denoise_peak_gb=9.8)


def test_the_advised_denoise_peak_comes_from_the_model() -> None:
    """A stale hardcoded peak told 24 GB users the wrong number to expect."""
    card = make(backend="mlx-cuda", os_name="Linux", gpu_memory_gb=24.0)
    assert "~10 GB peak" in card.memory_advice(working_set_gb=17.9, denoise_peak_gb=9.8)


def test_only_a_discrete_gpu_has_a_hard_limit() -> None:
    assert make(backend="mlx-cuda", os_name="Linux", gpu_memory_gb=24.0).has_hard_memory_limit
    assert not make().has_hard_memory_limit
    assert not make(backend="cpu", metal_available=False).has_hard_memory_limit
    # A CUDA backend whose VRAM could not be read has no number to bound with.
    assert not make(backend="mlx-cuda", os_name="Linux", gpu_memory_gb=None).has_hard_memory_limit


def test_cuda_banner_has_no_macos_leakage() -> None:
    info = make(
        chip="AMD EPYC 7763",
        backend="mlx-cuda",
        metal_available=False,
        os_name="Linux",
        macos_version="6.8.0-generic",
        total_memory_gb=128.0,
        gpu_name="NVIDIA A100",
        gpu_memory_gb=40.0,
        gpu_cores=None,
        torch_mps_available=False,
    )

    banner = startup_banner(info, "Qwen-Image-Edit 6-bit", "low")

    assert "macOS" not in banner
    assert "torch MPS" not in banner
    assert "NVIDIA A100" in banner
    assert "VRAM" in banner
    assert "Linux 6.8.0-generic" in banner
    assert "MLX / CUDA (NVIDIA GPU)" in banner


def test_apple_banner_keeps_mps_line() -> None:
    banner = startup_banner(make(), "Qwen-Image-Edit 6-bit", "low")
    assert "macOS 26.5.2" in banner
    assert "torch MPS: available" in banner
    assert "unified" in banner


def test_banner_box_borders_line_up() -> None:
    """A ragged box is the first thing anyone notices at startup."""
    for info in (make(), make(backend="mlx-cuda", os_name="Linux", gpu_memory_gb=40.0)):
        lines = startup_banner(info, "model", "low").split("\n")
        box = [ln for ln in lines if any(ch in ln for ch in "┌│└")]
        assert len({len(ln) for ln in box}) == 1, f"ragged box: {[len(l) for l in box]}"


# ---------------------------------------------------------------------- cpu


def test_cpu_fallback_labels_itself_plainly() -> None:
    info = make(backend="cpu", metal_available=False, gpu_cores=None)
    assert info.backend_label == "CPU"


def test_unknown_memory_does_not_crash_the_advice() -> None:
    assert "Could not determine" in make(total_memory_gb=0.0, backend="cpu").memory_advice()


# ------------------------------------------------------------ live machine


def test_detect_device_matches_this_machine() -> None:
    """Sanity check against whatever we are actually running on."""
    from src.device import detect_device

    info = detect_device()
    assert info.backend in {"mlx-metal", "mlx-cuda", "cpu"}
    assert info.cpu_cores > 0
    assert isinstance(startup_banner(info, "model", "low"), str)
