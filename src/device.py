"""Device detection and the startup banner.

The compute path is MLX rather than PyTorch MPS: the only 6-bit
Qwen-Image-Edit weights that exist are in mflux/MLX format, and PyTorch has no
6-bit path on Apple GPUs (bitsandbytes is CUDA-only, and diffusers' GGUF loader
dequantises back to bf16 at compute time). PyTorch MPS availability is still
probed and reported, because torch is pulled in as an mflux dependency and
users troubleshooting "is my GPU working" expect to see it.

The primary target is Apple Silicon via Metal. MLX also ships Linux wheels and
mflux declares ``mlx[cuda13]`` on Linux, so an NVIDIA GPU is detected and used
where present. That path is supported but not verified on this machine — see
the Linux section of README.md.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Backend = Literal["mlx-metal", "mlx-cuda", "cpu"]

# Fraction of a discrete GPU's VRAM MLX is allowed to allocate. The remainder
# covers the driver's own context, whatever else is on the display, and
# allocator fragmentation. Exceeding the card aborts the process from inside
# the CUDA allocator, so this is deliberately conservative.
GPU_MEMORY_SAFETY_FACTOR = 0.92

# Room to leave beyond the weights on a device with a hard ceiling: activations,
# latents, and the preview and final VAE decodes.
#
# Measured on an RTX A5000 (24 GB) with FLUX.2 Klein 9B: 17.9 GB of weights
# peaked at 20.1 GB by the first denoise step at the 0.6 MP preset, and the
# 1.0 MP preset decodes larger tiles on top of that. 2 GB — enough on unified
# memory, where an overshoot merely pages — is not enough here.
HARD_LIMIT_HEADROOM_GB = 6.0


@dataclass(slots=True)
class DeviceInfo:
    """Everything we can learn about the machine we are about to run on."""

    chip: str
    backend: Backend
    metal_available: bool
    total_memory_gb: float
    cpu_cores: int
    gpu_cores: int | None
    macos_version: str
    python_version: str
    mlx_version: str
    torch_mps_available: bool | None
    os_name: str = "Darwin"
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None

    @property
    def is_apple_silicon(self) -> bool:
        return platform.machine() == "arm64" and platform.system() == "Darwin"

    @property
    def is_unified_memory(self) -> bool:
        """True when the GPU shares system RAM, as on Apple Silicon.

        Discrete GPUs are bounded by VRAM instead, which changes what the
        memory modes buy you.
        """
        return self.backend == "mlx-metal"

    @property
    def has_hard_memory_limit(self) -> bool:
        """True when overshooting aborts rather than swaps.

        A discrete GPU has nothing behind its VRAM: an over-allocation is fatal,
        not slow. That changes both how much headroom the memory policy has to
        reserve and whether the model is worth refusing up front.
        """
        return self.backend == "mlx-cuda" and bool(self.gpu_memory_gb)

    @property
    def device_memory_ceiling_gb(self) -> float:
        """The most the app may actually allocate, after the safety margin."""
        if self.has_hard_memory_limit:
            return self.usable_memory_gb * GPU_MEMORY_SAFETY_FACTOR
        return self.usable_memory_gb

    @property
    def usable_memory_gb(self) -> float:
        """Memory the model actually has to fit into."""
        if self.backend == "mlx-cuda" and self.gpu_memory_gb:
            return self.gpu_memory_gb
        return self.total_memory_gb

    @property
    def backend_label(self) -> str:
        return {
            "mlx-metal": "MLX / Metal (Apple GPU)",
            "mlx-cuda": "MLX / CUDA (NVIDIA GPU)",
            "cpu": "CPU",
        }[self.backend]

    def memory_advice(
        self, working_set_gb: float = 32.4, denoise_peak_gb: float = 16.9
    ) -> str:
        """Human-readable guidance on whether this machine fits the model."""
        available = self.usable_memory_gb
        kind = "RAM" if self.is_unified_memory else "VRAM"

        if available <= 0:
            return "Could not determine available memory."

        # Keeping everything resident needs room for the activations on top.
        # On a hard ceiling that margin is the difference between running and
        # aborting, so it is held to the same bar the memory policy uses.
        headroom = HARD_LIMIT_HEADROOM_GB if self.has_hard_memory_limit else 4.0
        if self.device_memory_ceiling_gb >= working_set_gb + headroom:
            return f"Comfortable — full model fits in {kind} with headroom."

        # Low mode keeps only the transformer resident during denoising.
        denoise_peak = denoise_peak_gb
        if available >= denoise_peak + 2:
            return (
                f"Tight — {working_set_gb:.0f} GB working set on "
                f"{available:.0f} GB {kind}. Use memory.mode: low "
                f"(~{denoise_peak:.0f} GB peak)."
            )
        return (
            f"Constrained — {working_set_gb:.0f} GB working set on "
            f"{available:.0f} GB {kind}. Expect "
            f"{'swapping' if self.is_unified_memory else 'OOM'}; prefer 512-768 px."
        )


def _sysctl(key: str) -> str | None:
    """Read a sysctl value, returning None if unavailable."""
    if not shutil.which("sysctl"):
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("sysctl %s failed: %s", key, exc)
        return None
    value = out.stdout.strip()
    return value or None


def _detect_chip() -> str:
    """Return the marketing name of the CPU/SoC, e.g. 'Apple M5'."""
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or (
            "Apple Silicon" if platform.machine() == "arm64" else "Intel"
        )

    # Linux: /proc/cpuinfo carries the model name on both x86 and arm.
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                for key in ("model name", "Model", "Hardware"):
                    if line.startswith(key):
                        _, _, value = line.partition(":")
                        if value.strip():
                            return value.strip()
    except OSError:
        pass

    return platform.processor() or platform.machine() or "Unknown"


def _detect_nvidia_gpu() -> tuple[str | None, float | None]:
    """(name, VRAM in GB) of the first NVIDIA GPU, via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("nvidia-smi failed: %s", exc)
        return None, None

    first = out.stdout.strip().splitlines()
    if not first:
        return None, None

    name, _, memory = first[0].partition(",")
    try:
        return name.strip(), float(memory.strip()) / 1024
    except ValueError:
        return name.strip() or None, None


def _detect_gpu_cores() -> int | None:
    """GPU core count via system_profiler. Returns None if it can't be read."""
    if platform.system() != "Darwin" or not shutil.which("system_profiler"):
        return None
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("system_profiler failed: %s", exc)
        return None

    for line in out.stdout.splitlines():
        if "Total Number of Cores" in line:
            _, _, value = line.partition(":")
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                return int(digits)
    return None


def _detect_total_memory_gb() -> float:
    raw = _sysctl("hw.memsize")
    if raw and raw.isdigit():
        return int(raw) / 1024**3
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return 0.0


def _detect_gpu_backend() -> tuple[Backend, bool, str]:
    """Probe MLX for a usable GPU.

    Returns ``(backend, metal_available, mlx_version)``. Metal and CUDA are
    both MLX GPU devices; which one you get depends on the wheel installed
    (``mlx-metal`` on macOS, ``mlx[cuda13]`` on Linux).
    """
    try:
        import mlx.core as mx
    except ImportError as exc:
        logger.warning("MLX is not installed: %s", exc)
        return "cpu", False, "not installed"

    version = getattr(mx, "__version__", "unknown")

    metal = False
    try:
        if hasattr(mx, "metal") and hasattr(mx.metal, "is_available"):
            metal = bool(mx.metal.is_available())
    except Exception as exc:  # noqa: BLE001 - probing must never crash startup
        logger.debug("Metal probe failed: %s", exc)

    if metal:
        return "mlx-metal", True, version

    # No Metal: check whether MLX resolved a GPU device anyway (CUDA on Linux).
    try:
        if mx.default_device().type == mx.DeviceType.gpu:
            backend: Backend = "mlx-metal" if platform.system() == "Darwin" else "mlx-cuda"
            return backend, backend == "mlx-metal", version
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not inspect the MLX default device: %s", exc)

    return "cpu", False, version


def _detect_torch_mps() -> bool | None:
    """Report PyTorch MPS availability. None if torch isn't importable."""
    try:
        import torch
    except ImportError:
        return None
    try:
        return bool(torch.backends.mps.is_available())
    except Exception as exc:  # noqa: BLE001
        logger.debug("torch MPS probe failed: %s", exc)
        return False


def detect_device() -> DeviceInfo:
    """Detect the best available backend.

    Priority: MLX/Metal (Apple GPU) -> MLX/CUDA (NVIDIA, Linux) -> CPU.
    """
    backend, metal_available, mlx_version = _detect_gpu_backend()
    gpu_name, gpu_memory_gb = (None, None)
    if backend == "mlx-cuda" or platform.system() != "Darwin":
        gpu_name, gpu_memory_gb = _detect_nvidia_gpu()

    info = DeviceInfo(
        chip=_detect_chip(),
        backend=backend,
        metal_available=metal_available,
        total_memory_gb=_detect_total_memory_gb(),
        cpu_cores=os.cpu_count() or 0,
        gpu_cores=_detect_gpu_cores(),
        macos_version=platform.mac_ver()[0] or platform.release(),
        python_version=platform.python_version(),
        mlx_version=mlx_version,
        torch_mps_available=_detect_torch_mps(),
        os_name=platform.system(),
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )

    if backend == "cpu":
        if info.is_apple_silicon:
            logger.warning(
                "Metal is unavailable, falling back to CPU. A single edit may take "
                "well over an hour. See the troubleshooting section in README.md."
            )
        elif platform.system() == "Linux":
            logger.warning(
                "No MLX GPU device found. On Linux this app needs an NVIDIA GPU "
                "with the CUDA build: pip install 'mlx[cuda13]'. Falling back to "
                "CPU, which is impractically slow for a 20B model."
            )
        else:
            logger.warning(
                "No GPU backend available on %s / %s — expect very slow CPU-only "
                "inference.",
                platform.system(),
                platform.machine(),
            )
    elif backend == "mlx-cuda":
        logger.info(
            "Using the MLX CUDA backend%s. This path is supported but less "
            "exercised than Apple Silicon; see the Linux notes in README.md.",
            f" on {gpu_name}" if gpu_name else "",
        )

    return info


def apply_memory_settings(cache_limit_bytes: int | None) -> None:
    """Bound MLX's buffer cache.

    Without a limit, MLX retains freed GPU buffers across denoise passes, so a
    long editing session creeps upward until the machine starts swapping.
    """
    if cache_limit_bytes is None:
        return
    try:
        import mlx.core as mx

        mx.set_cache_limit(int(cache_limit_bytes))
        mx.clear_cache()
        logger.debug("MLX cache limit set to %.2f GB", cache_limit_bytes / 1e9)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not set MLX cache limit: %s", exc)


def peak_memory_gb() -> float:
    """Peak MLX allocation so far, in GB. Returns 0.0 if unavailable."""
    try:
        import mlx.core as mx

        return mx.get_peak_memory() / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


def active_memory_gb() -> float:
    """Currently-allocated MLX memory, in GB. Returns 0.0 if unavailable."""
    try:
        import mlx.core as mx

        return mx.get_active_memory() / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


def reset_peak_memory() -> None:
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except Exception:  # noqa: BLE001
        pass


def startup_banner(
    info: DeviceInfo,
    model_label: str,
    memory_mode: str,
    working_set_gb: float = 32.4,
    denoise_peak_gb: float = 16.9,
) -> str:
    """Render the startup information block.

    The footprints come from the configured model family, so the memory advice
    reflects the model actually being loaded.
    """
    if info.backend == "mlx-cuda":
        title = "Local Image Edit — MLX / CUDA"
        gpu = info.gpu_name or "NVIDIA GPU"
        memory_line = (
            f"  Memory        {info.gpu_memory_gb:.0f} GB VRAM, "
            f"{info.total_memory_gb:.0f} GB system"
            if info.gpu_memory_gb
            else f"  Memory        {info.total_memory_gb:.0f} GB system"
        )
    else:
        title = "Local Image Edit — Apple Silicon"
        gpu = f"{info.gpu_cores}-core GPU" if info.gpu_cores else "GPU"
        memory_line = f"  Memory        {info.total_memory_gb:.0f} GB unified"

    if info.os_name == "Darwin":
        platform_line = f"macOS {info.macos_version}"
        mps = {True: "available", False: "unavailable", None: "torch not installed"}[
            info.torch_mps_available
        ]
        platform_suffix = f"   torch MPS: {mps}"
    else:
        platform_line = f"{info.os_name} {info.macos_version}"
        platform_suffix = ""

    inner = 59
    lines = [
        "",
        f"  ┌{'─' * inner}┐",
        f"  │  {title.ljust(inner - 2)}│",
        f"  └{'─' * inner}┘",
        "",
        f"  Device        {info.chip}  ({info.cpu_cores}-core CPU, {gpu})",
        f"  Backend       {info.backend_label}",
        f"  Model         {model_label}",
        f"  Memory mode   {memory_mode}",
        memory_line,
        f"                {info.memory_advice(working_set_gb, denoise_peak_gb)}",
        "",
        f"  {platform_line}   Python {info.python_version}   "
        f"MLX {info.mlx_version}{platform_suffix}",
        "",
    ]
    return "\n".join(lines)
