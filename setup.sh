#!/usr/bin/env bash
#
# One-command setup. Apple Silicon Mac, or Linux with an NVIDIA GPU.
#
#   ./setup.sh                              full setup: deps + ~32 GB of weights
#   ./setup.sh --no-model                   environment only, fetch weights later
#   ./setup.sh --model-only                 weights only (deps already installed)
#   ./setup.sh --mirror https://hf-mirror.com   use a Hub mirror
#   ./setup.sh --import-from /media/usb/models   copy weights from a drive
#   ./setup.sh --check                      verify an existing install
#
# Safe to re-run: dependency install and weight download both resume, so an
# interrupted setup continues rather than starting over.
#
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.requirements-stamp"

DO_DEPS=1
DO_MODEL=1
CHECK_ONLY=0
MIRROR=""
IMPORT_FROM=""
WORKERS=4
DISABLE_XET=""

BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GREEN=$'\033[32m'
YELLOW=$'\033[33m'; RESET=$'\033[0m'
[[ -t 1 ]] || { BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""; }

step() { printf '\n%s==>%s %s%s%s\n' "$BOLD" "$RESET" "$BOLD" "$*" "$RESET"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '    %s⚠  %s%s\n' "$YELLOW" "$*" "$RESET"; }
ok()   { printf '    %s✓%s  %s\n' "$GREEN" "$RESET" "$*"; }
fail() { printf '\n    %s✗  %s%s\n\n' "$RED" "$*" "$RESET" >&2; exit 1; }

usage() { sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-model)    DO_MODEL=0 ;;
    --model-only)  DO_DEPS=0 ;;
    --check)       CHECK_ONLY=1 ;;
    --no-xet)      DISABLE_XET=1 ;;
    --mirror)      MIRROR="${2:?--mirror needs a URL}"; shift ;;
    --import-from) IMPORT_FROM="${2:?--import-from needs a path}"; shift ;;
    --jobs)        WORKERS="${2:?--jobs needs a number}"; shift ;;
    -h|--help)     usage ;;
    *)             fail "Unknown option: $1  (try --help)" ;;
  esac
  shift
done

printf '\n  %sQwen Image Edit — setup%s\n' "$BOLD" "$RESET"
printf '  %s%s%s\n' "$DIM" "$PROJECT_DIR" "$RESET"

# ---------------------------------------------------------------- platform
step "Checking the machine"

OS="$(uname -s)"
ARCH="$(uname -m)"
PLATFORM="unsupported"

case "$OS" in
  Darwin)
    if [[ "$ARCH" == "arm64" ]]; then
      PLATFORM="apple"
      CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
      MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
      ok "$CHIP, ${MEM_GB} GB unified memory"
      if (( MEM_GB > 0 && MEM_GB < 16 )); then
        warn "${MEM_GB} GB is below the practical minimum. Expect heavy swapping."
      elif (( MEM_GB > 0 && MEM_GB < 32 )); then
        info "${DIM}Under 32 GB: keep memory.mode: low in config.yaml.${RESET}"
      fi
    else
      PLATFORM="cpu"
      warn "Intel Mac ($ARCH) — no GPU backend. A single edit may take hours."
    fi
    ;;
  Linux)
    # MLX ships Linux wheels and mflux declares mlx[cuda13] there, so an
    # NVIDIA GPU works. Supported, but less exercised than Apple Silicon.
    if command -v nvidia-smi >/dev/null 2>&1; then
      PLATFORM="cuda"
      GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
      ok "Linux with NVIDIA GPU: ${GPU_INFO:-detected}"
      VRAM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
      if [[ -n "${VRAM_MB:-}" ]] && (( VRAM_MB < 18000 )); then
        warn "$((VRAM_MB / 1024)) GB VRAM. The denoise phase needs ~17 GB even in"
        warn "low memory mode — expect OOM above 768 px."
      fi
      info "${DIM}Note: the Linux/CUDA path is supported but untested by the author.${RESET}"
    else
      PLATFORM="cpu"
      warn "Linux without nvidia-smi. MLX needs an NVIDIA GPU here; CPU-only"
      warn "inference on a 20B model is impractically slow."
    fi
    MEM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1048576 ))
    (( MEM_GB > 0 )) && info "${DIM}${MEM_GB} GB system RAM${RESET}"
    ;;
  *)
    fail "Unsupported OS: $OS. This app runs on macOS (Apple Silicon) or Linux (NVIDIA)."
    ;;
esac

if (( DO_MODEL )); then
  if [[ "$OS" == "Darwin" ]]; then
    FREE_GB=$(df -g "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
  else
    FREE_GB=$(df -BG "$PROJECT_DIR" 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
  fi
  if [[ -n "${FREE_GB:-}" ]] && [[ "$FREE_GB" =~ ^[0-9]+$ ]]; then
    if (( FREE_GB < 35 )); then
      fail "Only ${FREE_GB} GB free. The model needs ~32 GB (35 GB recommended)."
    fi
    ok "${FREE_GB} GB free disk"
  fi
fi

# ------------------------------------------------------------------ python
find_python() {
  for candidate in python3.12 python3.13 python3.11 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null \
      && { command -v "$candidate"; return 0; }
  done
  return 1
}

if (( DO_DEPS )); then
  step "Setting up the Python environment"

  if [[ ! -x "$PYTHON_BIN" ]]; then
    if [[ "$OS" == "Darwin" ]]; then
      PY_HINT="brew install python@3.12"
    else
      PY_HINT="sudo apt install python3.12 python3.12-venv   # or your distro's equivalent"
    fi
    BOOTSTRAP="$(find_python)" || fail "Python 3.11+ not found. Install it with:  $PY_HINT"
    info "Using $("$BOOTSTRAP" -V) at $BOOTSTRAP"
    "$BOOTSTRAP" -m venv "$VENV_DIR" || fail "Could not create a virtualenv at $VENV_DIR"
    ok "Created venv/"
  else
    ok "venv/ already exists ($("$PYTHON_BIN" -V 2>&1))"
  fi

  if command -v shasum >/dev/null 2>&1; then
    REQ_HASH="$(shasum -a 256 requirements.txt | cut -d' ' -f1)"
  else
    REQ_HASH="$(sha256sum requirements.txt | cut -d' ' -f1)"
  fi
  if [[ "$(cat "$STAMP" 2>/dev/null)" == "$REQ_HASH" ]]; then
    ok "Dependencies already up to date"
  else
    info "Installing dependencies (a few hundred MB; retries on network errors)…"
    installed=0
    for attempt in 1 2 3; do
      if command -v uv >/dev/null 2>&1; then
        VIRTUAL_ENV="$VENV_DIR" uv pip install -q -r requirements.txt && { installed=1; break; }
      else
        "$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || \
          "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1
        "$PYTHON_BIN" -m pip install -q --upgrade pip >/dev/null 2>&1
        "$PYTHON_BIN" -m pip install -q -r requirements.txt && { installed=1; break; }
      fi
      warn "Install attempt $attempt failed; retrying in $((attempt * 5))s…"
      sleep $((attempt * 5))
    done
    (( installed )) || fail "Dependency installation failed after 3 attempts."
    printf '%s' "$REQ_HASH" > "$STAMP"
    ok "Dependencies installed"
  fi
fi

[[ -x "$PYTHON_BIN" ]] || fail "No virtualenv at $VENV_DIR. Run without --model-only first."

# --------------------------------------------------------------------- gpu
step "Checking GPU acceleration"
GPU_PROBE='
import sys
try:
    import mlx.core as mx
except ImportError:
    print("mlx-missing"); sys.exit(1)
try:
    if hasattr(mx, "metal") and mx.metal.is_available():
        print("metal"); sys.exit(0)
except Exception:
    pass
try:
    if mx.default_device().type == mx.DeviceType.gpu:
        print("gpu"); sys.exit(0)
except Exception:
    pass
print("cpu"); sys.exit(1)
'
GPU_RESULT="$("$PYTHON_BIN" -c "$GPU_PROBE" 2>/dev/null)" || GPU_RESULT="${GPU_RESULT:-cpu}"

case "$GPU_RESULT" in
  metal) ok "Metal GPU available (MLX)" ;;
  gpu)   ok "MLX GPU device available (CUDA)" ;;
  mlx-missing) warn "MLX is not installed — run without --model-only first." ;;
  *)
    warn "No GPU backend — inference would run on CPU."
    if [[ "$OS" == "Darwin" ]]; then
      warn "Usually a non-native Python (Rosetta). See README troubleshooting."
    else
      warn "On Linux, MLX needs the CUDA build:  pip install 'mlx[cuda13]'"
    fi
    ;;
esac

# ------------------------------------------------------------------- model
if (( CHECK_ONLY )); then
  step "Verifying the installation"
  "$PYTHON_BIN" app.py --check || true
  "$PYTHON_BIN" app.py --verify
  exit $?
fi

if (( DO_MODEL )); then
  if [[ -n "$IMPORT_FROM" ]]; then
    step "Importing weights from $IMPORT_FROM"
    "$PYTHON_BIN" app.py --import-from "$IMPORT_FROM" || fail "Import failed."
    ok "Weights imported"
  else
    step "Downloading the model (~32 GB)"
    info "Resumable — safe to interrupt and re-run this script."
    info "${DIM}Watch progress from another terminal: ./watch-download.py${RESET}"
    [[ -n "$MIRROR" ]] && info "Mirror: $MIRROR"

    DOWNLOAD_ARGS=(--download --jobs "$WORKERS")
    [[ -n "$MIRROR" ]] && DOWNLOAD_ARGS+=(--mirror "$MIRROR")
    [[ -n "$DISABLE_XET" ]] && DOWNLOAD_ARGS+=(--no-xet)

    if ! "$PYTHON_BIN" app.py "${DOWNLOAD_ARGS[@]}"; then
      printf '\n'
      warn "The download did not finish."
      warn "Re-run ./setup.sh --model-only to resume from where it stopped."
      warn "On a restricted network try:  ./setup.sh --model-only --mirror <url>"
      warn "Or copy models/ from another Mac:  ./setup.sh --import-from <path>"
      exit 1
    fi
    ok "Model downloaded and verified"
  fi
fi

# -------------------------------------------------------------------- done
step "Ready"
"$PYTHON_BIN" app.py --check || true

cat <<EOF

    Start the app:      ./run.sh
    Watch a download:   ./watch-download.py
    Run the tests:      ./venv/bin/python -m pytest tests/ -q

EOF
