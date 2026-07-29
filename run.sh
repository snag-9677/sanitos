#!/usr/bin/env bash
#
# Launch the Qwen Image Edit app, creating the virtualenv and installing
# dependencies on first run.
#
#   ./run.sh                  launch the UI
#   ./run.sh --preload        download/load weights first, then launch
#   ./run.sh --check          environment report only
#   ./run.sh --verify         check cached weights for truncation
#   ./run.sh --port 7861      any app.py flag passes straight through
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.requirements-stamp"

info()  { printf '  %s\n' "$*"; }
fail()  { printf '\n  ❌ %s\n\n' "$*" >&2; exit 1; }

# --- platform guardrails --------------------------------------------------
# Primary target is Apple Silicon; MLX also has Linux/CUDA wheels, which mflux
# declares, so Linux with an NVIDIA GPU is supported too.
case "$(uname -s)" in
  Darwin)
    if [[ "$(uname -m)" != "arm64" ]]; then
      printf '\n  ⚠️  Not an Apple Silicon Mac (%s). Inference will fall back to CPU\n' "$(uname -m)"
      printf '     and a single edit may take over an hour.\n\n'
    fi
    ;;
  Linux)
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      printf '\n  ⚠️  No NVIDIA GPU detected. On Linux this app needs the MLX CUDA\n'
      printf '     build; CPU-only inference on a 20B model is impractically slow.\n\n'
    fi
    ;;
  *)
    fail "Unsupported OS: $(uname -s). Requires macOS (Apple Silicon) or Linux (NVIDIA)."
    ;;
esac

# --- interpreter ----------------------------------------------------------
find_python() {
  # Prefer 3.12, then 3.13/3.11, then whatever python3 is; mflux needs >= 3.10
  # but this app is written against 3.11+ syntax.
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        command -v "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  BOOTSTRAP_PYTHON="$(find_python)" || fail "Python 3.11+ not found. Install it with: brew install python@3.12"
  info "Creating virtualenv with $("$BOOTSTRAP_PYTHON" -V) …"
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR" || fail "Could not create a virtualenv at $VENV_DIR"
fi

# --- dependencies ---------------------------------------------------------
# On Linux the CUDA override is a separate, second install. It overrides an
# mflux pin, and pip solves one requirements file as a single problem, so
# merging the two files fails with ResolutionImpossible instead of warning.
install_cuda_override() {
  [[ "$(uname -s)" == "Linux" ]] || return 0
  if command -v uv >/dev/null 2>&1; then
    VIRTUAL_ENV="$VENV_DIR" uv pip install --quiet -r requirements-cuda.txt && return 0
  fi
  "$PYTHON_BIN" -m pip install --quiet -r requirements-cuda.txt
}

install_requirements() {
  # uv is dramatically faster and is also what creates pip-less venvs, so try
  # it first and fall back to pip (bootstrapping pip if the venv lacks it).
  if command -v uv >/dev/null 2>&1; then
    VIRTUAL_ENV="$VENV_DIR" uv pip install --quiet -r requirements.txt && \
      install_cuda_override && return 0
  fi
  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    info "Bootstrapping pip …"
    "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || return 1
  fi
  "$PYTHON_BIN" -m pip install --quiet --upgrade pip
  "$PYTHON_BIN" -m pip install --quiet -r requirements.txt && install_cuda_override
}

# Reinstall when either requirements file changes. shasum is the macOS spelling
# and sha256sum the Linux one; neither is reliably present on the other.
if command -v shasum >/dev/null 2>&1; then
  REQ_HASH="$(cat requirements.txt requirements-cuda.txt | shasum -a 256 | cut -d' ' -f1)"
else
  REQ_HASH="$(cat requirements.txt requirements-cuda.txt | sha256sum | cut -d' ' -f1)"
fi
if [[ ! -f "$STAMP" ]] || [[ "$(cat "$STAMP" 2>/dev/null)" != "$REQ_HASH" ]]; then
  info "Installing dependencies (first run downloads a few hundred MB) …"
  install_requirements || fail "Dependency installation failed."
  printf '%s' "$REQ_HASH" > "$STAMP"
  info "Dependencies ready."
fi

exec "$PYTHON_BIN" app.py "$@"
