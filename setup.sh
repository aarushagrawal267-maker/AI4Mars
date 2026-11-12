#!/usr/bin/env bash
# One-time setup of Mars Terrain Intelligence on macOS / Linux.
#
# Usage: ./setup.sh [--cuda | --cpu-only] [--with-e2e]
#   --cuda       install the CUDA 12.4 PyTorch build even if nvidia-smi is not found (Linux only)
#   --cpu-only   install the CPU-only PyTorch build even if an NVIDIA GPU is present
#   --with-e2e   also download the Playwright Chromium browser for the end-to-end tests
#
# Set PYTHON=/path/to/python3 to choose a specific interpreter. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
VENV_PY="$ROOT/.venv/bin/python"

FORCE_CUDA=0
CPU_ONLY=0
WITH_E2E=0
for arg in "$@"; do
  case "$arg" in
    --cuda) FORCE_CUDA=1 ;;
    --cpu-only) CPU_ONLY=1 ;;
    --with-e2e) WITH_E2E=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done

if [ -t 1 ]; then
  C_STEP=$'\033[36m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[90m'; C_OFF=$'\033[0m'
else
  C_STEP=""; C_OK=""; C_WARN=""; C_ERR=""; C_DIM=""; C_OFF=""
fi
step() { printf '\n%s==> %s%s\n' "$C_STEP" "$1" "$C_OFF"; }
ok()   { printf '    %s[ok]%s %s\n' "$C_OK" "$C_OFF" "$1"; }
info() { printf '    %s\n' "$1"; }
warn() { printf '    %s[warning]%s %s\n' "$C_WARN" "$C_OFF" "$1"; }
fail() { printf '\n%s[setup failed]%s %s\n' "$C_ERR" "$C_OFF" "$1" >&2; exit 1; }

[ "$FORCE_CUDA" = 1 ] && [ "$CPU_ONLY" = 1 ] && fail "Use either --cuda or --cpu-only, not both."

echo "Mars Terrain Intelligence - setup"
printf '%sIndependent student research project. Not a NASA system and not connected to any rover.%s\n' "$C_DIM" "$C_OFF"

# ----------------------------------------------------------------------------- 1. Python
step "Checking Python (3.10 or newer required)"
PY=""
for cand in ${PYTHON:-} python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    PY="$(command -v "$cand")"
    break
  fi
done
[ -n "$PY" ] || fail "Python 3.10 or newer was not found. Install it (e.g. from https://www.python.org/downloads/, Homebrew or your package manager) and run ./setup.sh again."
ok "Python $("$PY" -c 'import platform; print(platform.python_version())') ($PY)"

# ----------------------------------------------------------------------------- 2. Node.js and npm
step "Checking Node.js (18 or newer) and npm"
command -v node >/dev/null 2>&1 || fail "Node.js was not found. Install the LTS version from https://nodejs.org/ (or via nvm/Homebrew)."
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
[ "$NODE_MAJOR" -ge 18 ] 2>/dev/null || fail "Node.js $(node --version 2>/dev/null) is too old. Install Node.js 18 or newer (LTS)."
ok "Node.js $(node --version)"
command -v npm >/dev/null 2>&1 || fail "npm was not found. It is installed together with Node.js."
ok "npm $(npm --version)"

# ----------------------------------------------------------------------------- 3. virtual environment
step "Creating the Python virtual environment (.venv)"
if [ -x "$VENV_PY" ]; then
  ok ".venv already exists; reusing it"
else
  "$PY" -m venv .venv || fail "Could not create .venv. On Debian/Ubuntu install the venv module first: sudo apt install python3-venv"
  [ -x "$VENV_PY" ] || fail ".venv was created but $VENV_PY is missing."
  ok ".venv created"
fi

step "Upgrading pip"
"$VENV_PY" -m pip install --upgrade pip || fail "pip upgrade failed. Check your internet connection or proxy settings."
ok "pip is up to date"

# ----------------------------------------------------------------------------- 4. PyTorch
step "Selecting the PyTorch build"
OS_NAME="$(uname -s)"
GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
fi
if [ "$CPU_ONLY" = 1 ]; then
  WANT="cpu"; info "--cpu-only given: installing the CPU build."
elif [ "$FORCE_CUDA" = 1 ]; then
  [ "$OS_NAME" = "Darwin" ] && fail "CUDA builds of PyTorch are not available for macOS."
  WANT="cuda"; info "--cuda given: installing the CUDA 12.4 build."
elif [ -n "$GPU_NAME" ]; then
  WANT="cuda"; info "NVIDIA GPU detected: $GPU_NAME"
else
  WANT="cpu"; info "No NVIDIA GPU detected (nvidia-smi not found or not working): installing the CPU build."
fi

CURRENT="$("$VENV_PY" -c "import torch; print('cuda' if torch.version.cuda else 'cpu')" 2>/dev/null || true)"
if [ "$CURRENT" = "$WANT" ]; then
  ok "PyTorch ($WANT build) is already installed"
else
  if [ -n "$CURRENT" ]; then
    info "Replacing the installed $CURRENT build of PyTorch with the $WANT build."
    "$VENV_PY" -m pip uninstall -y torch torchvision || fail "Could not uninstall the existing PyTorch build."
  fi
  if [ "$OS_NAME" = "Darwin" ]; then
    # On macOS the standard PyPI wheels are the official build. This app uses CUDA or the CPU (not Apple MPS).
    info "Installing torch + torchvision from PyPI (macOS build; this can take several minutes)"
    "$VENV_PY" -m pip install torch torchvision || fail "PyTorch installation failed."
  else
    if [ "$WANT" = "cuda" ]; then INDEX="$TORCH_CUDA_INDEX"; else INDEX="$TORCH_CPU_INDEX"; fi
    info "Installing torch + torchvision from $INDEX (this can take several minutes)"
    "$VENV_PY" -m pip install torch torchvision --index-url "$INDEX" || \
      fail "PyTorch installation from $INDEX failed. Check your connection; with an older GPU driver try --cpu-only."
  fi
  ok "PyTorch ($WANT build) installed"
fi

# ----------------------------------------------------------------------------- 5. Python requirements
step "Installing Python requirements (requirements.txt)"
"$VENV_PY" -m pip install -r requirements.txt || fail "pip install -r requirements.txt failed."
ok "Python requirements installed"

# ----------------------------------------------------------------------------- 6. frontend packages
step "Installing frontend packages"
[ -f frontend/package.json ] || fail "frontend/package.json was not found."
if [ -f frontend/package-lock.json ]; then
  (cd frontend && npm ci) || fail "npm ci failed in frontend/."
else
  (cd frontend && npm install) || fail "npm install failed in frontend/."
fi
ok "Frontend packages installed"

# ----------------------------------------------------------------------------- 7. folders and .env
step "Creating local data folders"
for d in data/raw/imports data/processed data/analyses checkpoints experiments; do
  mkdir -p "$d"
  ok "$d"
done

step "Configuration file (.env)"
if [ -f .env ]; then
  ok ".env already exists; left unchanged"
elif [ -f .env.example ]; then
  cp .env.example .env
  ok ".env created from .env.example"
else
  warn ".env.example not found; built-in defaults will be used."
fi

# ----------------------------------------------------------------------------- 8. database
step "Initialising the SQLite database"
"$VENV_PY" -c "from backend.app.core.config import get_settings; from backend.app.database.db import Database; s=get_settings(); s.ensure_dirs(); Database(s.db_url); print('database ready')" \
  || fail "Database initialisation failed."

# ----------------------------------------------------------------------------- 9. hardware diagnostics
step "Hardware diagnostics"
"$VENV_PY" -c "import os, torch, psutil; c=torch.cuda.is_available(); print('    torch', torch.__version__, '| CUDA build:', torch.version.cuda); print('    CUDA available:', c); print('    GPU:', torch.cuda.get_device_name(0) if c else 'none'); print('    CPU cores (logical):', os.cpu_count()); print('    RAM (GB):', round(psutil.virtual_memory().total / 1e9, 1))" \
  || fail "Could not import torch/psutil from .venv."
if [ "$WANT" = "cuda" ] && [ "$("$VENV_PY" -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)" != "True" ]; then
  warn "The CUDA build is installed but PyTorch cannot use the GPU. Update the NVIDIA driver, or run ./setup.sh --cpu-only."
fi

# ----------------------------------------------------------------------------- 10. optional e2e browser
if [ "$WITH_E2E" = 1 ]; then
  step "Installing the Playwright Chromium browser (for npm run e2e)"
  (cd frontend && npx playwright install chromium) || \
    fail "Playwright browser installation failed. On Linux, missing system libraries can be installed with: (cd frontend && sudo npx playwright install-deps chromium)"
  ok "Chromium for Playwright installed"
fi

# ----------------------------------------------------------------------------- done
printf '\n%sSetup complete.%s\n' "$C_OK" "$C_OFF"
echo "Next steps:"
echo "  1. Start the app:       ./start.sh        (add --open to open the browser)"
echo "  2. Open the web app:    http://localhost:5173"
echo "  3. API documentation:   http://127.0.0.1:8000/docs"
echo "  4. Stop everything:     ./stop.sh"
echo "  Tests: .venv/bin/python -m pytest   (see docs/testing.md)"
