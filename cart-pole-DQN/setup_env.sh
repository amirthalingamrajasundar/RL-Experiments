#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# Environment Setup: DQN, DDQN, PER
# - Creates conda env "dqn_env" with Python 3.10
# - Installs required packages via pip
# ==========================================

ENV_NAME="tala_env"
PYTHON_VERSION="3.10"

# Packages to install with pip (list form)
PIP_PKGS=(
  numpy
  torch
  matplotlib
  tqdm
  swig
  "gymnasium[box2d,other]"
)

# Helper: print message
info() { printf "\n[INFO] %s\n" "$*"; }
err()  { printf "\n[ERROR] %s\n" "$*" >&2; }

# 1) Ensure conda exists; if not, try to install Miniconda on macOS/Linux
if command -v conda >/dev/null 2>&1; then
  info "Found conda: $(command -v conda)"
else
  OS_TYPE="$(uname -s)"
  info "conda not found on PATH. Detected OS: ${OS_TYPE}"
  case "$OS_TYPE" in
    Linux|Darwin)
      MINICONDA_DIR="$HOME/miniconda3"
      if [ -d "$MINICONDA_DIR" ]; then
        info "Found existing Miniconda at $MINICONDA_DIR. Adding to PATH for this session."
        export PATH="$MINICONDA_DIR/bin:$PATH"
      else
        info "Downloading and installing Miniconda (non-interactive) to $MINICONDA_DIR..."
        # Choose installer URL
        if [ "$OS_TYPE" = "Darwin" ]; then
          INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
        else
          INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
        fi
        TMP_DIR=$(mktemp -d)
        INSTALLER_PATH="$TMP_DIR/$INSTALLER"
        if command -v curl >/dev/null 2>&1; then
          curl -fsSL -o "$INSTALLER_PATH" "https://repo.anaconda.com/miniconda/$INSTALLER"
        elif command -v wget >/dev/null 2>&1; then
          wget -q -O "$INSTALLER_PATH" "https://repo.anaconda.com/miniconda/$INSTALLER"
        else
          err "Neither curl nor wget is available to download Miniconda. Please install Miniconda manually."
          exit 1
        fi
        bash "$INSTALLER_PATH" -b -p "$MINICONDA_DIR"
        rm -rf "$TMP_DIR"
        export PATH="$MINICONDA_DIR/bin:$PATH"
        info "Miniconda installed to $MINICONDA_DIR and added to PATH for this session."
        # Initialize conda for bash/zsh if available (safe to run)
        if command -v conda >/dev/null 2>&1; then
          # shellcheck disable=SC2016
          CONDA_SHELL_INIT="$("$MINICONDA_DIR/bin/conda" init --all 2>/dev/null || true)"
        fi
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
      err "Automatic Miniconda installation is not supported for Windows in this script."
      err "Please download and install Miniconda/Anaconda manually from:"
      err "  https://docs.conda.io/en/latest/miniconda.html"
      err "Then re-run this script from an Anaconda Prompt or Git Bash."
      exit 1
      ;;
    *)
      err "Unsupported OS: $OS_TYPE. Please install conda/miniconda manually."
      exit 1
      ;;
  esac

  # Final check
  if ! command -v conda >/dev/null 2>&1; then
    err "conda still not available in PATH after attempted install. Please open a new shell or run 'source \$HOME/miniconda3/bin/activate' then retry."
    exit 1
  fi
fi

# 2) Create (or update) the conda env with Python 3.10
info "Creating/updating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
# Use --yes and allow overwriting if exists
if conda env list | grep -qE "^[^#]*\b${ENV_NAME}\b"; then
  info "Environment ${ENV_NAME} already exists. Will update Python version if needed."
  conda install -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
else
  conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# 3) Activate the environment in this script
# Use conda's activate mechanism
# shellcheck disable=SC1091
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # Miniconda typical location
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" ]; then
  . "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
else
  err "Could not find conda.sh to source. Activating via 'conda activate' may require a new shell."
fi

info "Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# 4) Install required pip packages
info "Installing required Python packages via pip..."
# Use pip install -U to upgrade if needed
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "${PIP_PKGS[@]}"

# 5) Quick verification
info "Verifying installations..."
python - <<'PY'
import sys
packages = ["numpy","torch","matplotlib","tqdm","gymnasium"]
miss = []
for p in packages:
    try:
        __import__(p)
    except Exception as e:
        miss.append((p,str(e)))
if miss:
    print("Some imports failed:", miss, file=sys.stderr)
    sys.exit(2)
print("Setup successful. Python:", sys.version.splitlines()[0])
PY

info "Environment setup complete."
info "To begin, run: conda activate ${ENV_NAME}"

