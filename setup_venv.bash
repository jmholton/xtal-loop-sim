#!/bin/bash
# setup_venv.bash: build .venv/ next to this script from requirements.txt and run the
# test suite. Works unchanged on a dev box and on the beamline hosts (voltron,
# dataserver3: CentOS 7, glibc 2.17, gcc 4.8.5), where every package must come as a
# wheel and the base interpreter is the PyTorch bundle's python3.10.
#
#   bash setup_venv.bash                 # build if missing, then pytest
#   bash setup_venv.bash --force         # rebuild the venv first
#   bash setup_venv.bash --skip-tests
#   bash setup_venv.bash --acceptance    # also run tools/acceptance_voltron.py (GPU)
#
# voltron's login shell is tcsh: run it through bash as above. torch.compile (the
# compiled preview path) needs a modern compiler at runtime; when devtoolset-7 is
# present this script exports CC/CXX for the test run, and the RUNBOOK says to export
# them before launching a live-render server.

set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

VENV=".venv"
FORCE=0
SKIP_TESTS=0
ACCEPTANCE=0
for arg in "$@"; do
    case "$arg" in
        --force)       FORCE=1 ;;
        --skip-tests)  SKIP_TESTS=1 ;;
        --acceptance)  ACCEPTANCE=1 ;;
        *) echo "unknown argument: $arg (known: --force, --skip-tests, --acceptance)" >&2
           exit 1 ;;
    esac
done

# Base interpreter: the beamline's PyTorch bundle when present, else the system
# python3 (never a conda python that happens to be first on PATH: pillow 10.4.0 ships
# no wheel for 3.13, so a 3.13 base fails the --only-binary install).
BASE_PY="${LOOPSIM_BASE_PYTHON:-}"
if [[ -z "$BASE_PY" ]]; then
    if [[ -x /home/programs/pytorch/envs/pt/bin/python3.10 ]]; then
        BASE_PY=/home/programs/pytorch/envs/pt/bin/python3.10
    elif [[ -x /usr/bin/python3 ]]; then
        BASE_PY=/usr/bin/python3
    else
        BASE_PY=python3
    fi
fi

if [[ -d "$VENV" && "$FORCE" -eq 1 ]]; then
    echo "[setup] --force: removing $VENV"
    rm -rf "$VENV"
fi
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "[setup] creating $VENV with $BASE_PY"
    "$BASE_PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install --only-binary=:all: \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r requirements.txt
else
    echo "[setup] $VENV exists (--force to rebuild)"
fi

"$VENV/bin/python" - <<'EOF'
import torch, numpy, PIL
print(f"[setup] torch {torch.__version__}, cuda available: {torch.cuda.is_available()}, "
      f"numpy {numpy.__version__}, pillow {PIL.__version__}")
EOF

if [[ -x /opt/rh/devtoolset-7/root/usr/bin/gcc ]]; then
    export CC=/opt/rh/devtoolset-7/root/usr/bin/gcc
    export CXX=/opt/rh/devtoolset-7/root/usr/bin/g++
    echo "[setup] using devtoolset-7 for torch.compile"
fi

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    "$VENV/bin/python" -m pytest tests/ -q
fi
if [[ "$ACCEPTANCE" -eq 1 ]]; then
    "$VENV/bin/python" tools/acceptance_voltron.py
fi
echo "[setup] done: run everything with $VENV/bin/python"
