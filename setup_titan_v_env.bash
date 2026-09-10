#!/bin/bash
# setup_titan_v_env.bash: build the torch-2.6 + devtoolset-7 venv that the compiled
# preview path needs on voltron (the beamline's TITAN V), then run
# acceptance_voltron.py. The stock beamline stack (torch 2.0.1, gcc 4.8.5) cannot
# run torch.compile and silently falls back to eager at about half the frame rate.
#
# Run ON voltron (its login shell is tcsh, so invoke through bash):
#   ssh voltron "cd ~/projects/loop_sim_MINE/xtal-loop-sim ; bash setup_titan_v_env.bash"
# Builds and verifies only; the launch commands, and the note that the default
# template-serving path needs none of this, are in docs/RUNBOOK.md "Deploy on the
# TITAN V".

set -euo pipefail

VENV="${LOOPSIM_TORCH26_VENV:-$HOME/projects/loopsim-torch26}"
PT_PY="/programs/pytorch/envs/pt/bin/python3.10"
DEVTOOLSET_CC="/opt/rh/devtoolset-7/root/usr/bin/gcc"
DEVTOOLSET_CXX="/opt/rh/devtoolset-7/root/usr/bin/g++"
FORCE=0
SKIP_VERIFY=0

for arg in "$@"; do
    case "$arg" in
        --force)        FORCE=1 ;;
        --skip-verify)  SKIP_VERIFY=1 ;;
        *) echo "unknown argument: $arg (known: --force, --skip-verify)" >&2
           exit 1 ;;
    esac
done

if [[ ! -x "$PT_PY" ]]; then
    echo "ERROR: $PT_PY not found — this script must run on a host with the" >&2
    echo "beamline pytorch env (RUNBOOK.md 'Environment')." >&2
    exit 1
fi

if [[ ! -x "$DEVTOOLSET_CC" || ! -x "$DEVTOOLSET_CXX" ]]; then
    echo "ERROR: devtoolset-7 not found at /opt/rh/devtoolset-7 — torch.compile" >&2
    echo "needs it (system gcc 4.8.5 is too old for Inductor's codegen)." >&2
    exit 1
fi

if [[ -d "$VENV" && "$FORCE" -eq 0 ]]; then
    echo "[setup] $VENV already exists — skipping build (--force to rebuild)."
else
    if [[ -d "$VENV" ]]; then
        echo "[setup] --force: removing existing $VENV"
        rm -rf "$VENV"
    fi
    echo "[setup] building venv at $VENV"
    "$PT_PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    # cu118, not the dev box's cu124 — this is voltron's driver.
    "$VENV/bin/python" -m pip install torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu118
    # pillow 12 has no glibc-2.17 wheel (RHEL7) and won't build on gcc 4.8.5
    # (the system compiler) — pin 10.4.0.
    "$VENV/bin/python" -m pip install numpy scipy "pillow==10.4.0" pyyaml
    echo "[setup] venv built."
fi

export CC="$DEVTOOLSET_CC"
export CXX="$DEVTOOLSET_CXX"

if [[ "$SKIP_VERIFY" -eq 1 ]]; then
    echo "[setup] --skip-verify: not running acceptance_voltron.py."
else
    echo "[setup] verifying the full stack with acceptance_voltron.py ..."
    echo "[setup] (auto-picks a free GPU; expect ~1-2 min compile warmup)"
    "$VENV/bin/python" "$(dirname "$0")/acceptance_voltron.py"
fi

cat <<EOF

[setup] done. To launch the LIVE-RENDER path (--templates off, or a scene
with no library yet) with this env, pin a free GPU by hand first:

    nvidia-smi                                    # find a free card
    export CUDA_VISIBLE_DEVICES=<N>
    export CC="$DEVTOOLSET_CC"
    export CXX="$DEVTOOLSET_CXX"
    $VENV/bin/python -m loop_sim.server.camera_server \\
        --scene scene_files/hampton_300um.yaml --port 8080 --templates off

The default TEMPLATE-serving path needs none of this venv — launch it with
the stock interpreter instead (RUNBOOK.md "Deploy on the TITAN V"):

    /programs/pytorch/envs/pt/bin/python -m loop_sim.server.camera_server \\
        --scene scene_files/hampton_300um_realistic.yaml --port 8080
EOF
