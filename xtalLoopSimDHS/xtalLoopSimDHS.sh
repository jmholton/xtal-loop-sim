#!/bin/bash
# Starts the xtalLoopSim DHS through its own venv, the way every other DHS here
# is started by hand:
#
#     ./xtalLoopSimDHS.sh real|pretend [LOCAL|SIM831] [DHS options]
#
# `real` drives the loop-sim camera server over HTTP; `pretend` models the pose
# internally with the same timing and contacts nothing. The beamline word picks
# config/<name>.config and defaults to LOCAL. Anything after it goes to the DHS
# unchanged. Runs from any directory. Ends in `exec`, so the python process
# replaces this shell rather than running under it; match `xtal_loop_sim_DHS.py`,
# not this launcher, to find or kill it.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1

MODE="${1:-}"
shift || true
case "$MODE" in
    real|pretend) ;;
    *)  echo "usage: $0 real|pretend [LOCAL|SIM831] [DHS options]" >&2
        echo "  real: moves go to the camera server; pretend: the pose is modelled here" >&2
        exit 2 ;;
esac

BEAMLINE=LOCAL
case "${1:-}" in
    -*|'') ;;
    *) BEAMLINE="$1"; shift ;;
esac

if [ ! -x .venv/bin/python ]; then
    echo "$0: no .venv in $PWD; build it per README.md 'Create the env'" >&2
    exit 3
fi
if [ ! -r "config/$BEAMLINE.config" ]; then
    echo "$0: $PWD/config/$BEAMLINE.config is missing" >&2
    exit 4
fi

exec .venv/bin/python -u xtal_loop_sim_DHS.py "$MODE" "$BEAMLINE" -v "$@"
