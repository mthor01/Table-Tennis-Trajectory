#!/usr/bin/env sh
# Runs all pipeline stages in order; results go to output/. Usage:
#   ./run_pipeline.sh           full run (all trajectories and segments, takes a while)
#   ./run_pipeline.sh --quick   small run to check that everything works (about a minute)
# Any further arguments are passed on to the reconstruction stage, e.g. --pixel-noise 2
set -e
cd "$(dirname "$0")"

RECONSTRUCTION_ARGS=""
if [ "$1" = "--quick" ]; then
    RECONSTRUCTION_ARGS="--trajectories 6 --max-segments 5 --generations 150 --population 60"
    shift
fi
PYTHON="${PYTHON:-python}"

echo "== 1/5 detecting camera cuts"
"$PYTHON" -m pipeline.detect_cuts
echo "== 2/5 estimating camera poses"
"$PYTHON" -m pipeline.estimate_poses --debug
echo "== 3/5 simulating ball trajectories"
"$PYTHON" -m pipeline.simulate_trajectories
echo "== 4/5 reconstructing 3D trajectories"
"$PYTHON" -m pipeline.reconstruct $RECONSTRUCTION_ARGS "$@"
echo "== 5/5 evaluating"
"$PYTHON" -m pipeline.evaluate
"$PYTHON" -m pipeline.visualize
