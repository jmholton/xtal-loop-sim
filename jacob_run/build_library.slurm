#!/bin/bash
#SBATCH --job-name=jhoop_lib
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jamesh/projects/loop_sim/jacob_run/build_%j.log

# Full 360-frame library for James's hi-mag scene (634x480 @ 0.9139 um, served
# on the 704x480 sensor raster).  supersample 1 / pan 0.2 mm: the defaults
# (4 / 0.6) are sized for the 7.4 um/px mid-mag scenes and give ~100 Mpx per
# frame here (~1 h/frame).
J=/home/jadoughty/projects/loop_sim_MINE/xtal-loop-sim
W=/home/jamesh/projects/loop_sim/jacob_run
SCENE=$W/scene_files/${NAME:-james_himag}.yaml   # sbatch --export=ALL,NAME=<scene stem>
echo "=== scene $SCENE ==="
export PYTHONUNBUFFERED=1
cd $J

echo "=== Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES  start $(date) ==="
echo "=== check render at photo pose (scene motor), n_cond=7 ==="
$J/.venv/bin/python render.py $SCENE --device cuda --n-cond 7 --output $W/probe/check_${NAME:-james_himag}.png

echo "=== full library ==="
time $J/.venv/bin/python -m loop_sim.library --scene $SCENE --root $W/frame_library \
    --supersample 1 --pan-mm 0.2
echo "=== done $(date) ==="
$J/.venv/bin/python -m loop_sim.library --scene $SCENE --root $W/frame_library \
    --supersample 1 --pan-mm 0.2 --status
