#!/bin/bash
#SBATCH --job-name=jhoop_1shot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jamesh/projects/loop_sim/jacob_run/probe/oneshot_%j.log

# One-shot renders of a jacob_run scene at 0, 80 and 90 deg with Jacob's renderer.
#   sbatch --export=ALL,NAME=james_himag_rod60 oneshot.slurm
# then blur + camera match + panel:
#   cd /home/jadoughty/projects/loop_sim_MINE/xtal-loop-sim && env PYTHONPATH=. \
#       .venv/bin/python /home/jamesh/projects/loop_sim/jacob_run/postprocess.py $NAME
# 80/90 use ty=0: the scene's ty puts the photo pose's vertical offset into depth.
W=/home/jamesh/projects/loop_sim/jacob_run
SCENE=$W/scene_files/$NAME.yaml
cd /home/jadoughty/projects/loop_sim_MINE/xtal-loop-sim
.venv/bin/python render.py $SCENE --device cuda --n-cond 7 --rotx 0 --output $W/probe/${NAME#james_himag_}_raw_rotx0.png
.venv/bin/python render.py $SCENE --device cuda --n-cond 7 --rotx 80 --ty 0 --output $W/probe/${NAME#james_himag_}_raw_rotx80.png
.venv/bin/python render.py $SCENE --device cuda --n-cond 7 --rotx 90 --ty 0 --output $W/probe/${NAME#james_himag_}_raw_rotx90.png
