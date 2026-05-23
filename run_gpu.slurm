#!/bin/bash
#SBATCH --job-name=loop_sim_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/jamesh/projects/loop_sim/claude/slurm_%j.log

PT=/programs/pytorch/envs/pt/bin/python
cd /home/jamesh/projects/loop_sim/claude

echo "=== Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES ===" >&2

echo "=== CPU render n_cond=1 (fast reference) ===" >&2
time $PT render.py scene.yaml --device cpu --n-cond 1 --output scene_ref1.png

echo "=== GPU render n_cond=1 ===" >&2
time $PT render.py scene.yaml --device cuda --n-cond 1 --output scene_optim1.png

echo "=== CPU render n_cond=7 (reference) ===" >&2
time $PT render.py scene.yaml --device cpu --n-cond 7 --output scene_ref.png

echo "=== GPU render n_cond=7 ===" >&2
time $PT render.py scene.yaml --device cuda --n-cond 7 --output scene_optim.png

echo "=== Compare (PNG, lossless) ===" >&2
$PT - <<'PYEOF'
from PIL import Image
import numpy as np

def analyze(ref_path, new_path, label):
    ref = np.array(Image.open(ref_path)).astype(float)
    new = np.array(Image.open(new_path)).astype(float)
    diff = np.abs(ref - new)
    mx = diff.max(axis=2)   # (H, W) max over channels
    n10 = (mx > 10).sum()
    n30 = (mx > 30).sum()
    n60 = (mx > 60).sum()
    # row-wise: where are the big diffs?
    row_bad = (mx > 10).sum(axis=1)
    top_rows = row_bad.argsort()[-5:][::-1]
    print(f"{label}: max={diff.max():.0f}  mean={diff.mean():.2f}  "
          f">10:{n10}  >30:{n30}  >60:{n60}")
    print(f"  top diff rows (px>10 count): {list(zip(top_rows.tolist(), row_bad[top_rows].tolist()))}")

analyze("scene_ref1.png", "scene_optim1.png", "CPU vs GPU n_cond=1")
analyze("scene_ref.png",  "scene_optim.png",  "CPU vs GPU n_cond=7")
PYEOF
