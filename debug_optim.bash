#!/bin/bash
PT=/programs/pytorch/envs/pt/bin/python
cd /home/jamesh/projects/loop_sim/claude

echo "=== CPU render n_cond=7 (reference) ===" >&2
time $PT render.py scene.yaml --device cpu --n-cond 7 --output scene_ref.jpg

echo "=== GPU render n_cond=7 ===" >&2
time $PT render.py scene.yaml --device cuda --n-cond 7 --output scene_optim.jpg

echo "=== Compare ===" >&2
$PT - <<'EOF'
from PIL import Image
import numpy as np
ref = np.array(Image.open("scene_ref.jpg")).astype(float)
new = np.array(Image.open("scene_optim.jpg")).astype(float)
diff = np.abs(ref - new)
print(f"CPU vs GPU (n_cond=7): max={diff.max():.1f}  mean={diff.mean():.3f}  pixels_off_by_>10: {(diff.max(axis=2)>10).sum()}")
EOF
