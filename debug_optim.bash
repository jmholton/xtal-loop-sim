#!/bin/bash
PT=/programs/pytorch/envs/pt/bin/python
cd /home/jamesh/projects/loop_sim/claude

echo "=== CPU render (reference) ===" >&2
time $PT render.py scene.yaml --device cpu --output scene_ref.jpg

echo "=== Optimized GPU render ===" >&2
$PT render.py scene.yaml --device cuda --output scene_optim.jpg

echo "=== Compare ===" >&2
$PT - <<'EOF'
from PIL import Image
import numpy as np
ref = np.array(Image.open("scene_ref.jpg")).astype(float)
new = np.array(Image.open("scene_optim.jpg")).astype(float)
diff = np.abs(ref - new)
print(f"CPU vs GPU-optim: max={diff.max():.1f}  mean={diff.mean():.3f}  pixels_off_by_>10: {(diff.max(axis=2)>10).sum()}")
EOF
