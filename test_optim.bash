#!/bin/bash
PT=/programs/pytorch/envs/pt/bin/python
cd /home/jamesh/projects/loop_sim/claude

echo "=== GPU render (optimized) ===" >&2
time $PT render.py scene.yaml --device cuda --output scene_optim.jpg

echo "=== Compare with previous GPU output ===" >&2
if [ -f scene_gpu.jpg ]; then
    $PT - <<'EOF'
from PIL import Image
import numpy as np
a = np.array(Image.open("scene_gpu.jpg")).astype(float)
b = np.array(Image.open("scene_optim.jpg")).astype(float)
diff = np.abs(a - b)
print(f"Max pixel diff: {diff.max():.1f}  Mean: {diff.mean():.3f}")
EOF
else
    echo "No previous scene_gpu.jpg to compare against"
fi
