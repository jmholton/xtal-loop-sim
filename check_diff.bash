#!/bin/bash
PT=/programs/pytorch/envs/pt/bin/python
cd /home/jamesh/projects/loop_sim/claude

$PT - <<'EOF'
from PIL import Image
import numpy as np

ref = np.array(Image.open("scene_ref.jpg")).astype(float)
gpu_orig = np.array(Image.open("scene_gpu.jpg")).astype(float)
gpu_new  = np.array(Image.open("scene_optim.jpg")).astype(float)

d1 = np.abs(ref - gpu_orig)
d2 = np.abs(ref - gpu_new)
d3 = np.abs(gpu_orig - gpu_new)

print(f"CPU vs GPU-original:  max={d1.max():.0f}  mean={d1.mean():.3f}  pixels>10: {(d1.max(2)>10).sum()}")
print(f"CPU vs GPU-new:       max={d2.max():.0f}  mean={d2.mean():.3f}  pixels>10: {(d2.max(2)>10).sum()}")
print(f"GPU-original vs new:  max={d3.max():.0f}  mean={d3.mean():.3f}  pixels>10: {(d3.max(2)>10).sum()}")
EOF
