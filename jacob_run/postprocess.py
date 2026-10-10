"""Blur + matched camera for one-shot renders, and the 2x2 panel (0 deg | photo / 80 | 90).

Run with Jacob's venv from his repo:
  cd /home/jadoughty/projects/loop_sim_MINE/xtal-loop-sim && env PYTHONPATH=. .venv/bin/python \
      /home/jamesh/projects/loop_sim/jacob_run/postprocess.py TAG
reads  probe/TAG_raw_rotx{0,80,90}.png
writes probe/TAG_rotx{0,80,90}.png and probe/TAG_panel.png
"""
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from loop_sim.renderer.field import to_sensor, apply_camera

tag = sys.argv[1]
P = '/home/jamesh/projects/loop_sim/jacob_run/probe/'
E, F, amp = np.load('/home/jamesh/projects/loop_sim/claude/jacob_run/camera_fit.npy')
q = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
ims = {}
for a in ('0', '80', '90'):
    r = np.asarray(Image.open(P + f'{tag}_raw_rotx{a}.png').convert('RGB'), float) / 255
    ims[a] = q(apply_camera(to_sensor(gaussian_filter(r, (2.5, 2.5, 0))), level=E, floor=F,
                            amplitude=amp, mono=False, streak=False))
    Image.fromarray(ims[a]).save(P + f'{tag}_rotx{a}.png')
photo = np.asarray(Image.open('/home/jamesh/projects/loop_sim/claude/real_loop_himag.jpg').convert('RGB'))
g = np.full((480, 8, 3), 255, np.uint8)
top = np.concatenate([ims['0'], g, photo], 1)
bot = np.concatenate([ims['80'], g, ims['90']], 1)
Image.fromarray(np.concatenate([top, np.full((8, top.shape[1], 3), 255, np.uint8), bot], 0)
                ).save(P + f'{tag}_panel.png')
print(f'saved {tag}_panel.png')
