#!/usr/bin/env python3
"""Profile the GPU renderer."""
import sys, os, cProfile, pstats, io, time
sys.path.insert(0, os.path.dirname(__file__))

from loop_sim.scene.scene       import load
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer.microscope import render

scene  = load("scene.yaml", device='cuda')
gonio  = Goniometer(scene.geometry)

# Warm-up
t0 = time.perf_counter()
img, jpg = render(scene, gonio, n_cond=1)
t1 = time.perf_counter()
print(f"Warm-up: {t1-t0:.2f}s", file=sys.stderr)

# Profile second render (GPU caches warmed)
pr = cProfile.Profile()
pr.enable()
img, jpg = render(scene, gonio, n_cond=1)
pr.disable()
t2 = time.perf_counter()
print(f"Profiled: {t2-t1:.2f}s", file=sys.stderr)

s  = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(25)
print(s.getvalue())
