#!/usr/bin/env python3
"""Profile the renderer to find bottlenecks."""
import sys, os, cProfile, pstats, io, time
sys.path.insert(0, os.path.dirname(__file__))

from loop_sim.scene.scene       import load
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer.microscope import render

scene  = load("scene.yaml")
gonio  = Goniometer(scene.geometry)

# Warm-up (scene is already loaded, just time render)
t0 = time.perf_counter()
img, jpg = render(scene, gonio, n_cond=1)
t1 = time.perf_counter()
print(f"Cold render: {t1-t0:.2f}s  image={img.shape}", file=sys.stderr)

# Profile
pr = cProfile.Profile()
pr.enable()
img, jpg = render(scene, gonio, n_cond=1)
pr.disable()

t2 = time.perf_counter()
print(f"Profiled render: {t2-t1:.2f}s", file=sys.stderr)

s  = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())
