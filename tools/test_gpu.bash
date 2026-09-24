#!/bin/bash
# Renders hampton_300um on CPU then GPU, timed: the base A/B smoke test.
# Writes scene_cpu.jpg and scene_gpu.jpg at the repo root.
PT=.venv/bin/python
cd "$(dirname "$0")/.."

echo "=== CPU render ===" >&2
time $PT render.py data/scene_files/hampton_300um.yaml --device cpu  --output scene_cpu.jpg

echo "=== GPU render ===" >&2
time $PT render.py data/scene_files/hampton_300um.yaml --device cuda --output scene_gpu.jpg
