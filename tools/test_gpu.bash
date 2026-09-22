#!/bin/bash
PT=.venv/bin/python
cd "$(dirname "$0")/.."

echo "=== CPU render ===" >&2
time $PT render.py data/scene_files/hampton_300um.yaml --device cpu  --output scene_cpu.jpg

echo "=== GPU render ===" >&2
time $PT render.py data/scene_files/hampton_300um.yaml --device cuda --output scene_gpu.jpg
