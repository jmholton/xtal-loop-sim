#!/bin/bash
PT=/programs/pytorch/envs/pt/bin/python
cd /home/jamesh/projects/loop_sim/claude

echo "=== CPU render ===" >&2
time $PT render.py scene.yaml --device cpu  --output scene_cpu.jpg

echo "=== GPU render ===" >&2
time $PT render.py scene.yaml --device cuda --output scene_gpu.jpg
