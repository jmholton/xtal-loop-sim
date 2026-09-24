#!/bin/bash
# cProfile the CPU (numpy reference) renderer on hampton_300um.
cd "$(dirname "$0")/.."
.venv/bin/python tools/profile_render.py
