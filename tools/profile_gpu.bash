#!/bin/bash
# cProfile the GPU render path on hampton_300um, with a warm-up render discarded first.
cd "$(dirname "$0")/.."
.venv/bin/python tools/profile_gpu.py
