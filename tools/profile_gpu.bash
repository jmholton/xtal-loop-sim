#!/bin/bash
cd "$(dirname "$0")/.."
.venv/bin/python tools/profile_gpu.py
