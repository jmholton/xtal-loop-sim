#!/bin/bash
PT=/programs/pytorch/envs/pt/bin/python

# Install missing deps if needed
$PT -c "import yaml" 2>/dev/null || $PT -m pip install pyyaml --quiet
$PT -c "import PIL"  2>/dev/null || $PT -m pip install pillow --quiet
$PT -c "import scipy" 2>/dev/null || $PT -m pip install scipy --quiet

$PT profile_render.py
