"""
Rung-2/3 render parity: render_torch == numpy render (whole pipeline:
next_interface + trace_rays + condenser accumulation + AABB cull + compaction).

The float64 torch engine is byte-identical to the numpy reference. CPU-f64 runs
anywhere (rung 2, reduced res for speed); full-res CUDA (rung 3) is GPU-gated.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

from loop_sim.scene.scene import load
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer.microscope import render as np_render
from loop_sim.renderer.engine_torch import TorchScene, render_torch

HAMPTON = os.path.join(REPO_ROOT, "scene_files", "hampton_300um.yaml")
MITEGEN = os.path.join(REPO_ROOT, "scene_files", "mitegen_200um.yaml")


def _u8(img):
    return (np.asarray(img) * 255).clip(0, 255).astype(int)


def _render_pair(scene_path, dev, pose, res=None, n_cond=1):
    sc = load(scene_path, device="cpu")
    if res:
        sc.camera_cfg = dict(sc.camera_cfg, width=res[0], height=res[1])
    npi, _ = np_render(sc, Goniometer(sc.geometry).set(**pose), n_cond=n_cond)
    ts = TorchScene(sc, dev, torch.float64)
    ti = render_torch(ts, Goniometer(sc.geometry).set(**pose), n_cond=n_cond).cpu().numpy()
    return _u8(npi), _u8(ti)


CPU = torch.device("cpu")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.mark.parametrize("pose", [{}, {"rotx": 45}, {"roty": 30, "tx": 0.05}])
def test_render_parity_cpu_f64_hampton(pose):
    a, b = _render_pair(HAMPTON, CPU, pose, res=(96, 72))
    assert int(np.abs(a - b).max()) == 0


def test_render_parity_cpu_f64_mitegen_thinshell():
    # exercises the ThinShell -> TSurfaceMesh dispatch + CSG
    a, b = _render_pair(MITEGEN, CPU, {}, res=(96, 72))
    assert int(np.abs(a - b).max()) == 0


@cuda_only
@pytest.mark.parametrize("pose", [{}, {"rotx": 45}, {"zoom": 2.0}, {"rotx": 33, "tx": 0.07}])
def test_render_parity_cuda_fullres_hampton(pose):
    a, b = _render_pair(HAMPTON, torch.device("cuda"), pose)  # full scene resolution
    assert int(np.abs(a - b).max()) == 0
