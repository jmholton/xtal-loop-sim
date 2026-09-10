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
REALISTIC = os.path.join(REPO_ROOT, "scene_files", "hampton_300um_realistic.yaml")


def _u8(img):
    return (np.asarray(img) * 255).clip(0, 255).astype(int)


def _render_pair(scene_path, dev, pose, res=None, n_cond=1, psf=False):
    sc = load(scene_path, device="cpu")
    if res:
        sc.camera_cfg = dict(sc.camera_cfg, width=res[0], height=res[1])
    npi, _ = np_render(sc, Goniometer(sc.geometry).set(**pose), n_cond=n_cond,
                       psf=psf)
    ts = TorchScene(sc, dev, torch.float64)
    ti = render_torch(ts, Goniometer(sc.geometry).set(**pose), n_cond=n_cond,
                      psf=psf).cpu().numpy()
    return _u8(npi), _u8(ti)


CPU = torch.device("cpu")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# The geometric trace: byte-identical, exactly, as it always was.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pose", [{}, {"rotx": 45}, {"roty": 30, "tx": 0.05}])
def test_render_parity_cpu_f64_hampton(pose):
    a, b = _render_pair(HAMPTON, CPU, pose, res=(96, 72))
    assert int(np.abs(a - b).max()) == 0


def test_render_parity_cpu_f64_mitegen_thinshell():
    # exercises the ThinShell -> TSurfaceMesh dispatch + CSG
    a, b = _render_pair(MITEGEN, CPU, {}, res=(96, 72))
    assert int(np.abs(a - b).max()) == 0


# The mesh path at full scale: `hampton_300um_realistic` (5472 faces, a
# real SurfaceMesh, the scene whose library build dominates every timing
# in the docs) must be pinned here, not just mitegen's 234-face ThinShell.
# rotx=45 puts the droplet, the loop fiber and the pin CSG all in frame at
# once.
@pytest.mark.parametrize("pose", [{}, {"rotx": 45}])
def test_render_parity_cpu_f64_realistic_droplet_mesh(pose):
    a, b = _render_pair(REALISTIC, CPU, pose, res=(96, 72))
    assert int(np.abs(a - b).max()) == 0


@cuda_only
@pytest.mark.parametrize("pose", [{"rotx": 45}])
def test_render_parity_cuda_realistic_droplet_mesh(pose):
    """The CUDA leg of the same guard.

    `TSurfaceMesh` is a different implementation from the numpy `SurfaceMesh`
    (brute-force vs culled), so CPU parity alone does not prove the CUDA mesh
    path.  Reduced resolution keeps this affordable -- full res on this scene is
    ~30 s -- while still running the real kernel over the real 5472 faces.
    """
    a, b = _render_pair(REALISTIC, torch.device("cuda"), pose, res=(320, 240))
    assert int(np.abs(a - b).max()) == 0


@cuda_only
@pytest.mark.parametrize("pose", [{}, {"rotx": 45}, {"zoom": 2.0}, {"rotx": 33, "tx": 0.07}])
def test_render_parity_cuda_fullres_hampton(pose):
    a, b = _render_pair(HAMPTON, torch.device("cuda"), pose)  # full scene resolution
    assert int(np.abs(a - b).max()) == 0


# ---------------------------------------------------------------------------
# With the PSF: agreement is +/-1 grey level, and that is not a regression.
# The two f64 traces differ by ~3e-8 on ~0.7% of values; the PSF makes that
# visible at the quantisation boundary; exceeding 1 means something
# structural broke.  Keep both families of test: the exact one above still
# guards the trace.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pose", [{}, {"rotx": 45}, {"roty": 30, "tx": 0.05}])
def test_render_parity_with_psf_cpu_within_one_level(pose):
    a, b = _render_pair(HAMPTON, CPU, pose, res=(96, 72), psf=True)
    assert int(np.abs(a - b).max()) <= 1


@cuda_only
@pytest.mark.parametrize("pose", [{}, {"rotx": 37}, {"zoom": 2.0}])
def test_render_parity_with_psf_cuda_fullres_within_one_level(pose):
    a, b = _render_pair(HAMPTON, torch.device("cuda"), pose, psf=True)
    assert int(np.abs(a - b).max()) <= 1


def test_psf_actually_changes_the_image():
    """Guard against the PSF silently becoming a no-op.

    mitegen's 1.0 um camera pixel puts sigma at 1.155 px, so the softening is
    unambiguous there even at camera resolution.
    """
    sharp, _ = _render_pair(MITEGEN, CPU, {}, res=(96, 72), psf=False)
    soft, _ = _render_pair(MITEGEN, CPU, {}, res=(96, 72), psf=True)
    assert int(np.abs(sharp - soft).max()) > 8, "PSF had no visible effect"
