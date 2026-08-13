"""
Converge-on-idle gate: what the camera server SERVES at idle must be byte-equal
to the exact f64 reference render, encoded the same way.

This is the guard for all preview-mode work: approximations may run while a
move animates, but the settled (idle) frame the server publishes must remain
the exact engine output. Runs the real server render path (_render_now) — no
HTTP, no threads.

NOT WEAKENED BY CAMERA EMULATION (2026-08-10).  The server now maps
transmittance through the camera model in `loop_sim/renderer/field.py` before
encoding, so "the exact engine output" is no longer the same bytes as raw
transmittance.  The reference below therefore routes through the SAME shared
`encode_frame` the server uses, and the assertion still means exactly what it
did: the served frame is the exact f64 engine output carried through the
documented, deterministic delivery chain, with nothing approximated and
nothing stochastic in it.  If a future delivery stage is added and this test
is not updated with it, the test fails — which is the point.  Keep the
reference routed through the server's own helper rather than reimplementing
the chain here; a second implementation is what this guard exists to catch.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

from loop_sim.scene.scene import load
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer.engine_torch import TorchScene, render_torch
from loop_sim.server.camera_server import CameraServer, encode_frame, pose_phase

HAMPTON = os.path.join(REPO_ROOT, "scene_files", "hampton_300um.yaml")

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="CUDA not available")


def _reference_jpeg(server, pose, n_cond, quality=85):
    """The exact f64 engine output, delivered exactly as the server delivers it.

    Everything downstream of the trace is taken FROM THE SERVER -- `_camera`,
    `_sensor` and `_live_pin` -- so the reference and the server share one
    delivery implementation rather than two that agree today.  `_live_pin`
    joined that list on 2026-08-12, when the specular glint stopped inferring
    the pin from the silhouette and started projecting it from the scene: it is
    a delivery stage that depends on the pose, so a reference that skipped it
    would be a second implementation, which is what this guard exists to catch.

    The grain phase is taken from the GONIOMETER, not from the `pose` dict, for
    the same reason: the server reads `gono.get()`, which resolves every motor
    including `zoom` to 1.0, while a partial dict would default it to 0.0 and
    silently re-roll the grain. Same source, same bytes.
    """
    scene = server._scene
    ts = TorchScene(scene, torch.device("cuda"), torch.float64)
    gono = Goniometer(scene.geometry).set(**pose)
    img = render_torch(ts, gono, n_cond=n_cond)
    return encode_frame(img.detach().cpu().numpy(), quality, server._camera,
                        server._sensor, pose_phase(gono.get()), 0.0,
                        server._live_pin(gono))


@pytest.fixture()
def server():
    scene = load(HAMPTON, device="cpu")
    srv = CameraServer(scene, host="127.0.0.1", port=0, engine="auto")
    yield srv
    srv.server_close()


@cuda_only
@pytest.mark.parametrize("pose", [{}, {"rotx": 45}, {"zoom": 2.0}])
def test_idle_served_frame_is_exact(server, pose):
    """Idle (settled) serve path == exact f64 reference, byte-for-byte."""
    server._goniometer.set(**pose)
    server._anim_active = False
    served = server._render_now()
    assert served == _reference_jpeg(server, pose, n_cond=server._n_cond)


@cuda_only
def test_animating_preview_uses_n_cond_1(server):
    """While a move animates, the served preview is the exact n_cond=1 frame.

    (Later preview modes may relax pixel equality during motion behind the
    preview flag; until then the motion frame is exact n_cond=1.)
    """
    server._anim_active = True
    served = server._render_now()
    assert served == _reference_jpeg(server, {}, n_cond=1)


@cuda_only
def test_preview_mode_off_is_always_exact():
    """--preview-mode off: even mid-animation, the served frame is the exact
    full-n_cond render (pure-exact streaming)."""
    scene = load(HAMPTON, device="cpu")
    srv = CameraServer(scene, host="127.0.0.1", port=0, engine="auto",
                       preview_mode=False)
    try:
        srv._anim_active = True
        served = srv._render_now()
        assert served == _reference_jpeg(srv, {}, n_cond=srv._n_cond)
    finally:
        srv.server_close()
