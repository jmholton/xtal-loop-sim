"""
/motor-driven motion window: instant pose sets (how AXIS-style consumers such
as MxCuBE/EPICS drive the goniometer) must get fast preview frames while the
pose is changing, and one exact full-quality frame once it settles.

CPU-only: the render is faked at the microscope_render seam so the REAL
preview/settle selection logic in _render_frame runs (engine='numpy' path).
"""
import os
import sys
import threading
import time

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.scene.scene import load
from loop_sim.server import camera_server as cs

HAMPTON = os.path.join(REPO_ROOT, "scene_files", "hampton_300um.yaml")

SETTLE = 0.25   # short window so tests stay fast; margins are ~4x below


@pytest.fixture()
def server(monkeypatch):
    """numpy-engine server with a fast fake render that records n_cond."""
    calls = []

    def fake_render(scene, gono, n_cond=1, jpeg_quality=85):
        calls.append(n_cond)
        time.sleep(0.01)
        # Must return a real (H, W, 3) float image, not None: since the camera
        # emulation stage landed, the server takes microscope_render's FLOAT
        # image and encodes it itself, rather than passing through the JPEG
        # this seam produces.  The assertions below key off `calls` (the n_cond
        # each render used), never these bytes, so a tiny frame is enough.
        return np.zeros((8, 8, 3), np.float32), b"FAKE%04d" % len(calls)

    monkeypatch.setattr(cs, "microscope_render", fake_render)
    scene = load(HAMPTON, device="cpu")
    srv = cs.CameraServer(scene, host="127.0.0.1", port=0, engine="numpy",
                          settle_delay=SETTLE)
    srv._calls = calls
    srv.start(background=True)
    yield srv
    srv.server_close()


def _wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_boot_frame_is_exact(server):
    """Before any pose change, the served frame is full quality."""
    assert _wait_for(lambda: len(server._calls) >= 1)
    assert server._calls[0] == server._n_cond


def test_motor_set_previews_then_settles_exact(server):
    """An instant /motor set renders a preview, then one exact frame after
    the pose has been quiet for settle_delay."""
    assert _wait_for(lambda: len(server._calls) >= 1)   # boot frame
    n0 = len(server._calls)
    server._set_pose_instant({"rotx": 10.0})
    assert _wait_for(lambda: len(server._calls) > n0)
    assert server._calls[n0] == 1                       # preview (window open)
    # After settle_delay of quiet, the producer forces an exact re-render.
    assert _wait_for(lambda: len(server._calls) > n0 + 1, timeout=SETTLE * 8)
    assert server._calls[-1] == server._n_cond


def test_motor_stream_stays_preview_until_quiet(server):
    """A stream of /motor sets renders previews throughout; only after the
    stream stops does the exact frame appear."""
    assert _wait_for(lambda: len(server._calls) >= 1)   # boot frame
    n0 = len(server._calls)
    for k in range(5):
        server._set_pose_instant({"rotx": float(k)})
        time.sleep(SETTLE / 5)
    assert _wait_for(lambda: len(server._calls) > n0 and
                     server._calls[-1] == server._n_cond,
                     timeout=SETTLE * 10)
    stream_calls = server._calls[n0:-1]
    assert stream_calls, "expected preview renders during the /motor stream"
    assert all(n == 1 for n in stream_calls)            # all previews
    assert server._calls[-1] == server._n_cond          # exact settle frame
