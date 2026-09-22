"""Tests for GET /xray, the live X-ray radiograph (CameraServer._render_xray_png).

The claims here, and the test that guards each:

  * /xray renders live and answers an 8-bit greyscale PNG of the camera's
    size -> test_live_render_serves_a_greyscale_png
  * the PNG is exactly `beam.transmission_png` of the numpy reference at the
    live pose, the same encoding `render.py --xray` writes
    -> test_served_png_matches_the_shared_encoding
  * a repeat at the same pose is served from the single-slot memo
    -> test_served_bytes_are_memoized_on_repeat

Every server here is --templates off, numpy engine, at a shrunk resolution:
render_xray_numpy's per-ray Python loop dominates at any scene complexity.
"""
import io
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.renderer.beam import render_xray_numpy, transmission_png  # noqa: E402
from loop_sim.scene.scene import load                                   # noqa: E402
from loop_sim.server.camera_server import CameraServer                  # noqa: E402

SCENE = os.path.join(REPO_ROOT, "data", "scene_files", "hampton_300um.yaml")
RES = (48, 36)


def _server():
    scene = load(SCENE, device="cpu")
    scene.camera_cfg["width"], scene.camera_cfg["height"] = RES
    return CameraServer(scene, host="127.0.0.1", port=0, engine="numpy",
                        scene_path=SCENE, templates=False, prewarm=False)


def test_live_render_serves_a_greyscale_png():
    from PIL import Image
    srv = _server()
    try:
        img = Image.open(io.BytesIO(srv._render_xray_png()))
        assert img.mode == "L"
        assert img.size == RES
    finally:
        srv.server_close()


def test_served_png_matches_the_shared_encoding():
    srv = _server()
    try:
        served = srv._render_xray_png()
        expected = transmission_png(render_xray_numpy(srv._scene,
                                                      srv._snapshot_gonio()))
        assert served == expected
    finally:
        srv.server_close()


def test_served_bytes_are_memoized_on_repeat():
    srv = _server()
    try:
        first = srv._render_xray_png()
        assert srv._xray_cache is not None
        key, cached = srv._xray_cache
        assert cached == first
        assert srv._render_xray_png() is first
    finally:
        srv.server_close()


def test_transmission_png_maps_clipped_transmission_to_grey():
    from PIL import Image
    T = np.array([[-0.5, 0.0, 0.5, 1.0, 2.0]])
    arr = np.array(Image.open(io.BytesIO(transmission_png(T))))
    assert arr.dtype == np.uint8
    assert arr.tolist() == [[0, 0, 127, 255, 255]]
