"""
Interactive-controls tests: the pure move/geometry helpers in camera_server,
plus a lightweight HTTP smoke test of the new endpoints.

The animation + recenter math is factored into pure functions so it can be
asserted exactly without a live render or GPU:

  * resolve_target   — absolute / relative / screen-fraction-pan resolution
  * move_duration    — FOV-crosses-in-2s, 360deg/s, speed-dial scaling
  * recenter_target  — click-to-centre (exact 3-D, Δtrans = -R^T p_lab)

Run:  pytest tests/test_server_controls.py -v
"""
import os
import sys
import threading
import urllib.request

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.scene.materials import AIR
from loop_sim.scene.scene import Scene
from loop_sim.server.camera_server import (
    resolve_target, move_duration, recenter_target, CameraServer,
)

GEOM = {
    "camera_fast": [1, 0, 0],
    "camera_slow": [0, 1, 0],
    "beam_axis":   [0, 0, 1],
    "rotx_axis":   [1, 0, 0],
    "roty_axis":   [0, 1, 0],
    "rotz_axis":   [0, 0, 1],
}
CAM = {"width": 640, "height": 480, "pixel_size": 0.0074}   # hampton-like
W, H, PX = CAM["width"], CAM["height"], CAM["pixel_size"]
FOV_W = W * PX     # 4.736 mm
REST = {"tx": 0.0, "ty": 0.0, "tz": 0.0,
        "rotx": 0.0, "roty": 0.0, "rotz": 0.0, "zoom": 1.0}


# ---------------------------------------------------------------------------
# resolve_target
# ---------------------------------------------------------------------------

def test_resolve_absolute_and_relative():
    cur = dict(REST, rotx=5.0)
    assert resolve_target(cur, {"rotx": "45"}, W, H, PX)["rotx"] == pytest.approx(45.0)
    assert resolve_target(cur, {"drotx": "10"}, W, H, PX)["rotx"] == pytest.approx(15.0)
    assert resolve_target(cur, {"dtx": "0.5"}, W, H, PX)["tx"] == pytest.approx(0.5)


def test_resolve_screen_pan():
    t = resolve_target(REST, {"panx": "0.25", "pany": "0.25"}, W, H, PX)
    assert t["tx"] == pytest.approx(0.25 * W * PX)     # quarter of FOV width
    assert t["ty"] == pytest.approx(0.25 * H * PX)     # quarter of FOV height


def test_resolve_pan_is_image_relative_under_rotation():
    """The pad moves the sample the way it looks on screen at any spindle
    angle.  The XYZ stage rides on the spindle, so a screen-vertical pan is
    pure ty at phi=0, pure tz at phi=90, and REVERSED ty at phi=180.  The
    bug this guards applied the same ty at every angle: at phi=90 the pad
    only defocused, and at phi=180 it moved the image the wrong way.
    """
    want = 0.25 * H * PX
    for phi, (ty, tz) in ((0.0, (want, 0.0)),
                          (90.0, (0.0, -want)),
                          (180.0, (-want, 0.0)),
                          (270.0, (0.0, want))):
        t = resolve_target(dict(REST, rotx=phi), {"pany": "0.25"},
                           W, H, PX, geometry=GEOM)
        assert t["ty"] == pytest.approx(ty, abs=1e-9), f"ty at phi={phi}"
        assert t["tz"] == pytest.approx(tz, abs=1e-9), f"tz at phi={phi}"
        assert t["tx"] == pytest.approx(0.0, abs=1e-9), f"tx at phi={phi}"


def test_resolve_pan_matches_recenter_convention():
    """resolve_target and recenter_target must agree about which way the
    image moves: panning by +0.25 screen then clicking the point that lands
    in the centre must return the pose to where it started.
    """
    scene_geom, phi = GEOM, 37.0
    start = dict(REST, rotx=phi)
    panned = resolve_target(start, {"panx": "0.25", "pany": "0.25"},
                            W, H, PX, geometry=scene_geom)
    # the feature that was at the centre is now offset by (+0.25W, +0.25H) px
    back = recenter_target(W / 2.0 + 0.25 * W, H / 2.0 + 0.25 * H,
                           panned, scene_geom, CAM)
    for k in ("tx", "ty", "tz"):
        assert back[k] == pytest.approx(start[k], abs=1e-9), k


def test_resolve_zoom_clamped_positive():
    t = resolve_target(REST, {"dzoom": "-99"}, W, H, PX)
    assert t["zoom"] >= 1e-3


def test_resolve_pan_scales_with_zoom():
    # at zoom=2 the field of view halves → a 0.25 pan moves half as far
    z2 = dict(REST, zoom=2.0)
    t = resolve_target(z2, {"panx": "0.25"}, W, H, PX)
    assert t["tx"] == pytest.approx(0.25 * W * (PX / 2.0))


# ---------------------------------------------------------------------------
# move_duration
# ---------------------------------------------------------------------------

def test_translation_crosses_screen_in_two_seconds():
    target = dict(REST, tx=FOV_W)        # pan exactly one screen width
    assert move_duration(REST, target, 1.0, W, PX) == pytest.approx(2.0, rel=1e-9)


def test_rotation_is_360_deg_per_second():
    target = dict(REST, rotx=360.0)      # one full turn = 60 rpm
    assert move_duration(REST, target, 1.0, W, PX) == pytest.approx(1.0, rel=1e-9)


def test_speed_dial_scales_duration_inversely():
    target = dict(REST, tx=FOV_W)
    assert move_duration(REST, target, 2.0, W, PX) == pytest.approx(1.0, rel=1e-9)   # faster
    assert move_duration(REST, target, 0.5, W, PX) == pytest.approx(4.0, rel=1e-9)   # slow-mo


def test_duration_is_slowest_parameter():
    target = dict(REST, tx=FOV_W, rotx=360.0)   # 2.0 s vs 1.0 s → 2.0 s
    assert move_duration(REST, target, 1.0, W, PX) == pytest.approx(2.0, rel=1e-9)


# ---------------------------------------------------------------------------
# recenter_target
# ---------------------------------------------------------------------------

def test_short_jog_gets_a_duration_floor():
    """A 15 deg phi click is 42 ms at 360 deg/s -- about one frame, so it
    lands as a jump and a burst of clicks reads as N separate jumps.  The
    floor makes it a glide the next click can preempt and extend.
    """
    jog = dict(REST, rotx=15.0)
    assert move_duration(REST, jog, 1.0, W, PX) == pytest.approx(0.25)
    # the floor is a floor, not a fixed cost: long moves keep their own timing
    assert move_duration(REST, dict(REST, rotx=360.0), 1.0, W, PX) == pytest.approx(1.0)
    # and the speed dial still scales it
    assert move_duration(REST, jog, 2.0, W, PX) == pytest.approx(0.125)


def test_zero_distance_move_stays_zero():
    """The floor must not make a no-op move animate for a quarter second."""
    assert move_duration(REST, dict(REST), 1.0, W, PX) == 0.0


def test_recenter_centre_click_is_noop():
    t = recenter_target(W / 2.0, H / 2.0, REST, GEOM, CAM)
    assert t["tx"] == pytest.approx(0.0, abs=1e-9)
    assert t["ty"] == pytest.approx(0.0, abs=1e-9)
    assert t["tz"] == pytest.approx(0.0, abs=1e-9)


def test_recenter_offcentre_click_zero_rotation():
    # click on the right edge → that point must move left to the centre (tx < 0)
    t = recenter_target(W, H / 2.0, REST, GEOM, CAM)
    assert t["tx"] == pytest.approx(-(W - W / 2.0) * PX)   # -(320*0.0074)
    assert t["ty"] == pytest.approx(0.0, abs=1e-9)
    assert t["tz"] == pytest.approx(0.0, abs=1e-9)
    # vertical offset maps to ty
    t2 = recenter_target(W / 2.0, H, REST, GEOM, CAM)
    assert t2["ty"] == pytest.approx(-(H - H / 2.0) * PX)
    assert t2["tx"] == pytest.approx(0.0, abs=1e-9)


def test_recenter_horizontal_exact_under_rotx():
    # rotx spins about the fast (horizontal) axis, so a horizontal click recenters
    # tx exactly the same as with no rotation, and leaves tz unchanged.
    state = dict(REST, rotx=90.0)
    t = recenter_target(W, H / 2.0, state, GEOM, CAM)
    assert t["tx"] == pytest.approx(-(W - W / 2.0) * PX)
    assert t["tz"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# HTTP smoke test (no rendering — only the lightweight control endpoints)
# ---------------------------------------------------------------------------

def _GET(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def test_control_endpoints_smoke():
    scene = Scene([], GEOM, CAM, {}, background=AIR)
    srv = CameraServer(scene, host="127.0.0.1", port=0, engine="numpy")
    port = srv.server_address[1]
    # serve_forever directly (skip start(): no bg render / animator needed here)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        st, ct, body = _GET(port, "/")
        assert st == 200 and "text/html" in ct and b"loop-sim" in body

        st, ct, body = _GET(port, "/move?drotx=90&speed=2")
        assert st == 200 and "application/json" in ct and b"rotx" in body

        st, ct, body = _GET(port, "/recenter?px=640&py=240")
        assert st == 200 and "application/json" in ct

        # fraction form (what the browser sends): centre click → no-op target
        st, ct, body = _GET(port, "/recenter?fx=0.5&fy=0.5")
        assert st == 200 and "application/json" in ct
        import json as _json
        tgt = _json.loads(body)
        assert abs(tgt["tx"]) < 1e-9 and abs(tgt["ty"]) < 1e-9

        st, ct, body = _GET(port, "/motor")
        assert st == 200 and "application/json" in ct
    finally:
        srv.shutdown()
        srv.server_close()
