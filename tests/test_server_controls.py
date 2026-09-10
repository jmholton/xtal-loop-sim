"""
Interactive-controls tests: the pure move/geometry helpers in camera_server,
plus a lightweight HTTP smoke test of the new endpoints.

The animation + recenter math is factored into pure functions so it can be
asserted exactly without a live render or GPU:

  * resolve_target   -- absolute / relative / screen-fraction-pan resolution
  * move_duration    -- FOV-crosses-in-2s, 360deg/s, speed-dial scaling
  * recenter_target  -- click-to-centre (exact 3-D, Δtrans = -R^T p_lab)

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
    velocity_step, DEFAULT_RAMP_S, ANIM_DT,
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

def test_translation_crosses_screen_in_four_seconds():
    """Pins the current rate constant; see docs/DECISIONS.md 2026-08-06
    (stage speeds halved)."""
    target = dict(REST, tx=FOV_W)        # pan exactly one screen width
    assert move_duration(REST, target, 1.0, W, PX) == pytest.approx(4.0, rel=1e-9)


def test_rotation_is_180_deg_per_second():
    target = dict(REST, rotx=360.0)      # one full turn = 30 rpm
    assert move_duration(REST, target, 1.0, W, PX) == pytest.approx(2.0, rel=1e-9)


def test_speed_dial_scales_duration_inversely():
    target = dict(REST, tx=FOV_W)
    assert move_duration(REST, target, 2.0, W, PX) == pytest.approx(2.0, rel=1e-9)   # faster
    assert move_duration(REST, target, 0.5, W, PX) == pytest.approx(8.0, rel=1e-9)   # slow-mo


def test_zoom_rate_halved_too():
    """Zoom is the microscope, not the goniometer, but it was rescaled with
    everything else so the dial means one thing across all axes."""
    assert move_duration(REST, dict(REST, zoom=3.0), 1.0, W, PX) == pytest.approx(1.0, rel=1e-9)


def test_duration_is_slowest_parameter():
    target = dict(REST, tx=FOV_W, rotx=360.0)   # 4.0 s vs 2.0 s → 4.0 s
    assert move_duration(REST, target, 1.0, W, PX) == pytest.approx(4.0, rel=1e-9)


# ---------------------------------------------------------------------------
# recenter_target
# ---------------------------------------------------------------------------

def test_zero_distance_move_stays_zero():
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
# HTTP smoke test (no rendering -- only the lightweight control endpoints)
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


# ---------------------------------------------------------------------------
# Animator preemption: a cancelled animation must not write anything
# ---------------------------------------------------------------------------

def test_preempted_animation_writes_nothing():
    """A superseded animation must not stamp its pose after losing the race.

    The generation check and the goniometer write must be inside the same
    critical section: a gap between them would let a preempt land between
    check and write, stamping the loser's pose on top of the winner's (a
    newer /move, an instant /motor, or a different scene's pose entirely).
    Both the step loop and the settle block are gen-checked.
    """
    scene = Scene([], GEOM, CAM, {}, background=AIR)
    srv = CameraServer(scene, host="127.0.0.1", port=0, engine="numpy")
    try:
        # Park the stage somewhere unambiguous, then invalidate the generation
        # the animation is about to run under -- exactly what a preempt does.
        with srv._gonio_lock:
            srv._goniometer.set(tx=0.5, ty=0.25, rotx=90.0)
            before = srv._goniometer.get()
        stale_gen = srv._anim_gen
        srv._anim_gen += 1                       # someone else won

        target = dict(before, tx=-9.0, ty=-9.0, rotx=-9.0)
        srv._run_animation(target, 1.0, stale_gen, W, PX)

        with srv._gonio_lock:
            after = srv._goniometer.get()
        assert after == before, (
            f"a preempted animation moved the stage: {before} -> {after}")
    finally:
        srv.server_close()


def test_current_animation_still_reaches_its_target():
    """The guard must not break the normal case: an uncontested animation
    still lands exactly on its target."""
    scene = Scene([], GEOM, CAM, {}, background=AIR)
    srv = CameraServer(scene, host="127.0.0.1", port=0, engine="numpy")
    try:
        target = dict(REST, tx=0.01, rotx=5.0)
        srv._run_animation(target, 50.0, srv._anim_gen, W, PX)   # fast: no sleep-bound wait
        with srv._gonio_lock:
            after = srv._goniometer.get()
        for k in ("tx", "rotx"):
            assert after[k] == pytest.approx(target[k]), k
        assert srv._anim_active is False
    finally:
        srv.server_close()


# ---------------------------------------------------------------------------
# velocity_step: trapezoidal motion, and what happens across a preempt
# ---------------------------------------------------------------------------

def _run_profile(duration, ramp=DEFAULT_RAMP_S, dt=ANIM_DT, u0=0.0, max_steps=100000):
    """Integrate a whole move, returning the (pos, u) trace."""
    pos, u, trace = 0.0, u0, [(0.0, u0)]
    for _ in range(max_steps):
        if pos >= 1.0:
            break
        pos, u = velocity_step(pos, u, dt, duration, ramp)
        trace.append((pos, u))
    return trace


def test_profile_is_trapezoidal():
    """Speed must rise, hold, then fall -- not jump to full and stop dead."""
    trace = _run_profile(2.0)
    us = [u for _, u in trace]
    peak = max(us)
    assert peak == pytest.approx(1.0, abs=1e-9), "never reached full speed"
    top = us.index(peak)
    assert top > 1, "reached full speed instantly -- no acceleration"
    assert us[-1] == 0.0, "did not come to rest"
    # rising then falling, with a cruise in between for a move this long
    assert all(us[i] <= us[i+1] + 1e-12 for i in range(top)), "speed dipped while ramping up"
    assert us.count(peak) > 5, "no cruise phase on a 2 s move"


def test_profile_arrives_exactly_and_stops():
    for duration in (0.05, 0.2, 1.0, 5.0):
        trace = _run_profile(duration)
        assert trace[-1][0] == 1.0, f"{duration}s move did not arrive"
        assert trace[-1][1] == 0.0, f"{duration}s move did not stop"


def test_short_move_is_triangular():
    """Too short to reach full speed: it accelerates then brakes, no cruise."""
    trace = _run_profile(0.05)          # far below the 0.15 s ramp
    us = [u for _, u in trace]
    assert max(us) < 1.0, "a very short move should never reach full speed"
    assert us[-1] == 0.0


def test_ramp_time_is_distance_independent():
    """Fixed acceleration: reaching full speed takes the same time whether the
    move is short or long -- that is what makes it a real motor rather than an
    eased tween."""
    for duration in (1.0, 4.0):
        trace = _run_profile(duration)
        steps_to_full = next(i for i, (_, u) in enumerate(trace) if u >= 1.0)
        assert steps_to_full * ANIM_DT == pytest.approx(DEFAULT_RAMP_S, abs=ANIM_DT)


def test_carried_velocity_skips_the_ramp():
    """A move that inherits speed from the one it preempted must not start
    from rest -- that restart is exactly the per-click stutter this avoids."""
    from_rest = _run_profile(1.0, u0=0.0)
    at_speed  = _run_profile(1.0, u0=1.0)
    assert at_speed[1][0] > from_rest[1][0], "inherited speed did not move sooner"
    assert len(at_speed) < len(from_rest), "inherited speed did not finish sooner"
    assert at_speed[-1][0] == 1.0 and at_speed[-1][1] == 0.0


def test_zero_duration_is_instant():
    assert velocity_step(0.0, 0.0, ANIM_DT, 0.0) == (1.0, 0.0)
