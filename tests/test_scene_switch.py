"""
Runtime scene switching: the swap itself, and the endpoints that drive it.

The tests that matter here are the silent ones -- every failure mode this
covers produces a plausible-looking picture rather than an exception:

  * a swap under a running animation, where the cancelled animation could
    stamp the old scene's pose onto the new scene's goniometer
  * a failed load, which must leave every live object identical BY IDENTITY
  * the goniometer keeping the OLD scene's axes, because Goniometer captures
    scene.geometry by reference
  * a render observing two different scenes across one frame
  * a /move resolved against one scene's pixel size and another's axes -- the
    two fixtures differ 7.4x on purpose, so a torn read is unmistakable

All CPU-only: empty scenes, engine='numpy', and a faked render.

Run:  pytest tests/test_scene_switch.py -v
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.scene.materials import AIR
from loop_sim.scene.scene import Scene
from loop_sim.server import camera_server as cs
from loop_sim.server.camera_server import CameraServer

GEOM = {
    "camera_fast": [1, 0, 0],
    "camera_slow": [0, 1, 0],
    "beam_axis":   [0, 0, 1],
    "rotx_axis":   [1, 0, 0],
    "roty_axis":   [0, 1, 0],
    "rotz_axis":   [0, 0, 1],
}
CAM = {"width": 640, "height": 480, "pixel_size": 0.0074}

# The two shipped scenes differ 7.4x in pixel size (hampton 0.0074, mitegen
# 0.001).  The fixtures reproduce that exactly, because it is what turns a torn
# camera read from "hard to observe" into an assertion that cannot be fudged.
PX_A, PX_B = 0.0074, 0.001


def _write_scene(path, pixel_size=PX_A, rotx_axis=(1, 0, 0), width=640, height=480):
    """A minimal but genuinely loadable scene YAML.

    scene.load() needs only `geometry` and `camera`; with no `objects` it
    returns the same empty Scene the other CPU-only server tests build by hand,
    so a switch exercises the real load path without paying for a render.
    """
    geom = dict(GEOM, rotx_axis=list(rotx_axis))
    lines = ["geometry:"]
    for k, v in geom.items():
        lines.append(f"  {k}: [{v[0]}, {v[1]}, {v[2]}]")
    lines += ["camera:",
              f"  width: {width}",
              f"  height: {height}",
              f"  pixel_size: {pixel_size}",
              "  na_objective: 0.10",
              "  na_condenser: 0.07"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


def _empty_server(**kw):
    """A server on an in-memory empty scene: no path, no library, no renderer."""
    scene = Scene([], GEOM, CAM, {}, background=AIR)
    kw.setdefault("templates", False)
    return CameraServer(scene, host="127.0.0.1", port=0, engine="numpy", **kw)


class _FakeRenderServer(CameraServer):
    """CameraServer whose render step is a fake, as in test_server_singleflight.

    _snapshot_gonio is still called, deliberately: it is what exercises the
    _scene_lock -> _gonio_lock edge, and a fake that skipped it would leave the
    ordering this whole design turns on untested.
    """
    render_s = 0.01

    def _render_frame(self):
        self._snapshot_gonio()
        time.sleep(self.render_s)
        return b"\xff\xd8FAKE%04d\xff\xd9" % (self._render_count + 1,)


def _wait_for(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# The silent ones
# ---------------------------------------------------------------------------

def test_swap_under_running_animation(tmp_path):
    """A switch mid-animation must not let the cancelled move write anything.

    The animation is cancelled by the _anim_gen bump inside _install_bundle, so
    it returns without touching the goniometer -- which means the stage is at
    the NEW scene's home, not wherever the old move had reached.
    """
    a = _write_scene(tmp_path / "a.yaml")
    b = _write_scene(tmp_path / "b.yaml")
    srv = _FakeRenderServer(Scene([], GEOM, CAM, {}, background=AIR),
                            host="127.0.0.1", port=0, engine="numpy",
                            templates=False, scene_path=a,
                            scene_dir=str(tmp_path))
    try:
        srv.start(background=True)
        # A long, slow spin so the animation is certainly still in flight.
        srv._command_move({"drotx": "3600"}, 0.05)
        assert _wait_for(lambda: srv._anim_active), "animation never started"

        srv.switch_scene(b)

        assert srv._goniometer.geometry is srv._scene.geometry
        assert srv._goniometer.get() == srv._target_pose
        # Home: the cancelled animation contributed nothing.
        assert all(v == 0.0 for k, v in srv._goniometer.get().items()
                   if k != "zoom")
        # Workers survived, and the server still renders and still moves.
        assert srv._anim_thread.is_alive() and srv._bg_thread.is_alive()
        rc = srv._render_count
        srv._invalidate()
        assert _wait_for(lambda: srv._render_count > rc)
        srv._command_move({"drotx": "10"}, 50.0)
        assert _wait_for(lambda: not srv._anim_active, 10.0)
    finally:
        srv.shutdown()
        srv.server_close()


def test_failed_switch_is_an_identity_no_op(tmp_path):
    """A switch that raises must leave every live object the SAME object.

    Identity, not equality: the whole failure-safety argument is that
    _build_bundle writes nothing, so if any of these were merely equal it would
    mean something had been rebuilt and reassigned along the way.
    """
    a = _write_scene(tmp_path / "a.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        before = (srv._scene, srv._scene.geometry, srv._goniometer,
                  srv._templates, srv._tscene, srv._scene_path, srv._scene_gen)
        pose = dict(srv._goniometer.get())

        with pytest.raises(FileNotFoundError):
            srv.switch_scene(str(tmp_path / "nope.yaml"))

        after = (srv._scene, srv._scene.geometry, srv._goniometer,
                 srv._templates, srv._tscene, srv._scene_path, srv._scene_gen)
        for i, (x, y) in enumerate(zip(before, after)):
            assert x is y, f"element {i} was replaced by a failed switch"
        assert srv._goniometer.get() == pose
        assert srv._get_jpeg()          # still serving
    finally:
        srv.server_close()


def test_resolution_change_is_refused(tmp_path):
    """An MJPEG stream cannot change size without reconnecting its consumers."""
    a = _write_scene(tmp_path / "a.yaml")
    small = _write_scene(tmp_path / "small.yaml", width=320, height=240)
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        scene_before = srv._scene
        with pytest.raises(ValueError, match="constant frame size"):
            srv.switch_scene(small)
        assert srv._scene is scene_before
    finally:
        srv.server_close()


def test_goniometer_is_rebuilt_on_the_new_axes(tmp_path):
    """Goniometer captures scene.geometry BY REFERENCE.

    Reassigning _scene without rebuilding the goniometer leaves transform() on
    the old axes forever -- no exception, no wrong number anywhere the tests
    look, just a sample that rotates about the wrong axis.
    """
    a = _write_scene(tmp_path / "a.yaml", rotx_axis=(1, 0, 0))
    b = _write_scene(tmp_path / "b.yaml", rotx_axis=(0, 1, 0))
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        srv.switch_scene(b)
        assert srv._goniometer.geometry is srv._scene.geometry
        assert list(srv._goniometer.geometry["rotx_axis"]) == [0, 1, 0]
        # _snapshot_gonio builds Goniometer(self._scene.geometry) -- the exact
        # torn reference this guards.
        assert srv._snapshot_gonio().geometry is srv._scene.geometry

        # And the transform really turns about the new axis: rotx=90 must leave
        # a point on the new rotx_axis (y) fixed.
        import numpy as np
        T = srv._goniometer.set(rotx=90.0).transform()
        y = np.array([0.0, 1.0, 0.0])
        assert np.allclose(T[:3, :3] @ y, y, atol=1e-9)
    finally:
        srv.server_close()


def test_render_never_sees_two_scenes_in_one_frame(tmp_path):
    """A render must observe exactly one scene from entry to exit.

    Also asserts the swaps actually landed DURING renders (>= 2 generations
    observed); without that the test could pass by never racing at all.
    """
    a = _write_scene(tmp_path / "a.yaml")
    b = _write_scene(tmp_path / "b.yaml")

    class _Witness(_FakeRenderServer):
        render_s = 0.01
        def _render_frame(self):
            before = (self._scene_gen, id(self._scene), id(self._templates))
            out = super()._render_frame()
            after = (self._scene_gen, id(self._scene), id(self._templates))
            self.witness.append((before, after))
            return out

    srv = _Witness(Scene([], GEOM, CAM, {}, background=AIR),
                   host="127.0.0.1", port=0, engine="numpy",
                   templates=False, scene_path=a, scene_dir=str(tmp_path))
    srv.witness = []
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            srv._invalidate()
            time.sleep(0.002)

    try:
        srv.start(background=True)
        t = threading.Thread(target=churn, daemon=True)
        t.start()
        for i in range(12):
            srv.switch_scene(b if i % 2 == 0 else a)
        stop.set()
        t.join(5.0)
        assert _wait_for(lambda: len(srv.witness) >= 20, 10.0)

        torn = [w for w in srv.witness if w[0] != w[1]]
        assert not torn, f"{len(torn)} render(s) saw a torn scene swap: {torn[:3]}"
        gens = {w[0][0] for w in srv.witness}
        assert len(gens) >= 2, ("no swap ever landed during a render -- the "
                                f"test never raced (generations seen: {gens})")
    finally:
        stop.set()
        srv.shutdown()
        srv.server_close()


def test_move_never_mixes_two_scenes(tmp_path):
    """A pan must resolve wholly within one scene.

    The two fixtures differ 7.4x in pixel size, so a target resolved against
    one scene's camera and another's geometry lands on a value that is neither
    of the two legal answers.  Silently clamped in production; a hard assertion
    here.
    """
    a = _write_scene(tmp_path / "a.yaml", pixel_size=PX_A)
    b = _write_scene(tmp_path / "b.yaml", pixel_size=PX_B)
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    legal = {0.25 * 640 * PX_A, 0.25 * 640 * PX_B}
    seen, bad = set(), []
    stop = threading.Event()

    def mover():
        while not stop.is_set():
            # Absolute each time: resolve from home so the answer is exactly
            # one pan, not an accumulation.
            srv._set_pose_instant({"tx": 0.0})
            t = srv._command_move({"panx": "0.25"}, 1.0)
            tx = round(t["tx"], 12)
            seen.add(tx)
            if not any(abs(tx - v) < 1e-9 for v in legal):
                bad.append(tx)

    try:
        th = threading.Thread(target=mover, daemon=True)
        th.start()
        for i in range(30):
            srv.switch_scene(b if i % 2 == 0 else a)
        stop.set()
        th.join(5.0)
        assert not bad, f"target resolved across two scenes: {bad[:5]}"
        assert len(seen) >= 2, f"never exercised both scenes (saw {seen})"
    finally:
        stop.set()
        srv.server_close()


def test_no_deadlock_under_concurrent_switch_render_and_motor(tmp_path):
    """Runtime backstop for the lock order.

    _servable reads self._templates and is called from inside _gonio_lock by
    _set_pose_instant, so a self-locking version would give
    _gonio_lock -> _scene_lock and deadlock against _render_now, which holds
    _scene_lock and then wants _gonio_lock.  Two threads is all it takes.
    """
    a = _write_scene(tmp_path / "a.yaml")
    b = _write_scene(tmp_path / "b.yaml")
    srv = _FakeRenderServer(Scene([], GEOM, CAM, {}, background=AIR),
                            host="127.0.0.1", port=0, engine="numpy",
                            templates=False, scene_path=a,
                            scene_dir=str(tmp_path))
    stop = threading.Event()
    errors = []

    def hammer(fn):
        def run():
            try:
                while not stop.is_set():
                    fn()
                    time.sleep(0.001)   # contention, not a spin
            except Exception as exc:        # pragma: no cover - diagnostic
                errors.append(exc)
        return run

    try:
        srv.start(background=True)
        # _set_pose_instant and _render_now are the two sides of the cycle that
        # a self-locking _servable would close (_gonio_lock -> _scene_lock
        # against _scene_lock -> _gonio_lock); the other two widen the net.
        threads = [threading.Thread(target=hammer(f), daemon=True) for f in (
            lambda: srv._set_pose_instant({"rotx": 5.0}),
            lambda: srv._command_move({"drotx": "5"}, 50.0),
            lambda: srv._render_now(),
            lambda: srv._snapshot_gonio(),
        )]
        for t in threads:
            t.start()
        for i in range(8):
            srv.switch_scene(b if i % 2 == 0 else a)
        stop.set()
        for t in threads:
            t.join(5.0)
            assert not t.is_alive(), "a worker never finished -- deadlock"
        assert not errors, errors
    finally:
        stop.set()
        srv.shutdown()
        srv.server_close()


def _preempt_a_running_animation(srv, preempt):
    """Run one animation to speed, preempt it, and return once it has bailed.

    Driven directly rather than through the animator thread so the preempt
    lands at a known point: _anim_u is a LOCAL while a move is in flight and is
    only published on the preempt branch, so this is the only place the handoff
    is observable at all.
    """
    target = dict(srv._goniometer.get(), rotx=3600.0)
    th = threading.Thread(
        target=srv._run_animation,
        args=(target, 0.05, srv._anim_gen, W_PX[0], W_PX[1], srv._scene_gen),
        daemon=True)
    th.start()
    assert _wait_for(lambda: srv._goniometer.get()["rotx"] > 0.0), "never moved"
    preempt()
    th.join(5.0)
    assert not th.is_alive(), "preempted animation never returned"


W_PX = (CAM["width"], CAM["pixel_size"])


def test_speed_handoff_survives_an_ordinary_preempt(tmp_path):
    """Positive control for the test below.

    Without this, a _scene_gen guard that simply never handed off would look
    identical to one that correctly refuses only across a switch.
    """
    a = _write_scene(tmp_path / "a.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        def preempt():
            with srv._anim_cv:
                srv._anim_gen += 1        # a newer /move won
        _preempt_a_running_animation(srv, preempt)
        assert srv._anim_u > 0.0, "speed was not handed to the replacement move"
        assert srv._anim_delta is not None
    finally:
        srv.server_close()


def test_speed_handoff_does_not_cross_a_switch(tmp_path):
    """The one thing a cancelled animation DOES write must not cross scenes.

    _run_animation's preempt branch hands its speed and heading to whoever won,
    and it writes them AFTER a switch has released _anim_cv -- so clearing them
    in _install_bundle is not enough on its own.  Inherited across a swap, the
    first jog in the new scene would start at speed, in a different
    mm-per-pixel, on a stage that was just reset to home.
    """
    a = _write_scene(tmp_path / "a.yaml")
    b = _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        _preempt_a_running_animation(srv, lambda: srv.switch_scene(b))
        assert srv._anim_u == 0.0, "speed was inherited across a scene switch"
        assert srv._anim_delta is None
    finally:
        srv.server_close()


# ---------------------------------------------------------------------------
# Install semantics
# ---------------------------------------------------------------------------

def test_install_dirties_the_cache_without_bumping_frame_gen(tmp_path):
    """MJPEG consumers hold no scene state, and _jpeg_cache still holds the OLD
    frame at install time -- bumping the generation would push every client one
    duplicate stale part for no new information."""
    a = _write_scene(tmp_path / "a.yaml")
    b = _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        srv._get_jpeg()                       # populate the slot
        gen, cached = srv._frame_gen, srv._jpeg_cache
        assert cached is not None
        srv.switch_scene(b)
        assert srv._frame_gen == gen
        assert srv._jpeg_cache is cached      # no black gap for viewers
        assert srv._cache_dirty is True       # ...but a redraw is queued
    finally:
        srv.server_close()


def test_switch_clears_compiled_ok(tmp_path):
    """The compiled trace was traced against the OLD TorchScene."""
    a = _write_scene(tmp_path / "a.yaml")
    b = _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        srv._compiled_ok = True
        srv.switch_scene(b)
        assert srv._compiled_ok is False
    finally:
        srv.server_close()


@pytest.mark.parametrize("engine,templates,expected", [
    ("numpy", False, False),
    ("numpy", True,  False),
    ("torch", False, True),
    # The bug this fixes: the old guard only suppressed the TorchScene for
    # engine='auto', so --engine torch --templates on built one that templates
    # short-circuit and nothing ever calls.
    ("torch", True,  False),
    ("auto",  True,  False),
])
def test_want_torch_engine(engine, templates, expected):
    assert cs._want_torch_engine(engine, templates) is expected


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _serve(srv):
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


def _req(port, path, method="GET"):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method,
                               data=b"" if method == "POST" else None)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_scenes_endpoint_reports_every_scene(tmp_path):
    a = _write_scene(tmp_path / "a.yaml")
    _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    port = _serve(srv)
    try:
        st, d = _req(port, "/scenes")
        assert st == 200
        names = [s["name"] for s in d["scenes"]]
        assert names == ["a", "b"]
        assert [s["is_current_scene"] for s in d["scenes"]] == [True, False]
        # templates off: everything is servable, nothing needs a library.
        assert all(s["can_serve"] for s in d["scenes"])
        for s in d["scenes"]:
            assert s["library"]["status"] == "missing"
            assert s["preview"]["status"] == "missing"
    finally:
        srv.shutdown()
        srv.server_close()


def test_post_scene_rejects_unknown_and_traversal(tmp_path):
    a = _write_scene(tmp_path / "a.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    port = _serve(srv)
    try:
        for bad in ("../../etc/passwd", "", "/etc/passwd", "nope"):
            st, d = _req(port, f"/scene?path={urllib.parse.quote(bad)}", "POST")
            assert st == 400, (bad, st)
            assert "unknown scene" in d["error"]
        # Nothing was started, and nothing changed.
        assert srv._switch["status"] == "idle"
        assert srv._switch_thread is None
        assert srv._scene_gen == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_post_scene_rejects_bad_build_value(tmp_path):
    a = _write_scene(tmp_path / "a.yaml")
    _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    port = _serve(srv)
    try:
        st, d = _req(port, "/scene?path=b&build=sortof", "POST")
        assert st == 400 and "preview" in d["error"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_post_scene_switches_and_reports(tmp_path):
    a = _write_scene(tmp_path / "a.yaml")
    b = _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    port = _serve(srv)
    try:
        st, d = _req(port, "/scene?path=b", "POST")
        assert st == 202 and d["accepted"] is True
        assert srv.wait_for_switch(10)["status"] == "ok"
        st, d = _req(port, "/scene")
        assert st == 200 and d["name"] == "b"
        assert d["switch"]["status"] == "ok"
        assert os.path.abspath(srv._scene_path) == os.path.abspath(b)
    finally:
        srv.shutdown()
        srv.server_close()


def test_post_scene_409_while_a_switch_is_running(tmp_path):
    a = _write_scene(tmp_path / "a.yaml")
    _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    port = _serve(srv)
    gate = threading.Event()
    real = srv.switch_scene
    srv.switch_scene = lambda *args, **kw: (gate.wait(10), real(*args, **kw))[1]
    try:
        st, _ = _req(port, "/scene?path=b", "POST")
        assert st == 202
        st, d = _req(port, "/scene?path=b", "POST")
        assert st == 409 and d.get("busy") is True
        assert "already running" in d["error"]
        gate.set()
        assert srv.wait_for_switch(10)["status"] == "ok"
        # Terminal state releases the slot.
        st, _ = _req(port, "/scene?path=a", "POST")
        assert st == 202
        assert srv.wait_for_switch(10)["status"] == "ok"
    finally:
        srv.shutdown()
        srv.server_close()


def test_post_scene_reports_a_failed_switch_and_keeps_the_old_scene(tmp_path):
    a = _write_scene(tmp_path / "a.yaml")
    _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    port = _serve(srv)
    srv.switch_scene = lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        st, _ = _req(port, "/scene?path=b", "POST")
        assert st == 202
        term = srv.wait_for_switch(10)
        assert term["status"] == "error" and "boom" in term["error"]
        st, d = _req(port, "/scene")
        assert d["name"] == "a"                     # still serving the old one
        assert d["switch"]["status"] == "error"
        # A terminal error does not wedge the slot.
        st, _ = _req(port, "/scene?path=b", "POST")
        assert st == 202
    finally:
        srv.shutdown()
        srv.server_close()


def test_build_is_refused_without_cuda(tmp_path, monkeypatch):
    a = _write_scene(tmp_path / "a.yaml")
    _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    port = _serve(srv)
    monkeypatch.setattr(cs, "cuda_available", lambda: False)
    try:
        st, d = _req(port, "/scene?path=b&build=preview", "POST")
        assert st == 503 and d["cuda"] is False
        assert "--allow-cpu" in d["error"]
        # Refusing must not claim the slot or start a thread.
        assert srv._switch["status"] == "idle"
        assert srv._switch_thread is None
        # ...but a switch that needs no build still works with no GPU: that is
        # the entire point of the template path.
        st, _ = _req(port, "/scene?path=b", "POST")
        assert st == 202
        assert srv.wait_for_switch(10)["status"] == "ok"
    finally:
        srv.shutdown()
        srv.server_close()


def test_switch_needing_a_build_is_409_needs_build(tmp_path):
    """Templates on, no library in either root: the operator must choose."""
    a = _write_scene(tmp_path / "a.yaml")
    _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    port = _serve(srv)
    # Constructed with templates=False so __init__ never touches a library;
    # flipped on afterwards so the switch path takes the template branch. Never
    # construct with templates=True and no library -- that builds for real.
    srv._templates_flag = True
    srv._want_templates = True
    srv._library_kwargs = {"root": str(tmp_path / "libs")}
    srv._library_root = str(tmp_path / "libs")
    srv._preview_root = str(tmp_path / "libs_preview")
    try:
        st, d = _req(port, "/scene?path=b", "POST")
        assert st == 409 and d.get("needs_build") is True
        assert d.get("busy") is None
        assert "build=preview" in d["error"]
        assert srv._switch["status"] == "idle"
    finally:
        srv.shutdown()
        srv.server_close()


def test_switch_progress_is_parsed_into_state(tmp_path):
    a = _write_scene(tmp_path / "a.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        with srv._switch_cv:
            srv._switch.update(status="validating", scene=a, build="preview")
        srv._switch_progress("    41/72 frames    92.1s elapsed (2.25s/frame)")
        s = srv._switch_state()
        assert (s["frames_done"], s["frames_total"]) == (41, 72)
        assert s["percent"] == pytest.approx(56.9, abs=0.1)
        assert s["status"] == "building" and "41/72" in s["message"]

        # A line the pattern does not recognise still reaches the operator.
        srv._switch_progress("[frame-library] device cuda")
        s = srv._switch_state()
        assert s["message"] == "[frame-library] device cuda"
        assert (s["frames_done"], s["frames_total"]) == (41, 72)
    finally:
        srv.server_close()


def test_supersample_is_not_graded_unless_asked_for(tmp_path):
    """supersample is per-scene, so it must not be graded against a global.

    It follows each camera's sampling against the objective's Nyquist limit --
    4 for hampton's 7.4 um pixel, 1 for mitegen's 1.0 um one. Grading both
    against one server-wide default puts a permanent "stale" on whichever scene
    does not match, and no rebuild can clear it: the value being called stale is
    the correct one for that scene.
    """
    a = _write_scene(tmp_path / "a.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        assert "supersample" not in srv._grading_params()
        # ...but an operator who names one means it, and it is graded.
        srv._library_kwargs = dict(srv._library_kwargs, supersample=2)
        assert srv._grading_params()["supersample"] == 2
        # A preview always states its own, so it is always graded.
        assert srv._grading_params(preview=True)["supersample"] == \
            cs.PREVIEW_BUILD["supersample"]
    finally:
        srv.server_close()


def test_grading_still_catches_the_policy_parameters(tmp_path):
    """Dropping supersample must not blunt the rest of the check."""
    a = _write_scene(tmp_path / "a.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        p = srv._grading_params()
        for key in ("format", "psf", "n_cond", "step_deg", "pan_mm", "axis"):
            assert p.get(key) is not None, f"{key} silently stopped being graded"
    finally:
        srv.server_close()


def test_library_root_flag_does_not_break_staleness(tmp_path):
    """`root` must never reach the manifest comparison.

    Exactly the shape of the --jpeg-quality bug: a value that describes WHERE a
    library lives leaking into the keys that decide WHETHER it is current, so
    every launch invalidates the library it just found.
    """
    from argparse import Namespace
    from loop_sim.library.frame_library import build_params
    args = Namespace(n_cond=7, supersample=None, template_quality=None,
                     template_format=None, library_root=str(tmp_path))
    kw = cs.library_kwargs_from_args(args)
    assert kw["root"] == str(tmp_path)
    assert "root" not in build_params(**kw)
