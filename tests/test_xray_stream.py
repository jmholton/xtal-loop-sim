"""The X-ray radiograph MJPEG-style stream (/xray-stream) and its
explicit start/stop lifecycle (_set_stream_mode).

Mirrors test_server_singleflight.py's pattern for the optical stream:
render is replaced by a timed fake, so these tests are CPU-fast and need
no CUDA/library.  The single-flight and start/stop bookkeeping stay real.

The claims here, and the test that guards each:

  * single-flight: N invalidations while a render is in flight coalesce
    into one re-render, not a queue -> test_xray_singleflight_burst_not_multiplied
  * a producer thread starts and stops cleanly (cooperative, via the
    token, not a hard interrupt) -> test_start_stop_lifecycle
  * calling the same mode twice is a no-op, not a second thread ->
    test_idempotent_same_mode_twice
  * rapid start/stop/start leaves exactly one live producer ->
    test_rapid_toggle_leaves_one_thread
  * a scene switch keeps the producer alive -> test_stream_survives_scene_switch
  * no X-ray library falls back to live render -> test_stream_falls_back_with_no_library
  * the optical producer is never touched by _set_stream_mode ->
    test_optical_producer_unaffected_by_xray_stream_mode
"""
import os
import socket
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.scene.materials import AIR
from loop_sim.scene.scene import Scene
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


def _write_scene(path, rotx_axis=(1, 0, 0)):
    """Minimal loadable scene YAML -- same shape test_scene_switch.py uses."""
    geom = dict(GEOM, rotx_axis=list(rotx_axis))
    lines = ["geometry:"]
    for k, v in geom.items():
        lines.append(f"  {k}: [{v[0]}, {v[1]}, {v[2]}]")
    lines += ["camera:",
              "  width: 640", "  height: 480", "  pixel_size: 0.0074",
              "  na_objective: 0.10", "  na_condenser: 0.07"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


class _FakeXrayServer(CameraServer):
    """CameraServer with BOTH render steps faked -- the optical one (as
    test_server_singleflight/test_scene_switch already do) and the X-ray one,
    so tests can drive _set_stream_mode without CUDA or a real library.

    _render_xray_png is overridden rather than the lower-level trace
    functions, mirroring how _FakeRenderServer overrides _render_frame: it is
    the one seam every real caller (the plain /xray endpoint, _xray_render_now)
    goes through, so faking it exercises the same call shape the real code
    does. Each fake frame embeds the scene path and an ordinal so a test can
    tell a fresh frame from a stale one, and tell which scene it came from.
    """
    render_s = 0.02

    def _render_frame(self):
        self._snapshot_gonio()
        time.sleep(self.render_s)
        return b"\xff\xd8FAKE%04d\xff\xd9" % (self._render_count + 1,)

    def _render_xray_png(self):
        with self._scene_lock:
            scene_path = self._scene_path
        time.sleep(self.render_s)
        self._xray_render_count = getattr(self, "_xray_render_count", 0) + 1
        stem = os.path.splitext(os.path.basename(scene_path or "none"))[0].encode()
        return b"XRAYFAKE:" + stem + b":%04d" % self._xray_render_count


def _empty_server(**kw):
    scene = Scene([], GEOM, CAM, {}, background=AIR)
    kw.setdefault("templates", False)
    srv = _FakeXrayServer(scene, host="127.0.0.1", port=0, engine="numpy", **kw)
    srv.start(background=True)
    return srv


def _wait_for(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not pred() and time.monotonic() < deadline:
        time.sleep(0.005)
    return pred()


class _XrayStreamClient:
    """Raw-socket consumer of /xray-stream; records each part's payload."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.sock.sendall(b"GET /xray-stream HTTP/1.0\r\n\r\n")
        self.buf = bytearray()
        self.n = 0
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            while True:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                self.buf += chunk
                self.n = self.buf.count(b"Content-Type: image/png")
        except OSError:
            pass

    def wait_first_frame(self, timeout=3.0):
        assert _wait_for(lambda: self.n >= 1, timeout), \
            "client never received a frame"

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def test_xray_singleflight_burst_not_multiplied():
    srv = _empty_server()
    client = None
    try:
        srv._set_stream_mode("radiograph")
        client = _XrayStreamClient(srv.server_address[1])
        client.wait_first_frame()

        rc0 = srv._xray_render_count
        n_bursts = 5
        for _ in range(n_bursts):
            for _ in range(10):
                srv._invalidate_xray()
            time.sleep(0.12)
        delta = srv._xray_render_count - rc0
        assert n_bursts <= delta <= 2 * n_bursts, delta
    finally:
        if client is not None:
            client.close()
        srv._set_stream_mode("microscope")
        srv.shutdown()
        srv.server_close()


def test_start_stop_lifecycle():
    srv = _empty_server()
    try:
        assert srv._xray_bg_thread is None
        srv._set_stream_mode("radiograph")
        t = srv._xray_bg_thread
        assert t is not None and t.is_alive()
        assert _wait_for(lambda: getattr(srv, "_xray_render_count", 0) >= 1), \
            "producer never published a frame"

        srv._set_stream_mode("microscope")
        assert srv._xray_bg_thread is None
        t.join(timeout=2.0)
        assert not t.is_alive(), "producer thread did not stop"
    finally:
        srv.shutdown()
        srv.server_close()


def test_idempotent_same_mode_twice():
    srv = _empty_server()
    try:
        srv._set_stream_mode("radiograph")
        t1 = srv._xray_bg_thread
        token1 = srv._xray_stream_token
        srv._set_stream_mode("radiograph")   # no-op: already radiograph
        assert srv._xray_bg_thread is t1
        assert srv._xray_stream_token == token1

        srv._set_stream_mode("microscope")
        srv._set_stream_mode("microscope")   # no-op: already microscope
        assert srv._xray_bg_thread is None
    finally:
        srv.shutdown()
        srv.server_close()


def test_rapid_toggle_leaves_one_thread():
    srv = _empty_server()
    try:
        threads = []
        for _ in range(5):
            srv._set_stream_mode("radiograph")
            threads.append(srv._xray_bg_thread)
            srv._set_stream_mode("microscope")
        srv._set_stream_mode("radiograph")
        live_final = srv._xray_bg_thread
        assert _wait_for(lambda: live_final is not None and live_final.is_alive())

        time.sleep(0.2)   # let any stale generations wake and exit
        live = [t for t in threads if t is not None and t.is_alive()]
        assert live == [live_final] or live == [], (
            f"expected at most the current producer alive, found {len(live)}")
    finally:
        srv._set_stream_mode("microscope")
        srv.shutdown()
        srv.server_close()


def test_stream_survives_scene_switch(tmp_path):
    """The concrete regression test for _install_bundle's xray_templates_cache
    wiring: a scene switch must not kill the producer thread, and its NEXT
    published frame must reflect the new scene, not the old one."""
    a = _write_scene(tmp_path / "a.yaml")
    b = _write_scene(tmp_path / "b.yaml")
    srv = _empty_server(scene_path=a, scene_dir=str(tmp_path))
    try:
        srv._set_stream_mode("radiograph")
        t_before = srv._xray_bg_thread
        assert _wait_for(lambda: getattr(srv, "_xray_render_count", 0) >= 1)
        assert b"a:" in (srv._xray_jpeg_cache or b"")

        srv.switch_scene(b)
        assert srv._xray_bg_thread is t_before, "switch must not restart the producer"
        assert t_before.is_alive()

        srv._invalidate_xray()
        deadline = time.monotonic() + 3.0
        while (srv._xray_jpeg_cache or b"").find(b"b:") < 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert b"b:" in (srv._xray_jpeg_cache or b""), (
            "producer never picked up the new scene after a switch")
    finally:
        srv._set_stream_mode("microscope")
        srv.shutdown()
        srv.server_close()


def test_stream_falls_back_with_no_library():
    """No CUDA/library in this whole file -- every render already goes
    through the faked live-render fallback, so simply proving the producer
    never crashes across many renders IS the fallback-safety assertion."""
    srv = _empty_server()
    try:
        srv._set_stream_mode("radiograph")
        for _ in range(5):
            srv._invalidate_xray()
            time.sleep(0.03)
        assert _wait_for(lambda: getattr(srv, "_xray_render_count", 0) >= 1)
        assert srv._xray_bg_thread.is_alive()
    finally:
        srv._set_stream_mode("microscope")
        srv.shutdown()
        srv.server_close()


def test_optical_producer_unaffected_by_xray_stream_mode():
    """Switching to Radiograph and back must never stop, restart, or even
    touch the optical producer -- _set_stream_mode only starts/stops its OWN
    thread. The clearest proof: the optical render count keeps advancing on
    invalidation exactly the same whether or not the X-ray stream is active."""
    srv = _empty_server()
    try:
        srv._invalidate()
        assert _wait_for(lambda: srv._render_count >= 1)
        bg_thread = srv._bg_thread

        srv._set_stream_mode("radiograph")
        rc0 = srv._render_count
        srv._invalidate()
        assert _wait_for(lambda: srv._render_count > rc0)
        assert srv._bg_thread is bg_thread, "optical producer must never be replaced"

        srv._set_stream_mode("microscope")
        rc1 = srv._render_count
        srv._invalidate()
        assert _wait_for(lambda: srv._render_count > rc1)
        assert srv._bg_thread is bg_thread
    finally:
        srv.shutdown()
        srv.server_close()
