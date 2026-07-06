"""
Single-flight frame production + event-driven MJPEG pacing.

While serving, the background render loop is the SOLE producer of frames:
MJPEG/snapshot handlers are pure consumers of the published (generation-
stamped) frame slot, so N streaming clients cost one render per dirty state
instead of N+1, and the wire rate is no longer hard-capped at 5 fps.

The render+encode step (_render_frame) is replaced by a timed fake — real
frames with engine='numpy' take minutes — so these tests are CPU-fast and
deterministic.  The single-flight bookkeeping in _render_now (dirty claim,
generation bump, notify) stays real.

Run:  pytest tests/test_server_singleflight.py -v
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


class _FakeRenderServer(CameraServer):
    """CameraServer with the render+encode step replaced by a ~30 ms fake.

    Each fake frame embeds the render ordinal so tests can tell a fresh
    frame from a keepalive resend of the cached one.
    """
    render_s = 0.03

    def _render_frame(self):
        time.sleep(self.render_s)
        return b"\xff\xd8FAKE%04d\xff\xd9" % (self._render_count + 1,)


def _make_server(started=True, **kw):
    scene = Scene([], GEOM, CAM, {}, background=AIR)
    srv = _FakeRenderServer(scene, host="127.0.0.1", port=0,
                            engine="numpy", **kw)
    if started:
        srv.start(background=True)
        deadline = time.monotonic() + 5.0
        while srv._render_count < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert srv._render_count >= 1, "background producer never rendered"
    return srv


class _MjpegClient:
    """Raw-socket MJPEG consumer; records the arrival time of each frame."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.sock.sendall(b"GET /axis-cgi/mjpg/video.cgi HTTP/1.0\r\n\r\n")
        self.buf = bytearray()
        self.frame_times = []
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        seen = 0
        try:
            while True:
                chunk = self.sock.recv(65536)
                if not chunk:
                    break
                self.buf += chunk
                n = self.buf.count(b"--myboundary")
                now = time.monotonic()
                for _ in range(n - seen):
                    self.frame_times.append(now)
                seen = n
        except OSError:
            pass

    def n_frames(self):
        return len(self.frame_times)

    def wait_first_frame(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while not self.frame_times and time.monotonic() < deadline:
            time.sleep(0.005)
        assert self.frame_times, "client never received a frame"

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Single-flight: renders scale with dirty states, not with client count
# ---------------------------------------------------------------------------

def test_singleflight_burst_not_multiplied_by_clients():
    srv = _make_server()
    port = srv.server_address[1]
    c1 = c2 = None
    try:
        c1 = _MjpegClient(port)
        c2 = _MjpegClient(port)
        c1.wait_first_frame()
        c2.wait_first_frame()

        rc0 = srv._render_count
        n_bursts = 5
        for _ in range(n_bursts):
            for _ in range(10):
                srv._invalidate()      # burst: one dirty state
            time.sleep(0.15)           # let the coalesced render(s) finish
        delta = srv._render_count - rc0

        # Each burst = one dirty state -> 1 render, plus at most 1 coalesced
        # re-render when an invalidation lands mid-render.  The old code let
        # every streaming client render too (N+1 per dirty state).
        assert n_bursts <= delta <= 2 * n_bursts, delta

        # Both consumers actually streamed during the bursts.
        assert c1.n_frames() >= 3 and c2.n_frames() >= 3
    finally:
        for c in (c1, c2):
            if c is not None:
                c.close()
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# fps_limit is a knob, not a hard 5 fps wire cap
# ---------------------------------------------------------------------------

def test_stream_exceeds_old_5fps_cap():
    srv = _make_server(fps_limit=15.0)
    port = srv.server_address[1]
    stop = threading.Event()

    def invalidator():                 # ~animator cadence
        while not stop.is_set():
            srv._invalidate()
            time.sleep(0.02)

    client = None
    try:
        client = _MjpegClient(port)
        client.wait_first_frame()
        threading.Thread(target=invalidator, daemon=True).start()
        n0 = client.n_frames()
        time.sleep(1.5)
        delta = client.n_frames() - n0
        # The old code slept 1/fps_limit=0.2 s per push -> <= ~8 frames in
        # 1.5 s.  At fps_limit=15 with a 30 ms fake render we expect ~22.
        assert delta >= 12, delta
    finally:
        stop.set()
        if client is not None:
            client.close()
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# Idle keepalive: the cached frame is resent so clients don't time out
# ---------------------------------------------------------------------------

def test_idle_keepalive_repeats_frame():
    srv = _make_server()
    port = srv.server_address[1]
    client = None
    try:
        client = _MjpegClient(port)
        client.wait_first_frame()
        deadline = time.monotonic() + 2.5
        while client.n_frames() < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.n_frames() >= 2, "no keepalive resend"
        assert client.frame_times[1] - client.frame_times[0] <= 1.5
        # It is the SAME cached frame (startup render #1), not a new render.
        assert client.buf.count(b"FAKE0001") >= 2
        assert srv._render_count == 1
    finally:
        if client is not None:
            client.close()
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# Snapshot path: consumes the producer's frame, never renders itself
# ---------------------------------------------------------------------------

def test_snapshot_consumes_producer_frame():
    srv = _make_server()
    try:
        rc0 = srv._render_count        # == 1 (startup frame)
        j0  = srv._get_jpeg()          # clean cache: served with no render
        assert srv._render_count == rc0

        srv._invalidate()
        j1 = srv._get_jpeg()           # waits for the producer's next frame
        deadline = time.monotonic() + 3.0
        while j1 == j0 and time.monotonic() < deadline:
            time.sleep(0.005)          # in-flight claim window: retry
            j1 = srv._get_jpeg()
        assert j1 != j0                            # fresh frame observed
        assert srv._render_count == rc0 + 1        # rendered once, by the bg loop
    finally:
        srv.shutdown()
        srv.server_close()


def test_get_jpeg_without_start_renders_sync_once():
    """Before start() there is no producer: _get_jpeg renders synchronously
    (the mode the settle-parity tests rely on), and only when dirty."""
    srv = _make_server(started=False)
    try:
        assert srv._render_count == 0
        j1 = srv._get_jpeg()           # no bg thread: sync render
        assert srv._render_count == 1
        j2 = srv._get_jpeg()           # clean cache: no re-render
        assert j2 == j1
        assert srv._render_count == 1
        srv._invalidate()
        srv._get_jpeg()                # dirty + still no bg thread: sync again
        assert srv._render_count == 2
    finally:
        srv.server_close()
