"""The camera-server calls xtalLoopSimDHS makes (its sim_link.CameraServerBackend).

  * GET /status: shape, and `moving` true during a /move, false after
    -> test_status_shape, test_status_moving_tracks_an_animated_move
  * /move duration=<s> lasts about that long; duration=0 is instant and stops
    a running move -> test_move_duration_sets_the_wall_time,
    test_zero_duration_is_instant_and_aborts
  * /video-trigger: open/closed round trip, and while open frames arrive at
    the receiver as image/jpeg POSTs -> test_video_trigger_round_trip,
    test_video_trigger_pushes_jpegs_to_the_receiver
  * ?camera=N on the VAPIX URLs: a zoom stop per camera, the goniometer
    untouched, unknown N a 400, other VAPIX parameters ignored
    -> test_camera_selects_a_zoom_stop, test_unknown_camera_is_400,
       test_other_vapix_parameters_are_ignored

One live server per module, templates off, numpy engine, at 64x48.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.scene.scene import load                     # noqa: E402
from loop_sim.server.camera_server import (               # noqa: E402
    CameraServer, parse_camera_zoom)

SCENE = os.path.join(REPO_ROOT, "data", "scene_files", "hampton_300um.yaml")
KEYS = {"tx", "ty", "tz", "rotx", "roty", "rotz", "zoom"}


class _Receiver(ThreadingHTTPServer):
    """A stand-in for pydhsfw's jpeg_receiver: records every POST."""

    def __init__(self):
        self.posts = []
        self.got_one = threading.Event()

        class H(BaseHTTPRequestHandler):
            def do_POST(h):
                body = h.rfile.read(int(h.headers.get("Content-Length", 0)))
                self.posts.append((h.headers.get("Content-Type"), body))
                self.got_one.set()
                h.send_response(200)
                h.send_header("Content-Length", "0")
                h.send_header("Connection", "close")
                h.end_headers()

            def log_message(h, *a):
                pass

        super().__init__(("127.0.0.1", 0), H)


@pytest.fixture(scope="module")
def receiver():
    rx = _Receiver()
    threading.Thread(target=rx.serve_forever, daemon=True).start()
    yield rx
    rx.shutdown()
    rx.server_close()


@pytest.fixture(scope="module")
def server(receiver):
    scene = load(SCENE, device="cpu")
    scene.camera_cfg["width"], scene.camera_cfg["height"] = 64, 48
    srv = CameraServer(scene, host="127.0.0.1", port=0, engine="numpy",
                       n_cond=1, scene_path=SCENE, templates=False,
                       prewarm=False, push_fps=50.0,
                       jpeg_receiver=f"http://127.0.0.1:{receiver.server_address[1]}/")
    srv.start(background=True)
    yield srv
    srv._set_video_trigger("closed")
    srv.shutdown()
    srv.server_close()


def _url(srv, path):
    return f"http://127.0.0.1:{srv.server_address[1]}{path}"


def _get(srv, path):
    with urllib.request.urlopen(_url(srv, path), timeout=30) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def _post(srv, path):
    req = urllib.request.Request(_url(srv, path), data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _status(srv):
    return json.loads(_get(srv, "/status")[2])


def _wait_idle(srv, timeout=10.0):
    deadline = time.monotonic() + timeout
    while _status(srv)["moving"]:
        assert time.monotonic() < deadline, "move never finished"
        time.sleep(0.02)


def test_status_shape(server):
    s = _status(server)
    assert set(s) == {"positions", "target", "moving"}
    assert set(s["positions"]) == KEYS and set(s["target"]) == KEYS
    assert isinstance(s["moving"], bool)
    assert all(isinstance(v, float) for v in s["positions"].values())


def test_status_moving_tracks_an_animated_move(server):
    _post(server, "/move?rotx=0&duration=0")
    _post(server, "/move?rotx=40&duration=0.4")
    s = _status(server)
    assert s["moving"] is True
    assert s["target"]["rotx"] == pytest.approx(40.0)
    _wait_idle(server)
    s = _status(server)
    assert s["moving"] is False
    assert s["positions"]["rotx"] == pytest.approx(40.0)


def test_move_duration_sets_the_wall_time(server):
    _post(server, "/move?rotx=0&duration=0")
    t0 = time.monotonic()
    _post(server, "/move?rotx=10&duration=0.3")
    _wait_idle(server)
    elapsed = time.monotonic() - t0
    assert 0.25 <= elapsed <= 0.6, f"a 0.3 s move took {elapsed:.3f} s"
    assert _status(server)["positions"]["rotx"] == pytest.approx(10.0)


def test_zero_duration_is_instant_and_aborts(server):
    _post(server, "/move?rotx=0&duration=0")
    s = _status(server)
    assert s["moving"] is False and s["positions"]["rotx"] == pytest.approx(0.0)

    _post(server, "/move?rotx=90&duration=5")
    time.sleep(0.3)
    here = _status(server)["positions"]["rotx"]
    assert 0.0 < here < 90.0
    _post(server, f"/move?rotx={here}&duration=0")        # the DHS's abort
    s = _status(server)
    assert s["moving"] is False
    assert s["positions"]["rotx"] == pytest.approx(here)
    time.sleep(0.2)
    assert _status(server)["positions"]["rotx"] == pytest.approx(here)


def test_negative_duration_is_400(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server, "/move?rotx=1&duration=-1")
    assert exc.value.code == 400


def test_video_trigger_round_trip(server):
    assert _post(server, "/video-trigger?state=closed") == {"state": "closed"}
    assert json.loads(_get(server, "/video-trigger")[2]) == {"state": "closed"}
    assert _post(server, "/video-trigger?state=open") == {"state": "open"}
    assert json.loads(_get(server, "/video-trigger")[2]) == {"state": "open"}
    assert _post(server, "/video-trigger?state=closed") == {"state": "closed"}
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(server, "/video-trigger?state=ajar")
    assert exc.value.code == 400


def test_video_trigger_pushes_jpegs_to_the_receiver(server, receiver):
    receiver.posts.clear()
    receiver.got_one.clear()
    _post(server, "/video-trigger?state=open")
    try:
        assert receiver.got_one.wait(15.0), "no frame reached the receiver"
        ctype, body = receiver.posts[0]
        assert ctype == "image/jpeg"
        assert body[:2] == b"\xff\xd8"
    finally:
        _post(server, "/video-trigger?state=closed")
    time.sleep(0.6)                      # a pusher mid-POST finishes this one
    n = len(receiver.posts)
    _post(server, "/move?drotx=5&duration=0")
    time.sleep(0.6)
    assert len(receiver.posts) == n, "frames still pushed after closed"


def test_camera_selects_a_zoom_stop(server):
    _post(server, "/move?rotx=0&zoom=1&duration=0")
    time.sleep(2 * server._settle_delay)  # past the preview, onto the settled frame
    _st, ct1, cam1 = _get(server, "/axis-cgi/jpg/image.cgi?camera=1")
    _st, ct2, cam2 = _get(server, "/axis-cgi/jpg/image.cgi?camera=2")
    assert ct1 == ct2 == "image/jpeg"
    assert cam2[:2] == b"\xff\xd8"
    assert cam1 != cam2
    assert _status(server)["positions"]["zoom"] == pytest.approx(1.0)
    _st, _ct, plain = _get(server, "/axis-cgi/jpg/image.cgi")
    assert plain == cam1                 # camera 1 is the goniometer's zoom


def test_unknown_camera_is_400(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/axis-cgi/jpg/image.cgi?camera=7")
    assert exc.value.code == 400
    assert "unknown camera" in json.loads(exc.value.read())["error"]


def test_other_vapix_parameters_are_ignored(server):
    st, ct, body = _get(server, "/axis-cgi/jpg/image.cgi?resolution=704x480"
                                "&compression=0&clock=1&date=1&text=0&fps=5")
    assert st == 200 and ct == "image/jpeg" and body[:2] == b"\xff\xd8"


def test_parse_camera_zoom():
    assert parse_camera_zoom("1:1.0,2:0.5,3:0.25") == {1: 1.0, 2: 0.5, 3: 0.25}
    for bad in ("1", "1:x", "1:0"):
        with pytest.raises(ValueError):
            parse_camera_zoom(bad)
