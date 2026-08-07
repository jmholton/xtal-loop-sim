"""
AXIS-compatible HTTP camera server.

Endpoints
---------
GET /axis-cgi/mjpg/video.cgi
    MJPEG stream  (multipart/x-mixed-replace).
    Client connects and receives a continuous stream of JPEG frames.

GET /axis-cgi/jpg/image.cgi
    Single JPEG snapshot of the current view.

GET/POST /motor
    Set motor positions.  Parameters: tx, ty, tz, rotx, roty, rotz, zoom.
    All are optional; unspecified motors keep their current values.
    Returns JSON with current motor state.

GET /beam
    Compute and return X-ray beam illuminated volumes + attenuation as JSON.

GET /xray
    Per-pixel X-ray transmission map (radiograph) as a grayscale PNG,
    registered to the optical view (bright = transmitted, dark = absorbed).

GET / , /index.html
    Interactive control page (live MJPEG view + pan/rotate/zoom buttons,
    editable angle box, click-to-recentre, center crosshair, speed dial).

GET/POST /move
    Animated move.  Absolute motor keys (tx, ty, ...), relative deltas
    (dtx, drotx, ...), screen-fraction pan (panx, pany), and `speed`
    (>1 faster, <1 slow-motion).  The sample interpolates to the target
    instead of teleporting.  Returns the target motor state as JSON.

GET /recenter?px=COL&py=ROW
    Animated move that brings the clicked pixel to the image centre.

Usage
-----
Command line (preferred):

    python -m loop_sim.server.camera_server --scene scene_files/hampton_300um.yaml
    python -m loop_sim.server.camera_server --preview-mode off   # every frame exact

Or from Python:

    from loop_sim.server.camera_server import CameraServer
    from loop_sim.scene.scene import load
    from loop_sim.motors.goniometer import Goniometer

    scene = load("scene_files/hampton_300um.yaml")
    server = CameraServer(scene, host="0.0.0.0", port=8080)
    server.start()   # blocks; Ctrl-C to stop
"""
import glob
import io
import json
import os
import re
import threading
import time
import urllib.parse
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ..motors.goniometer import Goniometer
from ..renderer.microscope import render as microscope_render
from ..renderer.beam        import beam_volumes_json
from ..library.frame_library import (CPU_BUILD_REFUSAL,
                                     DEFAULT_PREVIEW_ROOT as _LIB_PREVIEW_ROOT,
                                     DEFAULT_ROOT as _LIB_DEFAULT_ROOT,
                                     PREVIEW_BUILD, build_params, cuda_available,
                                     frame_for_angle, library_diff, library_dir,
                                     library_status, pose_crop, servable_pose)

_MJPEG_BOUNDARY = b"--myboundary"
_MOTOR_KEYS = ("tx", "ty", "tz", "rotx", "roty", "rotz", "zoom")

# Scenes offered for runtime switching. Anchored to the repo, like the frame
# library root, so the answer does not depend on the directory the server was
# launched from -- a cwd-relative glob would list nothing while the libraries
# it pairs with still resolved.
_SCENE_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scene_files")


# ---------------------------------------------------------------------------
# Template replay
# ---------------------------------------------------------------------------
class TemplateSource:
    """Serves frames from a pre-rendered spindle sweep.

    A frame is produced by picking the nearest template angle, cropping the
    window the pose asks for, scaling it to the camera resolution, and blurring
    by the defocus the depth component implies.  No raytracing, no GPU.

    Decoded templates are cached (they are large -- tens of MB each at 4x
    supersample -- so the cache is deliberately small; decoding is ~20 ms and
    is not the bottleneck).
    """

    def __init__(self, manifest, lib_dir, jpeg_quality=85, cache_size=8):
        self.manifest = manifest
        self.lib_dir = lib_dir
        self.jpeg_quality = jpeg_quality
        self._cache_size = max(1, int(cache_size))
        self._cache = {}
        self._order = []
        self._lock = threading.Lock()
        self._warned_clamp = False

    def _frame(self, name):
        from PIL import Image
        with self._lock:
            img = self._cache.get(name)
            if img is not None:
                self._order.remove(name)
                self._order.append(name)
                return img
        img = Image.open(os.path.join(self.lib_dir, name)).convert("RGB")
        img.load()
        with self._lock:
            # Re-check: two threads can miss on the same frame and both decode.
            # Inserting twice would leave the name in _order twice while _cache
            # holds it once, so the eviction loop would over-evict for good.
            cached = self._cache.get(name)
            if cached is not None:
                return cached
            self._cache[name] = img
            self._order.append(name)
            while len(self._order) > self._cache_size:
                self._cache.pop(self._order.pop(0), None)
        return img

    def render(self, pose):
        """JPEG bytes for a motor pose dict."""
        from PIL import Image, ImageFilter

        man = self.manifest
        angle = float(pose.get(man["axis"], 0.0))
        rec = frame_for_angle(man, angle)
        box, out_size, sigma, note = pose_crop(
            man, tx=float(pose.get("tx", 0.0)), ty=float(pose.get("ty", 0.0)),
            tz=float(pose.get("tz", 0.0)), angle_deg=angle,
            zoom=float(pose.get("zoom", 1.0)), clamp=True)
        if note and not self._warned_clamp:
            # Say it once: a clamped pose looks entirely plausible on screen.
            self._warned_clamp = True
            print(f"[templates] request clamped to what the library can serve: {note}")

        # Resample straight from the float source box -- cropping to integers
        # first would quantise the registration and the magnification.
        img = self._frame(rec["file"]).resize(out_size, Image.BILINEAR, box=box)
        if sigma > 0.05:
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Move resolution + animation geometry (pure functions — unit-tested directly)
# ---------------------------------------------------------------------------

def resolve_target(current, params, W, H, pixel_size, geometry=None):
    """Resolve a /move request into a full absolute target motor dict.

    Supports three param styles, applied in order:
      * absolute    — tx, ty, tz, rotx, roty, rotz, zoom
      * relative    — d<key> (dtx, drotx, dzoom, ...) added to the running base
      * screen pan  — panx/pany (fraction of the field of view) → mm, in IMAGE
        axes: the pad moves the sample the way it looks on screen at any
        spindle angle.  The XYZ stage rides on the spindle, so screen-vertical
        is ty at phi=0 but tz at phi=90; the pan is therefore built in lab
        space and mapped into motor space by Rᵀ, exactly as recenter_target
        does.  `geometry` supplies the camera axes.  Without it the pan is
        applied to tx/ty directly, which only matches the image at zero
        rotation — the server always passes geometry.

    `current` is the running command target (not the in-flight pose), so rapid
    button clicks accumulate (four 0.25-screen pans = one full screen).
    """
    t = dict(current)
    for k in _MOTOR_KEYS:
        if k in params:
            t[k] = float(params[k])
    for k in _MOTOR_KEYS:
        dk = "d" + k
        if dk in params:
            t[k] = t[k] + float(params[dk])
    eff_px = pixel_size / max(t.get("zoom", 1.0), 1e-6)
    if "panx" in params or "pany" in params:
        dx = float(params.get("panx", 0.0)) * W * eff_px
        dy = float(params.get("pany", 0.0)) * H * eff_px
        if geometry is None:
            t["tx"] = t["tx"] + dx
            t["ty"] = t["ty"] + dy
        else:
            fast = np.array(geometry.get("camera_fast", [1, 0, 0]), dtype=float)
            slow = np.array(geometry.get("camera_slow", [0, 1, 0]), dtype=float)
            beam = np.array(geometry.get("beam_axis",   [0, 0, 1]), dtype=float)
            p_lab  = dx * fast + dy * slow
            R      = Goniometer(geometry).set(**t).transform()[:3, :3]
            dtrans = R.T @ p_lab
            t["tx"] = t["tx"] + float(dtrans @ fast)
            t["ty"] = t["ty"] + float(dtrans @ slow)
            t["tz"] = t["tz"] + float(dtrans @ beam)
    t["zoom"] = max(t.get("zoom", 1.0), 1e-3)
    return t


def move_duration(start, target, speed, W, pixel_size,
                  cross_time=4.0, rot_rate=180.0, zoom_rate=2.0):
    """Cruise-speed duration (s) for a move: distance / maximum velocity.

    Translations cross the screen *width* in `cross_time` s (scene- and
    zoom-aware via eff_px), rotations spin at `rot_rate` deg/s (180 = 30 rpm),
    zoom changes at `zoom_rate` /s.  The move lasts as long as its slowest
    parameter needs, divided by the speed-dial multiplier (>1 faster).

    This is the time at CONSTANT full speed; the real move takes longer,
    because `velocity_step` ramps in and out of it.  It is the input to that
    stepper, not the wall-clock duration.

    Rates were halved on 2026-08-06: the previous values (2 s screen crossing,
    360 deg/s) were about twice as fast as the real goniometer looks, so what
    used to need the speed dial at 0.5x is now 1.0x.
    """
    zoom0 = max(start.get("zoom", 1.0), 1e-6)
    eff_px = pixel_size / zoom0
    trans_rate = (W * eff_px) / cross_time if cross_time > 0 else float("inf")  # mm/s
    durs = [0.0]
    for k in ("tx", "ty", "tz"):
        durs.append(abs(target[k] - start[k]) / trans_rate)
    for k in ("rotx", "roty", "rotz"):
        durs.append(abs(target[k] - start[k]) / rot_rate)
    durs.append(abs(target.get("zoom", zoom0) - zoom0) / zoom_rate)
    return max(durs) / max(speed, 1e-6)


# Time to reach full speed.  A real stage has a fixed acceleration, so this is
# constant whatever the distance -- short moves simply never get there.  Kept
# brief: enough to remove the instant start/stop, not enough to feel sluggish.
DEFAULT_RAMP_S = 0.15
# Control-loop tick.  Also the animation's step period.
ANIM_DT = 0.02


def velocity_step(pos, u, dt, linear_duration, ramp=DEFAULT_RAMP_S):
    """Advance one control tick along a move, returning `(pos, u)`.

    `pos` is the fraction of the move completed, 0..1.  `u` is the current
    speed as a fraction of the maximum -- deliberately dimensionless, so it
    survives being carried into a DIFFERENT move when one preempts another.

    This is a velocity profile, not a position curve: speed ramps up at a
    fixed acceleration, holds, then ramps down so the stage arrives stopped
    (trapezoidal, or triangular when the move is too short to reach full
    speed).  Braking starts when the distance left equals the distance needed
    to stop, which is what makes the arrival land on the target rather than
    past it.

    Velocity is STATE rather than a function of elapsed time, and that is the
    point: when a move is preempted -- the common case being a burst of jog
    clicks -- the replacement starts from the speed the stage is actually
    doing.  Recomputing a position curve from t=0 would silently decelerate to
    a stop at every click, which is exactly the per-click stutter this whole
    path exists to avoid.
    """
    D = linear_duration
    if D <= 0.0 or pos >= 1.0:
        return 1.0, 0.0
    ramp = max(float(ramp), 1e-6)
    # Distance still needed to brake to rest, in units of the move's length.
    stop_dist = u * u * ramp / (2.0 * D)
    if (1.0 - pos) <= stop_dist:
        u = max(0.0, u - dt / ramp)
    else:
        u = min(1.0, u + dt / ramp)
    pos = pos + u * dt / D
    if pos >= 1.0:
        return 1.0, 0.0
    return pos, u


def recenter_target(col, row, state, geometry, camera_cfg):
    """Target pose that brings pixel (col, row) to the image centre.

    Exact 3-D centring of the focal point: Δtrans = −Rᵀ·p_lab, where
    p_lab = (col−W/2)·eff_px·fast + (row−H/2)·eff_px·slow and R is the current
    rotation.  Only translation changes.  At zero rotation this reduces to
    Δtx = −(col−W/2)·eff_px, Δty = −(row−H/2)·eff_px.
    """
    W = int(camera_cfg.get("width", 640))
    H = int(camera_cfg.get("height", 480))
    pixel_size = float(camera_cfg.get("pixel_size", 0.005))
    eff_px = pixel_size / max(state.get("zoom", 1.0), 1e-6)

    fast = np.array(geometry.get("camera_fast", [1, 0, 0]), dtype=float)
    slow = np.array(geometry.get("camera_slow", [0, 1, 0]), dtype=float)
    beam = np.array(geometry.get("beam_axis",   [0, 0, 1]), dtype=float)

    p_lab  = (col - W / 2.0) * eff_px * fast + (row - H / 2.0) * eff_px * slow
    R      = Goniometer(geometry).set(**state).transform()[:3, :3]
    dtrans = -(R.T @ p_lab)

    t = dict(state)
    t["tx"] = state["tx"] + float(dtrans @ fast)
    t["ty"] = state["ty"] + float(dtrans @ slow)
    t["tz"] = state["tz"] + float(dtrans @ beam)
    return t


# ---------------------------------------------------------------------------
# Runtime scene switching
# ---------------------------------------------------------------------------

_SceneBundle = namedtuple("_SceneBundle",
                          "scene_path scene goniometer templates tscene "
                          "want_templates library_kwargs serving_from warning")
_SceneBundle.__doc__ = """Everything a scene contributes to the server, as one value.

Built entirely off-lock by `_build_bundle` and consumed by `_install_bundle`.
Immutable on purpose: it is constructed once in the fallible half of a switch
and consumed once in the infallible half, so there is never a moment when a
half-built scene exists as server state.
"""


def _want_torch_engine(engine, want_templates):
    """Should this server hold a GPU-resident TorchScene?

    Templates never call the renderer -- they crop pre-rendered frames -- and
    the library builder makes its own TorchScene, so a second one here pins GPU
    memory for nothing and makes "no GPU needed at runtime" untrue.  That is
    true for ANY engine: the guard used to also test `engine == "auto"`, so
    `--engine torch --templates on` built a TorchScene nothing would ever call.
    """
    if want_templates:
        return False
    if engine == "torch":
        return True          # explicit: honoured even without CUDA (CPU device)
    if engine != "auto":
        return False         # 'numpy' -- the reference renderer
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


_DIFF_PHRASES = {
    "format":       lambda d: f"stored as {d['have']}, not {d['want']}",
    "psf":          lambda d: ("built without the objective PSF" if not d["have"]
                               else "built with the objective PSF"),
    "supersample":  lambda d: (f"{d['have']}x supersample, so zoom is capped at "
                               f"{d['have']}x rather than {d['want']}x"),
    "step_deg":     lambda d: f"{d['have']:g}deg rotation steps, not {d['want']:g}",
    "n_cond":       lambda d: f"{d['have']} condenser rays, not {d['want']}",
    "pan_mm":       lambda d: f"{d['have']} mm pan margin, not {d['want']}",
    "jpeg_quality": lambda d: f"JPEG quality {d['have']}, not {d['want']}",
    "axis":         lambda d: f"swept about {d['have']}, not {d['want']}",
}


def describe_differences(differs):
    """One operator-readable sentence naming what a stale library differs in.

    "stale" on its own tells nobody whether to care.  Generated server-side so
    the log line, the tab tooltip and the control-page banner cannot disagree.
    """
    if not differs:
        return None
    parts = [_DIFF_PHRASES.get(k, lambda d, k=k: f"{k} is {d['have']}, not {d['want']}")(v)
             for k, v in sorted(differs.items())]
    return ("this frame library was built with different settings: "
            + "; ".join(parts)
            + ". It is complete and is being served as-is -- rebuild it only if "
              "you need those settings.")


# A switch in one of these states owns the slot; anything else is finished and
# is kept only so the last outcome survives long enough to be read.
_SWITCH_BUSY = ("validating", "building", "installing")


def _switch_idle():
    """A fresh, idle switch-state record.

    One constructor, so /scene's payload has the same shape whether or not a
    switch has ever run.  A client that must branch on a missing key is a client
    that will forget to.
    """
    return {"status": "idle", "scene": None, "build": None, "phase": "",
            "frames_done": 0, "frames_total": 0, "message": "",
            "error": None, "started_at": None, "finished_at": None}


class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass   # suppress default stdout logging

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if path in ("/axis-cgi/mjpg/video.cgi", "/mjpg/video.cgi"):
            self._handle_mjpeg()
        elif path in ("/axis-cgi/jpg/image.cgi", "/jpg/image.cgi"):
            self._handle_snapshot()
        elif path == "/motor":
            self._handle_motor(params)
        elif path == "/move":
            self._handle_move(params)
        elif path == "/recenter":
            self._handle_recenter(params)
        elif path == "/beam":
            self._handle_beam()
        elif path == "/xray":
            self._handle_xray()
        elif path == "/scenes":
            self._send_json(self.server._scenes_json())
        elif path == "/scene":
            self._send_json(self.server._scene_json())
        elif path in ("/", "/index.html"):
            self._handle_index()
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length).decode()
        params = dict(urllib.parse.parse_qsl(body))
        parsed = urllib.parse.urlparse(self.path)
        params.update(dict(urllib.parse.parse_qsl(parsed.query)))
        if parsed.path == "/scene":
            self._handle_scene_post(params)
        elif parsed.path == "/motor":
            self._handle_motor(params)
        elif parsed.path == "/move":
            self._handle_move(params)
        else:
            self.send_error(404)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_snapshot(self):
        jpeg = self.server._get_jpeg()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(jpeg)

    def _handle_mjpeg(self):
        """Pure consumer of the background producer's published frames.

        Waits on _frame_cv for a frame generation newer than the last one
        sent (newest-only — never a backlog), clamps to the fps_limit
        ceiling, and resends the cached frame after ~1 s idle so browsers /
        AXIS clients don't time out.  Socket writes happen outside the lock,
        so a slow client never blocks the producer or other clients.
        """
        srv = self.server
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=myboundary"
        )
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        keepalive = 1.0     # idle resend period (s)
        # Each part is TERMINATED as it is written: payload, then the boundary
        # that closes it.  The obvious framing (boundary first, then payload)
        # leaves the last frame of a motion unterminated until the next send,
        # so a client that renders on the boundary rather than on
        # Content-Length holds the second-to-last frame until the keepalive
        # fires -- the pose appears to stall just short of target and then
        # teleport.  Closing every part immediately removes that class of
        # stutter entirely instead of shortening it.
        self.wfile.write(_MJPEG_BOUNDARY + b"\r\n")
        # Closing the part is still not enough for the strictest consumers,
        # which only finalise a part once the NEXT part's headers arrive -- and
        # those cannot be sent early, because Content-Length is not known until
        # the next frame exists.  So the last frame of a motion is followed by
        # one prompt duplicate, which costs a single extra frame per motion and
        # bounds every consumer's wait at the normal cadence instead of a
        # keepalive.  Measured on one move: length-driven and boundary-driven
        # clients both saw 35 ms, a next-headers-driven client saw 1002 ms.
        flush_delay = srv._frame_interval
        last_gen  = 0
        last_send = 0.0
        fresh     = False   # the last send carried new content
        try:
            while True:
                # Wait for a frame newer than the last one sent (or keepalive).
                with srv._frame_cv:
                    deadline = time.monotonic() + (
                        flush_delay if fresh else keepalive)
                    while srv._frame_gen == last_gen:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            break             # idle: resend the cached frame
                        srv._frame_cv.wait(remaining)
                # Clamp to the fps_limit ceiling.
                delay = last_send + srv._frame_interval - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
                # Send the newest published frame.
                with srv._frame_cv:
                    jpeg = srv._jpeg_cache
                    gen  = srv._frame_gen
                if jpeg is None:
                    continue                  # nothing rendered yet
                frame = (
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n".encode()
                    + b"\r\n"
                    + jpeg
                    + b"\r\n"
                    + _MJPEG_BOUNDARY + b"\r\n"   # closes THIS part at once
                )
                self.wfile.write(frame)
                self.wfile.flush()
                fresh     = gen != last_gen
                last_gen  = gen
                last_send = time.monotonic()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, obj, status=200):
        # Errors from the scene endpoints go out as JSON, not via send_error:
        # send_error writes an HTML page, which the control page's
        # `await r.json()` cannot parse, so a refusal would surface to the
        # operator as a silent catch instead of the reason it was refused.
        body = json.dumps(obj, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _handle_motor(self, params):
        """Instant (non-animated) absolute set — kept for AXIS/back-compat."""
        srv = self.server
        updates = {k: float(v) for k, v in params.items() if k in _MOTOR_KEYS}
        if updates:
            srv._set_pose_instant(updates)
        with srv._gonio_lock:
            state = srv._goniometer.get()
        self._send_json(state)

    def _handle_move(self, params):
        try:
            speed  = float(params.get("speed", 1.0))
            target = self.server._command_move(params, speed)
        except (ValueError, KeyError):
            self.send_error(400)
            return
        self._send_json(target)

    def _handle_recenter(self, params):
        """Recentre on a clicked point.

        Preferred: fx/fy — the click as a fraction (0..1) of the *displayed*
        image; the server scales them by the true camera resolution, so the
        client need not know native dimensions (robust to CSS scaling / the
        unreliable MJPEG <img>.naturalWidth).  px/py (native pixels) also work.
        """
        srv = self.server
        try:
            if "fx" in params:
                # Pass the FRACTION down and let the server scale it under its
                # own lock, so the resolution and the geometry come from the
                # same scene.  Scaling here would read the camera outside any
                # lock, one scene swap away from a mismatch.
                frac = (float(params["fx"]), float(params["fy"]))
                pixel = None
            else:
                frac = None
                pixel = (float(params["px"]), float(params["py"]))
            speed = float(params.get("speed", 1.0))
        except (KeyError, ValueError):
            self.send_error(400)
            return
        target = srv._command_recenter(speed, frac=frac, pixel=pixel)
        self._send_json(target)

    def _handle_index(self):
        try:
            body = self.server._static_bytes("index.html")
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _handle_scene_post(self, params):
        """Switch the served scene.  202 accepted, or a JSON refusal.

        Returns immediately: a library build is minutes to an hour, far longer
        than any browser will hold a request open, so the work runs on a daemon
        thread and progress is read back from GET /scene.
        """
        srv  = self.server
        want = params.get("path", "")
        path = srv._resolve_scene(want)
        if path is None:
            self._send_json({"accepted": False,
                             "error": f"unknown scene: {want!r}"}, 400)
            return
        ok, err, code = srv._begin_switch(path, params.get("build") or None)
        if not ok:
            body = {"accepted": False, "error": err,
                    "switch": srv._switch_state()}
            if code == 503:
                body["cuda"] = False
            elif code == 409 and "already running" not in err:
                # Distinct from a busy slot: the scene is fine, it just has no
                # library yet.  The control page keys off this to open the
                # build dialogue rather than reporting a conflict.
                body["needs_build"] = True
            elif code == 409:
                body["busy"] = True
            self._send_json(body, code)
            return
        self._send_json({"accepted": True, "scene": path,
                         "build": params.get("build") or None,
                         "switch": srv._switch_state()}, 202)

    def _handle_beam(self):
        body = self.server._beam_json().encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_xray(self):
        png = self.server._render_xray_png()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(png)


class CameraServer(ThreadingHTTPServer):
    """
    Threaded (one thread per connection) so a browser's idle preconnect socket
    cannot starve the server, and /motor can be served while an MJPEG stream runs.

    Parameters
    ----------
    scene      : Scene
    host       : str
    port       : int
    n_cond     : int — condenser rays per pixel (1=fast preview, 7=smooth)
    fps_limit  : float — max frame rate for MJPEG stream
    scene_path : str | None — YAML the scene was loaded from. Required to serve
                 from a pre-computed frame library; without it every frame is a
                 live render.
    templates  : bool — serve from the frame library (building it first if it is
                 absent or stale). This is the low-latency path and needs no GPU
                 at runtime.
    """

    def __init__(self, scene, host="0.0.0.0", port=8080,
                 n_cond=7, fps_limit=30.0, engine="auto", jpeg_quality=85,
                 preview_mode=True, compile_preview=True, settle_delay=0.5,
                 scene_path=None, templates=True, library_kwargs=None,
                 scene_dir=None, preview_root=None):
        self._scene_path     = scene_path
        # The RAW --templates intent, kept separately from _want_templates
        # (which folds in "and we have a path"). A server constructed with
        # scene_path=None has _want_templates False even though templates were
        # asked for, so reusing that on a scene switch would silently serve a
        # scene live that has a perfectly good library.
        self._templates_flag = bool(templates)
        self._engine         = engine    # needed to rebuild the engine on a switch
        self._want_templates = self._templates_flag and scene_path is not None
        self._library_kwargs = dict(library_kwargs or {})
        self._templates      = None    # TemplateSource once the library is ready
        self._library_root   = os.path.abspath(
            self._library_kwargs.get("root", _LIB_DEFAULT_ROOT))
        self._preview_root   = os.path.abspath(preview_root or _LIB_PREVIEW_ROOT)
        self._scene_dir      = os.path.abspath(scene_dir or _SCENE_DIR_DEFAULT)
        self._serving_from   = None    # "full" | "preview" | None
        self._scene_warning  = None    # set when a stale library is served

        # Build BEFORE binding the socket. A cold build is tens of minutes; a
        # bound-but-unresponsive port leaves clients waiting in the backlog
        # instead of failing to connect, which reads as a hung server.
        manifest = None
        if self._want_templates:
            from ..library.frame_library import ensure_library
            manifest = ensure_library(scene_path, **self._library_kwargs)
            self._serving_from = "full"

        super().__init__((host, port), _Handler)
        self._scene          = scene
        self._goniometer     = Goniometer(scene.geometry)
        self._n_cond         = n_cond
        self._jpeg_quality   = jpeg_quality
        # preview_mode=True (default): fast approximate frames while a move
        # animates, refining to the exact full-quality frame on settle.
        # False: every served frame is the exact full-quality render.
        self._preview_mode   = bool(preview_mode)
        # compile_preview=True (default): run the PREVIEW hot trace through
        # torch.compile for a fusion win. Effective only on the CUDA engine with
        # preview_mode on, and only after start() warms the first (single-
        # threaded) compilation. Settle frames / /xray / offline stay eager+exact.
        self._compile_preview = bool(compile_preview)
        self._compiled_ok     = False   # flipped True once warmup compiles cleanly
        # "Moving" is what selects the fast preview path. Animated /move sets
        # _anim_active; instant pose sets (/motor -- how AXIS-style consumers
        # such as MxCuBE/EPICS drive the goniometer) instead stamp
        # _last_pose_change, and any render within settle_delay seconds of the
        # last stamp counts as moving. Once the pose goes quiet the producer
        # loop forces one exact full-quality re-render (see _bg_render_loop).
        self._settle_delay      = float(settle_delay)
        self._last_pose_change  = float("-inf")   # -inf: boot renders are exact
        self._last_render_preview = False         # written only by the producer
        self._frame_interval = 1.0 / fps_limit
        # Frame slot: the latest published JPEG plus a generation counter,
        # swapped atomically under _frame_cv.  The background render loop is
        # the single producer (single-flight); MJPEG/snapshot handlers are
        # pure consumers that wait on _frame_cv for a newer generation.
        self._jpeg_cache     = None
        self._cache_dirty    = True
        self._frame_gen      = 0     # bumped on every published frame
        self._render_count   = 0     # diagnostics: total _render_now calls
        self._frame_cv       = threading.Condition()
        self._bg_thread      = None

        # Animation: a daemon thread linearly interpolates the goniometer toward
        # a target pose so issued moves glide instead of teleporting.  _gonio_lock
        # guards the (otherwise unsynchronised) live goniometer; _anim_cv guards
        # the target/generation/running-target-pose handshake.
        #
        # LOCK ORDER -- the whole server obeys this, and inverting it hangs:
        #
        #     _anim_cv  >  _scene_lock  >  _gonio_lock          _frame_cv: LEAF
        #
        # Acquire only left-to-right.  _frame_cv is a leaf in the strict sense:
        # never held while acquiring anything, never acquired while holding
        # anything (note _get_jpeg calls _render_now OUTSIDE its `with`, and
        # _run_animation calls _invalidate OUTSIDE its `with` -- both deliberate).
        #
        # The order is not arbitrary.  _anim_cv must be outermost because
        # _command_move/_command_recenter/_animator_loop pair a read of
        # self._scene (camera_cfg AND geometry) with _target_pose, and a scene
        # swap has to make both new in the same instant -- hampton is 0.0074
        # mm/px against mitegen's 0.001, so a torn read is a 7.4x error in every
        # pan, silently clamped, with no exception and no log line.
        # _gonio_lock must be innermost because _render_now holds _scene_lock
        # across _snapshot_gonio; taking _scene_lock from under _gonio_lock (as
        # a self-locking _servable would have) deadlocks against the renderer.
        # tests/test_server_lock_order.py checks this statically -- it has to be
        # static, because threading.Condition wraps an RLock, so an accidental
        # re-entrant _anim_cv would silently succeed at runtime rather than hang.
        self._gonio_lock  = threading.Lock()
        self._anim_cv     = threading.Condition()
        # Guards every scene-derived attribute: _scene, _scene_path, _templates,
        # _tscene, _compiled_ok, _want_templates, _library_kwargs, _serving_from,
        # _scene_warning, and the BINDING of _goniometer (its contents stay under
        # _gonio_lock).  Re-entrant because _render_now holds it across
        # _render_frame, which calls _snapshot_gonio, which needs it too -- and
        # _snapshot_gonio has other callers that hold nothing.
        self._scene_lock  = threading.RLock()
        # Speed (fraction of maximum) and heading of the move in flight, handed
        # to its replacement when one preempts it -- see _run_animation.
        self._anim_u      = 0.0
        self._anim_delta  = None
        self._anim_target = None             # pending target dict (consumed by animator)
        self._anim_speed  = 1.0
        self._anim_gen    = 0                # bumped on every new command (preempt signal)
        self._anim_active = False           # True while interpolating → preview n_cond
        self._anim_thread = None
        self._target_pose = self._goniometer.get()   # last commanded target (running base)
        self._static_dir  = os.path.join(os.path.dirname(__file__), "static")

        # Scene switching.  _switch_cv guards the state dict below AND is what
        # wait_for_switch blocks on -- a Condition, like _frame_cv and _anim_cv,
        # rather than a flag someone has to poll.  _switch_lock serialises the
        # whole build+install so two switches cannot interleave: each is
        # individually atomic, but without it the last one INSTALLED need not be
        # the last one REQUESTED, and both would pay for a library build.
        self._switch_lock   = threading.Lock()
        self._switch_cv     = threading.Condition()
        self._switch        = _switch_idle()
        self._switch_thread = None
        # Bumped once per successful install, inside the install's critical
        # section.  A render that observes two different values across one frame
        # saw a torn swap; that is what tests/test_scene_switch.py asserts never
        # happens.  It also stops a cancelled animation handing its speed and
        # heading to the first move in a DIFFERENT scene (see _run_animation).
        self._scene_gen     = 0

        # Fast path: build the GPU-resident torch engine once (warm). It renders
        # byte-identically to the numpy reference but ~6-8x faster. engine='auto'
        # uses it when CUDA is available; 'numpy' forces the reference renderer.
        self._tscene = None
        if _want_torch_engine(engine, self._want_templates):
            import torch
            from ..renderer.engine_torch import TorchScene
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self._tscene = TorchScene(scene, dev, torch.float64)

        # Templates (built above, before the socket was bound) serve every
        # frame by cropping/scaling. Building needs the renderer; serving does
        # not, so a GPU is a build-time accelerator, not a runtime requirement.
        if manifest is not None:
            self._templates = TemplateSource(
                manifest, library_dir(scene_path, self._library_root),
                jpeg_quality=jpeg_quality)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _invalidate(self):
        with self._frame_cv:
            self._cache_dirty = True
            self._frame_cv.notify_all()   # wake the producer (and any waiters)

    def _snapshot_gonio(self):
        """A thread-safe, fresh Goniometer at the live pose.

        Rendering reads a consistent pose even while the animator thread mutates
        the live goniometer (get() copies the motor dict under the lock).

        _scene_lock spans BOTH the pose read and the geometry read because they
        are a pair: without it a scene swap landing between them builds the OLD
        pose on the NEW axes, i.e. a frame rendered on a transform that never
        existed.  The lock is re-entrant, so _render_now can hold it across this
        call and the two request-thread callers (/beam, /xray) can take it too.
        """
        with self._scene_lock:
            with self._gonio_lock:
                state = self._goniometer.get()
            return Goniometer(self._scene.geometry).set(**state)

    def _render_frame(self):
        """Produce + JPEG-encode the current pose (no cache bookkeeping)."""
        gono   = self._snapshot_gonio()
        # Templates serve every frame when a library is loaded: the preview /
        # settle split exists to trade quality for speed during motion, and a
        # template crop is already both.
        if self._templates is not None:
            self._last_render_preview = False
            return self._templates.render(gono.get())
        # Fast preview while the pose is moving (an animated /move, or an
        # instant /motor set within the last settle_delay seconds); full
        # quality once it settles.
        moving = self._anim_active or (
            time.monotonic() - self._last_pose_change < self._settle_delay)
        preview = moving and self._preview_mode
        self._last_render_preview = preview
        n_cond = 1 if preview else self._n_cond
        if self._tscene is not None:
            import torch
            from PIL import Image
            from ..renderer.engine_torch import render_torch
            # Compiled ONLY for previews and ONLY once warmup succeeded. Settle
            # frames (and /xray, elsewhere) always take the exact eager path.
            use_compiled = preview and self._compiled_ok
            if use_compiled:
                try:
                    img = render_torch(self._tscene, gono, n_cond=n_cond, compiled=True)
                except Exception as exc:      # once-and-done fallback to eager
                    self._compiled_ok = False
                    print(f"[compile-preview] runtime failure, reverting to eager: {exc}")
                    img = render_torch(self._tscene, gono, n_cond=n_cond, compiled=False)
            else:
                img = render_torch(self._tscene, gono, n_cond=n_cond, compiled=False)
            img8 = (img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            buf = io.BytesIO()
            Image.fromarray(img8, mode="RGB").save(
                buf, format="JPEG", quality=self._jpeg_quality)
            jpeg = buf.getvalue()
        else:
            _, jpeg = microscope_render(self._scene, gono, n_cond=n_cond,
                                        jpeg_quality=self._jpeg_quality)
        return jpeg

    def _render_now(self):
        """Render the current pose and publish it as the next frame generation.

        The dirty flag is claimed (cleared) BEFORE the pose snapshot, so an
        invalidation that lands mid-render leaves it set again and the
        producer loop re-renders the newest pose — a burst of invalidations
        coalesces into at most one extra render, never a queue.
        """
        with self._frame_cv:
            self._cache_dirty = False
        # _scene_lock is taken HERE rather than inside _render_frame: the
        # single-flight tests subclass this server and replace _render_frame
        # wholesale, so a lock in there would be silently bypassed by the very
        # tests that exercise the concurrency.  Holding it across the render
        # also hands _render_frame mutual exclusion for its unlocked reads of
        # _templates / _tscene / _scene / _compiled_ok, with no edits inside it.
        # A switch therefore waits at most one in-flight frame (~70 ms on the
        # shipped template path; see docs/DECISIONS.md).  _frame_cv stays a leaf.
        with self._scene_lock:
            jpeg = self._render_frame()
        with self._frame_cv:
            self._render_count += 1
            self._jpeg_cache    = jpeg
            self._frame_gen    += 1
            self._frame_cv.notify_all()   # wake stream/snapshot consumers
        return jpeg

    def _get_jpeg(self):
        """Serve the cached frame, waiting (bounded) for the producer if stale.

        Consumers never render while the background producer runs.  Before
        start() there is no producer thread, so (and only then) render
        synchronously — tests drive the server that way.
        """
        with self._frame_cv:
            if not self._cache_dirty and self._jpeg_cache is not None:
                return self._jpeg_cache
            if self._bg_thread is not None:
                gen0     = self._frame_gen
                deadline = time.monotonic() + 5.0
                while self._frame_gen == gen0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    self._frame_cv.wait(remaining)
                if self._jpeg_cache is not None:
                    return self._jpeg_cache
        return self._render_now()

    def _beam_json(self):
        """X-ray volumes/dose for the live pose, as a JSON string.

        On the server rather than in the handler so the scene and the pose it is
        measured at come from one _scene_lock hold -- otherwise a switch between
        the two reads reports one sample's volumes at another sample's pose.
        """
        with self._scene_lock:
            return beam_volumes_json(self._scene, self._snapshot_gonio())

    def _render_xray_png(self):
        """Render the X-ray transmission map (radiograph) as a grayscale PNG.

        Bright = transmitted, dark = absorbed.  Uses the GPU-resident engine
        when present, else the numpy reference (slow at full resolution).
        Rendered on demand (not cached) — it's a manual snapshot endpoint.
        """
        from PIL import Image
        # _scene_lock spans the whole render: the pose, the engine and the scene
        # must all belong to one scene, or the radiograph shows one sample
        # registered to another's pose.  This is a full-resolution render, so a
        # scene switch waits behind it -- acceptable for a manual snapshot
        # endpoint that is not part of the stream.
        with self._scene_lock:
            gono = self._snapshot_gonio()
            if self._tscene is not None:
                from ..renderer.engine_torch import render_xray_torch
                T = render_xray_torch(self._tscene, gono).clamp(0, 1).cpu().numpy()
            else:
                from ..renderer.beam import render_xray_numpy
                T = render_xray_numpy(self._scene, gono)
        img8 = (np.clip(T, 0.0, 1.0) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(img8, mode="L").save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Background render thread
    # ------------------------------------------------------------------

    def _bg_render_loop(self):
        """Single-flight producer: wait for an invalidation, render, publish.

        Sole caller of _render_now while serving — MJPEG/snapshot handlers
        only consume published frames, so N clients cost one GPU render per
        dirty state instead of N+1.
        """
        while True:
            with self._frame_cv:
                while not self._cache_dirty:
                    self._frame_cv.wait()
            self._render_now()
            # A /motor-driven preview settles here: once the pose has been
            # quiet for settle_delay, force one exact full-quality re-render.
            # (The animated /move path does its own settle in _run_animation;
            # this covers instant pose sets, which never set _anim_active.)
            if self._last_render_preview and not self._anim_active:
                with self._frame_cv:
                    while not self._cache_dirty:
                        remaining = (self._last_pose_change
                                     + self._settle_delay - time.monotonic())
                        if remaining <= 0.0:
                            self._cache_dirty = True   # exact settle render
                            break
                        self._frame_cv.wait(remaining)

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------

    def _static_bytes(self, name):
        with open(os.path.join(self._static_dir, name), "rb") as f:
            return f.read()

    # ------------------------------------------------------------------
    # Move commands + animator
    # ------------------------------------------------------------------

    def _servable(self, pose):
        """Clamp a commanded pose to what is actually on screen.

        Only the template path has limits -- they are a property of the
        rendered window, not of the goniometer -- so the live path is
        untouched.  Clamping the COMMANDED pose (rather than clamping the crop
        and leaving the pose where the operator put it) is what keeps the
        readout, the target boxes and the picture telling the same story: a
        pose the library cannot show is not silently accepted.

        CALLER MUST HOLD _scene_lock.  This reads self._templates, and it must
        NOT take the lock itself: _set_pose_instant calls it from inside
        _gonio_lock, so a self-locking version would create
        _gonio_lock -> _scene_lock and deadlock against _render_now, which holds
        _scene_lock and then wants _gonio_lock.  Two threads, one /motor during
        one background render, and the whole server hangs with the socket still
        accepting.  Hence the lock discipline lives at the call sites.
        """
        if self._templates is None:
            return pose
        eff, note = servable_pose(
            self._templates.manifest,
            tx=pose.get("tx", 0.0), ty=pose.get("ty", 0.0),
            tz=pose.get("tz", 0.0),
            angle_deg=pose.get(self._templates.manifest["axis"], 0.0),
            zoom=pose.get("zoom", 1.0))
        if note is None:
            return pose
        out = dict(pose)
        out.update(eff)
        return out

    def _set_pose_instant(self, updates):
        """Apply an absolute pose immediately, cancelling any running animation
        (the legacy /motor path).

        _scene_lock is taken OUTSIDE _gonio_lock and is not optional: _servable
        below reads self._templates, and reaching for the scene lock from under
        the goniometer lock would invert the renderer's order and hang the
        server.  See the lock-order note in __init__.
        """
        with self._anim_cv:
            self._anim_target = None        # cancel pending/running animation
            self._anim_gen   += 1
            with self._scene_lock:
                with self._gonio_lock:
                    self._goniometer.set(**updates)
                    self._goniometer.set(**self._servable(self._goniometer.get()))
                    self._target_pose = self._goniometer.get()
        self._anim_active = False
        self._last_pose_change = time.monotonic()
        self._invalidate()

    def _command_move(self, params, speed):
        """Resolve a /move against the running target and animate toward it."""
        with self._anim_cv:
            # Camera and geometry read inside the lock, so a target can never
            # be resolved against one scene's pixel size and another's axes.
            # _scene_lock nested inside _anim_cv (never the other way round)
            # extends that guarantee across a runtime scene switch.
            with self._scene_lock:
                cam = self._scene.camera_cfg
                W = int(cam.get("width", 640))
                H = int(cam.get("height", 480))
                pixel_size = float(cam.get("pixel_size", 0.005))
                target = resolve_target(self._target_pose, params, W, H, pixel_size,
                                        geometry=self._scene.geometry)
                target = self._servable(target)
                self._commit_target_locked(target, speed)
        return target

    def _command_recenter(self, speed, frac=None, pixel=None):
        """Animate so the clicked point moves to the image centre.

        Takes the click either as a FRACTION of the displayed image (`frac`,
        the path the UI uses) or in native pixels (`pixel`).  A fraction is
        scaled here rather than in the handler so the resolution, the geometry
        and the camera config all come from one scene.

        Resolved against the LIVE displayed pose (what the user clicked on),
        not the running command target — so a click maps to the frame on screen.
        """
        with self._anim_cv:
            with self._scene_lock:
                # The pose read is INSIDE the same hold as the camera and the
                # geometry.  Read separately (as it was), a scene switch landing
                # between them resolves an old-scene click against the new
                # scene's camera and commits it -- so the stage jumps to a
                # target that corresponds to nothing the operator clicked on.
                cam = self._scene.camera_cfg
                with self._gonio_lock:
                    state = self._goniometer.get()
                if frac is not None:
                    col = frac[0] * int(cam.get("width", 640))
                    row = frac[1] * int(cam.get("height", 480))
                else:
                    col, row = pixel
                target = recenter_target(col, row, state, self._scene.geometry, cam)
                target = self._servable(target)
                self._commit_target_locked(target, speed)
        return target

    def _commit_target_locked(self, target, speed):
        """Publish a new animation target.  Caller must hold self._anim_cv."""
        self._target_pose = target
        self._anim_target = target
        self._anim_speed  = float(speed)
        self._anim_gen   += 1
        self._anim_cv.notify()

    def _animator_loop(self):
        """Wait for a target, then interpolate the goniometer toward it."""
        while True:
            with self._anim_cv:
                while self._anim_target is None:
                    self._anim_cv.wait()
                target = self._anim_target
                speed  = self._anim_speed
                gen    = self._anim_gen
                # Read the camera WITH the target, under the same lock: the
                # geometry an animation is planned against must belong to the
                # same scene as the target it is moving toward.
                with self._scene_lock:
                    cam    = self._scene.camera_cfg
                    W          = int(cam.get("width", 640))
                    pixel_size = float(cam.get("pixel_size", 0.005))
                    scene_gen  = self._scene_gen
                self._anim_target = None
            self._run_animation(target, speed, gen, W, pixel_size, scene_gen)

    def _run_animation(self, target, speed, gen, W, pixel_size, scene_gen=None):
        with self._gonio_lock:
            start = self._goniometer.get()
        duration = move_duration(start, target, speed, W, pixel_size)

        delta = {k: target[k] - start[k] for k in start}
        # Inherit the speed of the move this one replaced, so a burst of jog
        # clicks is one continuous motion instead of N accelerate-brake cycles.
        # Only when the new move continues the old direction: the sign test is
        # a dot product over mixed units (mm and degrees), which is meaningless
        # as a magnitude but correct as a sign for the same-axis case that
        # matters.  A reversal starts from rest -- it needs the braking anyway.
        u = 0.0
        with self._anim_cv:
            prev_u, prev_delta = self._anim_u, self._anim_delta
            self._anim_u, self._anim_delta = 0.0, None
        if prev_u > 0.0 and prev_delta:
            dot = sum(prev_delta.get(k, 0.0) * delta.get(k, 0.0) for k in delta)
            if dot > 0.0:
                u = prev_u

        self._anim_active = True
        pos = 0.0
        while True:
            frac = 1.0 if duration <= 0 else pos
            pose = {k: start[k] + delta[k] * frac for k in start}
            # The generation check and the pose write are ONE critical section.
            # Split, a preempt landing between them stamps this animation's
            # stale pose on top of whatever the winner just committed -- a
            # newer /move, an instant /motor, or a scene swap that has already
            # installed a different goniometer.
            with self._anim_cv:
                if self._anim_gen != gen:
                    # Preempted: hand the current speed and heading to whoever
                    # won, so the replacement move picks up where this one is
                    # rather than braking to a stop first.
                    #
                    # ...but only within ONE scene.  These two fields are the
                    # sole exception to "a cancelled animation writes nothing",
                    # and they are written AFTER a scene swap has released
                    # _anim_cv -- so without this check a switch-cancelled move
                    # hands its speed and heading to the first jog in the new
                    # scene, which has a different mm-per-pixel and, on a swap,
                    # a stage that was just reset to home.
                    if scene_gen is None or self._scene_gen == scene_gen:
                        self._anim_u, self._anim_delta = u, delta
                    return                    # preempted → touch nothing else
                with self._gonio_lock:
                    self._goniometer.set(**pose)
            self._invalidate()
            if frac >= 1.0:
                break
            time.sleep(ANIM_DT)
            pos, u = velocity_step(pos, u, ANIM_DT, duration)

        # Settle: snap exactly to target and request one full-quality frame.
        # Gen-checked like every other write -- without it a preempt in the
        # window between the loop breaking and this block would still land.
        with self._anim_cv:
            if self._anim_gen != gen:
                return
            self._anim_active = False         # only the current owner clears it
            self._anim_u, self._anim_delta = 0.0, None   # arrived: at rest
            with self._gonio_lock:
                self._goniometer.set(**target)
        self._invalidate()

    # ------------------------------------------------------------------
    # Runtime scene switching
    # ------------------------------------------------------------------

    def _scene_choices(self):
        """Absolute paths of every scene this server will switch to.

        Repo-anchored (see _SCENE_DIR_DEFAULT), so the list does not depend on
        the directory the server was launched from.  Whatever is being served
        is always a member even if it lives elsewhere, so `--scene
        /somewhere/else.yaml` still gets a tab.
        """
        paths = sorted(os.path.abspath(p)
                       for p in glob.glob(os.path.join(self._scene_dir, "*.yaml")))
        if self._scene_path:
            cur = os.path.abspath(self._scene_path)
            if cur not in paths:
                paths.append(cur)
        return paths

    def _resolve_scene(self, want):
        """The enumerated scene matching `want`, or None.

        The enumerated set IS the allowlist, and the value returned is always a
        path this server produced -- the client's string is only ever a lookup
        key and never reaches the filesystem, so path traversal is structurally
        impossible and there is nothing to sanitise.  (Contrast _static_bytes,
        which joins an unsanitised name and is safe only because its one caller
        passes a literal.)  Matching on the stem as well as the absolute path
        lets a caller post back `mitegen_200um` instead of a full path.
        """
        if not want:
            return None
        want_abs  = os.path.abspath(want)
        want_stem = os.path.splitext(os.path.basename(want))[0]
        for p in self._scene_choices():
            if p == want_abs or os.path.splitext(os.path.basename(p))[0] == want_stem:
                return p
        return None

    def _build_kwargs(self, preview=False):
        """Library build parameters this server would ask for.

        `root` is dropped: ensure_library consumes it as a named parameter and
        build_params ignores it, but leaving it in would invite someone to pass
        this dict somewhere that treats every key as a build parameter.
        """
        kw = {k: v for k, v in self._library_kwargs.items()
              if k != "root" and v is not None}
        if preview:
            kw.update(PREVIEW_BUILD)
        return kw

    def _grading_params(self, preview=False):
        """The build parameters a library on disk is judged against.

        `supersample` is dropped unless the operator explicitly asked for one.
        It is the single build parameter documented as PER SCENE rather than as
        policy: it follows each camera's own sampling against the objective's
        Nyquist limit, which is 4 for hampton's 7.4 um pixel and 1 for mitegen's
        1.0 um one (RUNBOOK "Frame libraries"). Grading every scene against one
        server-wide default therefore guarantees a permanent false "stale" on
        whichever scene does not happen to match it -- and it cannot be fixed by
        rebuilding, because the correct value for that scene is the one being
        called stale. Everything else here is global policy and is graded.
        """
        kw = self._build_kwargs(preview=preview)
        params = build_params(**kw)
        if "supersample" not in kw:
            params.pop("supersample", None)     # -> library_diff skips the key
        return params

    def _library_states(self, scene_path):
        """(full_status, full_diff, preview_status, preview_diff) for one scene.

        The two roots are graded against DIFFERENT parameters on purpose: a
        preview library is meant to be coarse, so grading it against the full
        build parameters would report every preview ever built as stale.
        """
        full_kw = self._grading_params()
        prev_kw = self._grading_params(preview=True)
        fdir = library_dir(scene_path, self._library_root)
        pdir = library_dir(scene_path, self._preview_root)
        return (library_status(scene_path, fdir, **full_kw),
                library_diff(scene_path, fdir, **full_kw),
                library_status(scene_path, pdir, **prev_kw),
                library_diff(scene_path, pdir, **prev_kw))

    def _pick_library(self, scene_path):
        """(root, source, status, diff) for the library to SERVE, or None root.

        NEVER builds.  This is the whole reason the switch path does not call
        ensure_library: ensure_library rebuilds whenever is_current is false,
        and `frame_library/mitegen_200um` is false only because its manifest
        predates the `format` and `psf` keys -- so a switch to it would silently
        start a ~1.9 h rebuild of 360 frames that are already on disk and fine.

        A stale-but-complete FULL library beats a current PREVIEW one: the
        preview is a coarse stand-in with a 1x zoom ceiling, and quietly
        preferring it over real templates because a build key drifted would be a
        quality regression nobody asked for.
        """
        fstat, fdiff, pstat, pdiff = self._library_states(scene_path)
        if fstat != "missing":
            return self._library_root, "full", fstat, fdiff
        if pstat != "missing":
            return self._preview_root, "preview", pstat, pdiff
        return None, None, "missing", {}

    def _build_bundle(self, scene_path, build=None, progress=print):
        """Load a scene and everything derived from it, OFF-LOCK, as one value.

        This is the ONLY fallible half of a scene switch.  It writes NOTHING on
        self, so if anything here raises -- a missing file, bad YAML, a CUDA
        OOM, a library build that dies -- the live scene was never touched and
        there is no rollback path to get wrong.  Failure safety is structural,
        not careful.

        `build` is None (serve whatever library exists, never build), "preview"
        (build a coarse one into the preview root) or "full" (build into the
        serving root).  Building is ALWAYS explicit; nothing here ever decides
        to build on the operator's behalf.

        Caller contract: hold NO locks.  A cold build is tens of minutes, so
        this must never run on a request thread.
        """
        from ..scene.scene import load
        from ..library.frame_library import build_library, load_manifest

        scene_path = os.path.abspath(scene_path)
        if not os.path.isfile(scene_path):
            raise FileNotFoundError(scene_path)   # cheap check before slow work

        want_templates = self._templates_flag
        lib_kwargs = dict(self._library_kwargs)
        scene = load(scene_path)                  # parse + mesh build

        # Resolution guard.  Nothing in this server caches W/H -- every use
        # re-reads scene.camera_cfg -- but an MJPEG stream has no way to
        # announce a size change to consumers that sized themselves from the
        # first frame, and multipart/x-mixed-replace gives them nothing to
        # renegotiate with.  Refusing here, on the fallible side where a raise
        # costs nothing, turns an unaudited class of client breakage into one
        # clear error.  Both shipped scenes are 640x480, so this never fires;
        # lifting it is a client audit, not a server change.  pixel_size may
        # differ freely -- that is the point (hampton 0.0074 vs mitegen 0.001).
        live = getattr(self, "_scene", None)
        if live is not None:
            with self._scene_lock:
                lcam = self._scene.camera_cfg
                live_wh = (int(lcam.get("width", 640)), int(lcam.get("height", 480)))
            ncam = scene.camera_cfg
            new_wh = (int(ncam.get("width", 640)), int(ncam.get("height", 480)))
            if new_wh != live_wh:
                raise ValueError(
                    f"{os.path.basename(scene_path)} is "
                    f"{new_wh[0]}x{new_wh[1]}, but the stream is "
                    f"{live_wh[0]}x{live_wh[1]}; runtime switching assumes a "
                    f"constant frame size")

        templates = None
        serving_from = None
        warning = None
        if want_templates:
            # NOTE _library_kwargs is NOT rewritten for a preview build. It is
            # the server's configured FULL-build parameter set, and it is what
            # every library is graded against; replacing it with the preview
            # parameters would make the full root read as stale from then on.
            # A preview is a separate root with fixed parameters, not a change
            # of configuration.
            if build == "preview":
                root, serving_from = self._preview_root, "preview"
                manifest = build_library(scene_path, root=root,
                                         progress=progress,
                                         **self._build_kwargs(preview=True))
            elif build == "full":
                root, serving_from = self._library_root, "full"
                manifest = build_library(scene_path, root=root,
                                         progress=progress,
                                         **self._build_kwargs())
            else:
                root, serving_from, status, diff = self._pick_library(scene_path)
                if root is None:
                    raise ValueError(
                        f"no frame library for "
                        f"{os.path.splitext(os.path.basename(scene_path))[0]}; "
                        f"build one with build=preview or build=full")
                # ensure_library is deliberately NOT used: it rebuilds whenever
                # is_current is false, and mitegen_200um is false only because
                # its manifest predates two build keys -- so this path would
                # silently start a ~1.9 h rebuild of frames already on disk.
                manifest = load_manifest(library_dir(scene_path, root))
                if status == "stale":
                    warning = describe_differences(diff)
            templates = TemplateSource(manifest,
                                       library_dir(scene_path, root),
                                       jpeg_quality=self._jpeg_quality)
            if serving_from == "preview":
                warning = ("serving the coarse PREVIEW library "
                           f"({PREVIEW_BUILD['step_deg']:g}deg steps, "
                           f"{PREVIEW_BUILD['supersample']}x supersample) -- "
                           "build the full library for real templates."
                           + (f" Also: {warning}" if warning else ""))

        # GPU allocation last: it can OOM, and until install the OLD TorchScene
        # is still resident, so peak VRAM is the SUM of the two.  Only reachable
        # with --templates off; the default path holds no GPU state at all.
        tscene = None
        if _want_torch_engine(self._engine, want_templates):
            import torch
            from ..renderer.engine_torch import TorchScene
            dev = (torch.device("cuda") if torch.cuda.is_available()
                   else torch.device("cpu"))
            tscene = TorchScene(scene, dev, torch.float64)

        # Fresh goniometer, at home, bound to the NEW axes.  Rebuilding is not
        # optional: Goniometer captures scene.geometry BY REFERENCE, so a reused
        # one keeps transforming on the old axes forever -- silent, and visible
        # only as subtly wrong rotation.  The home-pose reset comes free with it,
        # and that is the decided semantics: a switch is a new sample on the
        # stage, not the same sample re-skinned.  mm do not carry across a 7.4x
        # pixel-size change anyway.
        return _SceneBundle(scene_path=scene_path, scene=scene,
                            goniometer=Goniometer(scene.geometry),
                            templates=templates, tscene=tscene,
                            want_templates=want_templates,
                            library_kwargs=lib_kwargs,
                            serving_from=serving_from, warning=warning)

    def _install_bundle(self, bundle):
        """Swap the live scene in.  Writes only; provably cannot raise.

        Owns its own locking, so a switch is one call and the discipline cannot
        be forgotten at the call site.  Lock order is the server-wide
        _anim_cv > _scene_lock > _gonio_lock, and _anim_cv is held for the WHOLE
        swap: /move, /recenter and the animator all read the camera and the
        geometry under _anim_cv, so anything less lets a target be resolved
        against one scene's pixel size and another's axes.

        Everything slow or fallible happened in _build_bundle.  The body here is
        ~20 stores and one dict copy -- no I/O, no allocation, and no print
        (which takes the stdout lock and blocks on a full pipe).  The only wait
        is for _scene_lock, i.e. at most one in-flight frame; and because
        _anim_cv is taken first the animator stops calling _invalidate(), so the
        producer goes idle and hands the lock over rather than renewing it.
        """
        outgoing = None
        with self._anim_cv:
            # Cancel first, under the lock the animator checks.  A cancelled
            # animation provably writes nothing (docs/DECISIONS.md 2026-08-06),
            # so there is no thread to join and nothing to quiesce -- this bump
            # IS the quiesce.
            self._anim_target = None
            self._anim_gen   += 1
            self._anim_u      = 0.0      # no speed to inherit across scenes
            self._anim_delta  = None
            self._anim_active = False

            with self._scene_lock:
                # Hold the outgoing objects so their destructors -- including
                # CUDA frees -- run after every lock is released.
                outgoing = (self._scene, self._templates, self._tscene,
                            self._goniometer)

                self._scene_path     = bundle.scene_path
                self._scene          = bundle.scene
                self._templates      = bundle.templates
                self._tscene         = bundle.tscene
                self._want_templates = bundle.want_templates
                self._library_kwargs = bundle.library_kwargs
                self._serving_from   = bundle.serving_from
                self._scene_warning  = bundle.warning
                self._scene_gen     += 1

                # The compiled preview trace was traced against the OLD
                # TorchScene's tensors.  Back to eager: leaving it set means the
                # first preview frame either recompiles inside a worker thread
                # -- the thing _warmup_compiled_preview exists to avoid -- or
                # trips its own exception fallback.  We do NOT re-warm, because
                # warmup must be single-threaded and by now it is not.
                self._compiled_ok    = False

                # Goniometer and command target together: _target_pose is the
                # base every relative /move accumulates from, so a pose the
                # stage is not actually at makes the next jog teleport.
                with self._gonio_lock:
                    self._goniometer  = bundle.goniometer
                    self._target_pose = self._goniometer.get()

                # Boot-like: -inf means "not moving", so the first frame of the
                # new scene is the exact one, not a preview.
                self._last_pose_change    = float("-inf")
                self._last_render_preview = False

        del outgoing        # unlocked: CUDA frees, mesh teardown
        # Outside every lock.  _frame_cv is a leaf, and the producer this wakes
        # immediately wants _scene_lock.  Deliberately NOT a _frame_gen bump:
        # MJPEG consumers hold no scene state, and _jpeg_cache still holds the
        # OLD scene's frame at this instant, so bumping would push every client
        # one duplicate stale part for no new information.
        self._invalidate()

    def switch_scene(self, scene_path, build=None, progress=print):
        """Serve a different scene, without restarting.  Synchronous.

        Either installs the new scene or leaves the old one completely
        untouched -- there is no partial outcome, because everything that can
        fail happens in _build_bundle before anything is written.

        Serialised by _switch_lock across build AND install: two concurrent
        switches would each be atomic, but the last one INSTALLED need not be
        the last one REQUESTED, and both would pay for a library build.
        """
        with self._switch_lock:
            bundle = self._build_bundle(scene_path, build=build,
                                        progress=progress)
            self._install_bundle(bundle)
        if bundle.warning:
            print(f"[scene-switch] {os.path.basename(bundle.scene_path)}: "
                  f"{bundle.warning}")
        if self._compile_preview and bundle.tscene is not None:
            print("[compile-preview] reset by the scene switch -- previews run "
                  "eager until the server is restarted (warmup must be "
                  "single-threaded, and this process no longer is)")
        return bundle.scene_path

    # -- the asynchronous wrapper the HTTP endpoints drive -----------------

    def _begin_switch(self, scene_path, build):
        """Claim the switch slot and start the worker.  -> (ok, error, code).

        Every reason to refuse is decided in ONE critical section together with
        the claim, so two operators clicking a tab in the same 600 ms cannot
        both find the slot idle.  Nothing slow happens here: this runs on an
        HTTP thread, and the point of answering 202 is that a browser is not
        left holding a socket open for the length of a library build.
        """
        if build not in (None, "preview", "full"):
            return False, "build must be 'preview' or 'full'", 400
        # The refusal is about BUILDING, never about serving.  Switching to a
        # library that already exists renders nothing at all, which is exactly
        # what the template path is for on a machine with no GPU.
        if build and not cuda_available():
            return False, CPU_BUILD_REFUSAL, 503
        if build is None and self._want_templates and \
                self._pick_library(scene_path)[0] is None:
            return (False,
                    f"no frame library for "
                    f"{os.path.splitext(os.path.basename(scene_path))[0]}; "
                    f"choose build=preview or build=full", 409)
        with self._switch_cv:
            if self._switch["status"] in _SWITCH_BUSY:
                return (False, "a scene switch is already running: "
                               f"{self._switch['scene']}", 409)
            step  = (PREVIEW_BUILD["step_deg"] if build == "preview"
                     else self._build_kwargs().get("step_deg", 1.0))
            self._switch = _switch_idle()
            self._switch.update(
                status="validating", scene=scene_path, build=build,
                phase="loading scene" if not build else "starting build",
                frames_total=int(round(360.0 / step)) if build else 0,
                started_at=time.time())
            self._switch_cv.notify_all()
        # Assigned BEFORE start(), like _bg_thread: a caller that reads the
        # handle must never see None for a switch that is already running.
        t = threading.Thread(target=self._switch_worker,
                             args=(scene_path, build), daemon=True)
        self._switch_thread = t
        t.start()
        return True, None, 202

    def _switch_worker(self, scene_path, build):
        """Daemon body of a scene switch.  Never raises.

        A daemon thread that dies with a traceback nobody sees is exactly the
        failure this endpoint exists to make visible, so every exception lands
        in the state dict where GET /scene will report it.  An exception here
        means nothing changed and the server is still serving what it was.
        """
        try:
            self.switch_scene(scene_path, build=build,
                              progress=self._switch_progress)
        except Exception as exc:
            print(f"[scene-switch] FAILED {scene_path}: {exc}")
            with self._switch_cv:
                self._switch.update(status="error", error=str(exc),
                                    phase="failed", finished_at=time.time())
                self._switch_cv.notify_all()
            return
        with self._switch_cv:
            self._switch.update(status="ok", phase="serving", error=None,
                                message="", finished_at=time.time())
            self._switch_cv.notify_all()

    _FRAMES_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+frames")

    def _switch_progress(self, line):
        """Capture one line of library-build output into the switch state.

        The builder emits free text through a `progress(str)` callable.  Only
        the `<i>/<n> frames` fragment is parsed, for the percentage; the whole
        line is kept verbatim as `message`, so a line this pattern does not
        recognise still reaches the operator instead of vanishing.
        """
        line = str(line).strip()
        print(f"[scene-switch] {line}")
        m = self._FRAMES_RE.search(line)
        with self._switch_cv:
            if self._switch["status"] not in _SWITCH_BUSY:
                return                     # a late line from a finished worker
            self._switch["message"] = line
            if m:
                self._switch.update(status="building", phase="rendering frames",
                                    frames_done=int(m.group(1)),
                                    frames_total=int(m.group(2)))
            self._switch_cv.notify_all()

    def _switch_state(self):
        """A snapshot of the switch state, plus the fields a UI wants derived.

        Copied under the lock: the worker mutates this dict from another thread,
        and json.dumps iterating it mid-update is a torn read waiting to happen.
        """
        with self._switch_cv:
            s = dict(self._switch)
        done, total = s["frames_done"], s["frames_total"]
        s["percent"] = round(100.0 * done / total, 1) if total else None
        end = s.pop("finished_at") or time.time()
        s["elapsed_s"] = (round(end - s["started_at"], 1)
                          if s["started_at"] else None)
        return s

    def wait_for_switch(self, timeout=None):
        """Block until no switch is running; return the terminal state.

        The synchronisation point for tests and for any caller that wants to
        follow a 202 to its conclusion.  A Condition rather than a polled flag,
        for the same reason _frame_cv is: sleep loops calibrated on one machine
        are how flaky tests get written.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._switch_cv:
            while self._switch["status"] in _SWITCH_BUSY:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0.0:
                    break
                self._switch_cv.wait(remaining)
            return dict(self._switch)

    def _scenes_json(self):
        """Every switchable scene, with the library state of both roots."""
        with self._scene_lock:
            current = os.path.abspath(self._scene_path) if self._scene_path else None
        out = []
        for path in self._scene_choices():
            fstat, fdiff, pstat, pdiff = self._library_states(path)
            _, source, status, diff = self._pick_library(path)
            can_serve = source is not None or not self._want_templates
            warning = describe_differences(diff) if status == "stale" else None
            out.append({
                "path": path,
                "name": os.path.splitext(os.path.basename(path))[0],
                "is_current_scene": path == current,
                "can_serve": can_serve,
                "serving_from": source,
                "warning": warning,
                "library": {"status": fstat, "differs": fdiff},
                "preview": {"status": pstat, "differs": pdiff},
            })
        return {"current": current,
                "root": self._library_root,
                "preview_root": self._preview_root,
                "templates": self._want_templates,
                "can_build": cuda_available(),
                "build_refusal": None if cuda_available() else CPU_BUILD_REFUSAL,
                "scenes": out}

    def _scene_json(self):
        """The scene being served, plus the state of any switch in flight."""
        with self._scene_lock:
            path = os.path.abspath(self._scene_path) if self._scene_path else None
            payload = {"scene": path,
                       "name": (os.path.splitext(os.path.basename(path))[0]
                                if path else None),
                       "templates": self._want_templates,
                       "serving_from": self._serving_from,
                       "warning": self._scene_warning}
        payload["switch"] = self._switch_state()
        return payload

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def goniometer(self):
        return self._goniometer

    def _warmup_compiled_preview(self):
        """Trigger the first (single-threaded) compilation BEFORE any worker
        thread exists.

        Concurrent first-compilation from request threads crashed dynamo in a
        prior experiment, so this must run while the server is still single-
        threaded (start(), pre-threads). Any failure degrades to eager: the
        server still serves, just without the fusion win. No-op unless the CUDA
        engine is live and preview compilation is both requested and reachable
        (preview_mode on).
        """
        # Templates never call the renderer, so compiling it would cost a minute
        # of startup for kernels nothing will run.
        with self._scene_lock:
            if self._templates is not None:
                return
            if not (self._compile_preview and self._preview_mode
                    and self._tscene is not None
                    and self._tscene.dev.type == "cuda"):
                return
            try:
                import torch
                from ..renderer.engine_torch import render_torch
                gono = self._snapshot_gonio()
                for _ in range(2):     # preview-shaped: n_cond=1, compiled
                    render_torch(self._tscene, gono, n_cond=1, compiled=True)
                torch.cuda.synchronize()
                self._compiled_ok = True
                print("[compile-preview] warmup ok — preview frames use torch.compile")
            except Exception as exc:
                self._compiled_ok = False
                print(f"[compile-preview] warmup failed, using eager preview: {exc}")

    def start(self, background=False):
        """
        Start serving.

        background=True → runs in a daemon thread and returns immediately.
        background=False → blocks (use Ctrl-C to stop).
        """
        with self._scene_lock:
            templates, warning = self._templates, self._scene_warning
        if warning:
            print(f"[templates] {warning}")
        if templates is not None:
            from ..library.frame_library import zoom_limits
            man = templates.manifest
            rnd = man["rendered"]
            zmin, zmax = zoom_limits(man)
            print(f"[templates] serving from {len(man['frames'])} pre-rendered "
                  f"frames at {rnd['width']}x{rnd['height']} "
                  f"({man['supersample']}x), {man['step_deg']}deg steps about "
                  f"{man['axis']}; zoom {zmin:.2f}-{zmax:.0f}x, "
                  f"X/Y/Z pan within the rendered window, no GPU needed")
            print(f"[templates] roty/rotz are NOT served from templates "
                  f"(one sweep covers one axis) -- use --templates off for those")

        # Single-threaded first compilation of the preview path (if enabled)
        # BEFORE any thread is spawned — concurrent first-compile crashes dynamo.
        self._warmup_compiled_preview()

        # Kick off background render + animator threads
        # Assign _bg_thread BEFORE starting the thread: this closes the startup
        # window where a concurrent _get_jpeg could see _bg_thread is None and
        # render synchronously instead of deferring to the owner.
        t = threading.Thread(target=self._bg_render_loop, daemon=True)
        self._bg_thread = t
        t.start()

        at = threading.Thread(target=self._animator_loop, daemon=True)
        at.start()
        self._anim_thread = at

        host, port = self.server_address
        print(f"loop-sim camera server on http://{host}:{port}")
        print(f"  UI    : http://{host}:{port}/")
        print(f"  MJPEG : http://{host}:{port}/axis-cgi/mjpg/video.cgi")
        print(f"  Snap  : http://{host}:{port}/axis-cgi/jpg/image.cgi")
        print(f"  Motor : http://{host}:{port}/motor?tx=0.1&rotz=45&zoom=2")
        print(f"  Move  : http://{host}:{port}/move?drotx=90&speed=1  (animated)")
        print(f"  Beam  : http://{host}:{port}/beam")
        print(f"  Xray  : http://{host}:{port}/xray")

        if background:
            st = threading.Thread(target=self.serve_forever, daemon=True)
            st.start()
        else:
            try:
                self.serve_forever()
            except KeyboardInterrupt:
                print("\nStopping server.")
                self.server_close()


def library_kwargs_from_args(args):
    """Map CLI options to frame-library build parameters.

    Only options that genuinely describe the STORED templates belong here. In
    particular `--jpeg-quality` does NOT: it is the quality of the JPEG this
    server sends, and forwarding it as the library's `quality` (a different
    default) made every no-flag launch disagree with the shipped library and
    silently kick off a full rebuild. `--template-quality` is the knob for the
    stored templates.

    Kept separate from main() so the mapping is testable without a CLI run --
    the bug above slipped through precisely because the tests constructed
    CameraServer directly and never exercised this path.
    """
    kwargs = {"n_cond": args.n_cond}
    if getattr(args, "supersample", None) is not None:
        kwargs["supersample"] = args.supersample
    if getattr(args, "template_quality", None) is not None:
        kwargs["quality"] = args.template_quality
    if getattr(args, "template_format", None) is not None:
        kwargs["format"] = args.template_format
    if getattr(args, "library_root", None) is not None:
        # Safe to carry here: ensure_library/build_library consume `root` as a
        # named parameter, and build_params ends in **_ignored, so it can never
        # reach the manifest comparison and make every library read as stale.
        kwargs["root"] = args.library_root
    return kwargs


def main(argv=None):
    """CLI entry point: python -m loop_sim.server.camera_server [options]."""
    import argparse

    from ..scene.scene import load

    ap = argparse.ArgumentParser(
        prog="python -m loop_sim.server.camera_server",
        description="AXIS-compatible camera server for the loop simulator.")
    ap.add_argument("--scene", default="scene_files/hampton_300um.yaml",
                    help="scene YAML to serve (default: %(default)s)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--n-cond", type=int, default=7,
                    help="condenser rays for settled frames (default: %(default)s)")
    ap.add_argument("--fps-limit", type=float, default=30.0,
                    help="max MJPEG stream frame rate (default: %(default)s)")
    ap.add_argument("--engine", choices=["auto", "torch", "numpy"], default="auto")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--preview-mode", choices=["on", "off"], default="on",
                    help="on (default): fast approximate frames while a move "
                         "animates, exact frame on settle; off: every served "
                         "frame is the exact full-quality render")
    ap.add_argument("--compile-preview", choices=["on", "off"], default="on",
                    help="on (default): run PREVIEW frames through torch.compile "
                         "(CUDA engine + preview-mode only) for a fusion speedup; "
                         "settle frames stay exact/eager. off: eager previews")
    ap.add_argument("--settle-delay", type=float, default=0.5,
                    help="seconds after the last instant pose set (/motor) "
                         "before rendering the exact full-quality frame "
                         "(default: %(default)s)")
    ap.add_argument("--templates", choices=["on", "off"], default="on",
                    help="on (default): serve from a pre-computed frame "
                         "library, building it first if absent or stale. This "
                         "is the low-latency path and needs no GPU at runtime. "
                         "off: raytrace every frame live")
    ap.add_argument("--supersample", type=int, default=None,
                    help="template sampling factor when building a library "
                         "(default: the library builder's own default)")
    ap.add_argument("--template-format", choices=["png", "jpeg"], default=None,
                    help="stored template format when a library has to be "
                         "built (default: the library builder's own default, "
                         "png). Changing it invalidates an existing library")
    ap.add_argument("--library-root", default=None,
                    help="frame-library root to serve from and report on "
                         "(default: the repo's frame_library/)")
    ap.add_argument("--preview-root", default=None,
                    help="root for the coarse PREVIEW libraries built on "
                         "demand when you switch to a scene that has none "
                         "(default: the repo's frame_library_preview/). Kept "
                         "separate from --library-root deliberately: building "
                         "into the live root would overwrite frames the "
                         "serving TemplateSource is decoding and caching by "
                         "filename")
    ap.add_argument("--scene-dir", default=None,
                    help="directory whose *.yaml are offered for runtime "
                         "switching via /scenes (default: the repo's "
                         "scene_files/)")
    ap.add_argument("--template-quality", type=int, default=None,
                    help="JPEG quality of the STORED templates when a library "
                         "has to be built. Distinct from --jpeg-quality, which "
                         "is the quality of the frames this server sends. "
                         "Changing it invalidates an existing library")
    args = ap.parse_args(argv)

    lib_kwargs = library_kwargs_from_args(args)

    scene = load(args.scene)
    server = CameraServer(scene, host=args.host, port=args.port,
                          n_cond=args.n_cond, fps_limit=args.fps_limit,
                          engine=args.engine, jpeg_quality=args.jpeg_quality,
                          preview_mode=args.preview_mode == "on",
                          compile_preview=args.compile_preview == "on",
                          settle_delay=args.settle_delay,
                          scene_path=args.scene,
                          templates=args.templates == "on",
                          library_kwargs=lib_kwargs,
                          scene_dir=args.scene_dir,
                          preview_root=args.preview_root)
    server.start()


if __name__ == "__main__":
    main()
