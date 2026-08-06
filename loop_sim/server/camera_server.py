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
import io
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ..motors.goniometer import Goniometer
from ..renderer.microscope import render as microscope_render
from ..renderer.beam        import beam_volumes_json
from ..library.frame_library import (DEFAULT_ROOT as _LIB_DEFAULT_ROOT,
                                     frame_for_angle, pose_crop, servable_pose)

_MJPEG_BOUNDARY = b"--myboundary"
_MOTOR_KEYS = ("tx", "ty", "tz", "rotx", "roty", "rotz", "zoom")


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
                  cross_time=2.0, rot_rate=360.0, zoom_rate=4.0,
                  min_time=0.25):
    """Linear-interpolation duration (s) for a move.

    Translations cross the screen *width* in `cross_time` s (scene- and
    zoom-aware via eff_px), rotations spin at `rot_rate` deg/s (360 = 60 rpm),
    zoom changes at `zoom_rate` /s.  The move lasts as long as its slowest
    parameter needs, divided by the speed-dial multiplier (>1 faster).

    `min_time` puts a floor under a *non-zero* move.  A 15 deg phi jog is only
    42 ms at 360 deg/s -- about one frame -- so it arrives as a jump rather
    than a move, and a burst of clicks reads as N separate jumps with a pause
    between each.  With the floor a single click glides, and a click landing
    while the previous is still running preempts and extends it, so a burst
    becomes one continuous rotation.  A zero-distance move stays 0.
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
    longest = max(durs)
    if longest <= 0.0:
        return 0.0
    return max(longest, min_time) / max(speed, 1e-6)


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
        if parsed.path == "/motor":
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

    def _send_json(self, obj):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(200)
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
        cam = srv._scene.camera_cfg
        W = int(cam.get("width", 640))
        H = int(cam.get("height", 480))
        try:
            if "fx" in params:
                col = float(params["fx"]) * W
                row = float(params["fy"]) * H
            else:
                col = float(params["px"])
                row = float(params["py"])
            speed = float(params.get("speed", 1.0))
        except (KeyError, ValueError):
            self.send_error(400)
            return
        target = srv._command_recenter(col, row, speed)
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

    def _handle_beam(self):
        gono = self.server._snapshot_gonio()
        body = beam_volumes_json(self.server._scene, gono).encode()
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
                 scene_path=None, templates=True, library_kwargs=None):
        self._scene_path     = scene_path
        self._want_templates = bool(templates) and scene_path is not None
        self._library_kwargs = dict(library_kwargs or {})
        self._templates      = None    # TemplateSource once the library is ready

        # Build BEFORE binding the socket. A cold build is tens of minutes; a
        # bound-but-unresponsive port leaves clients waiting in the backlog
        # instead of failing to connect, which reads as a hung server.
        manifest = None
        if self._want_templates:
            from ..library.frame_library import ensure_library
            manifest = ensure_library(scene_path, **self._library_kwargs)

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
        self._gonio_lock  = threading.Lock()
        self._anim_cv     = threading.Condition()
        self._anim_target = None             # pending target dict (consumed by animator)
        self._anim_speed  = 1.0
        self._anim_gen    = 0                # bumped on every new command (preempt signal)
        self._anim_active = False           # True while interpolating → preview n_cond
        self._anim_thread = None
        self._target_pose = self._goniometer.get()   # last commanded target (running base)
        self._static_dir  = os.path.join(os.path.dirname(__file__), "static")

        # Fast path: build the GPU-resident torch engine once (warm). It renders
        # byte-identically to the numpy reference but ~6-8x faster. engine='auto'
        # uses it when CUDA is available; 'numpy' forces the reference renderer.
        self._tscene = None
        want_torch = engine == "torch"
        if engine == "auto":
            try:
                import torch
                want_torch = torch.cuda.is_available()
            except Exception:
                want_torch = False
        # Templates never call the renderer, and the library builder makes its
        # own TorchScene -- holding a second one here would pin GPU memory for
        # nothing and make "no GPU needed at runtime" untrue.
        if self._want_templates and engine == "auto":
            want_torch = False
        if want_torch:
            import torch
            from ..renderer.engine_torch import TorchScene
            dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self._tscene = TorchScene(scene, dev, torch.float64)

        # Templates (built above, before the socket was bound) serve every
        # frame by cropping/scaling. Building needs the renderer; serving does
        # not, so a GPU is a build-time accelerator, not a runtime requirement.
        if manifest is not None:
            from ..library.frame_library import library_dir
            self._templates = TemplateSource(
                manifest,
                library_dir(scene_path,
                            self._library_kwargs.get("root", _LIB_DEFAULT_ROOT)),
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
        """
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

    def _render_xray_png(self):
        """Render the X-ray transmission map (radiograph) as a grayscale PNG.

        Bright = transmitted, dark = absorbed.  Uses the GPU-resident engine
        when present, else the numpy reference (slow at full resolution).
        Rendered on demand (not cached) — it's a manual snapshot endpoint.
        """
        from PIL import Image
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
        (the legacy /motor path)."""
        with self._anim_cv:
            self._anim_target = None        # cancel pending/running animation
            self._anim_gen   += 1
            with self._gonio_lock:
                self._goniometer.set(**updates)
                self._goniometer.set(**self._servable(self._goniometer.get()))
                self._target_pose = self._goniometer.get()
        self._anim_active = False
        self._last_pose_change = time.monotonic()
        self._invalidate()

    def _command_move(self, params, speed):
        """Resolve a /move against the running target and animate toward it."""
        cam = self._scene.camera_cfg
        W = int(cam.get("width", 640))
        H = int(cam.get("height", 480))
        pixel_size = float(cam.get("pixel_size", 0.005))
        with self._anim_cv:
            target = resolve_target(self._target_pose, params, W, H, pixel_size,
                                    geometry=self._scene.geometry)
            target = self._servable(target)
            self._commit_target_locked(target, speed)
        return target

    def _command_recenter(self, col, row, speed):
        """Animate so the clicked pixel moves to the image centre.

        Resolved against the LIVE displayed pose (what the user clicked on),
        not the running command target — so a click maps to the frame on screen.
        """
        with self._gonio_lock:
            state = self._goniometer.get()
        with self._anim_cv:
            target = recenter_target(col, row, state,
                                     self._scene.geometry, self._scene.camera_cfg)
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
                self._anim_target = None
            self._run_animation(target, speed, gen)

    def _run_animation(self, target, speed, gen):
        with self._gonio_lock:
            start = self._goniometer.get()
        cam = self._scene.camera_cfg
        W = int(cam.get("width", 640))
        pixel_size = float(cam.get("pixel_size", 0.005))
        duration = move_duration(start, target, speed, W, pixel_size)

        self._anim_active = True
        t0 = time.monotonic()
        while True:
            with self._anim_cv:
                if self._anim_gen != gen:
                    return                    # preempted → outer loop takes over
            frac = 1.0 if duration <= 0 else min(1.0, (time.monotonic() - t0) / duration)
            pose = {k: start[k] + (target[k] - start[k]) * frac for k in start}
            with self._gonio_lock:
                self._goniometer.set(**pose)
            self._invalidate()
            if frac >= 1.0:
                break
            time.sleep(0.02)

        # Settle: snap exactly to target and request one full-quality frame.
        self._anim_active = False
        with self._gonio_lock:
            self._goniometer.set(**target)
        self._invalidate()

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
        if self._templates is not None:
            return
        if not (self._compile_preview and self._preview_mode
                and self._tscene is not None and self._tscene.dev.type == "cuda"):
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
        if self._templates is not None:
            from ..library.frame_library import zoom_limits
            man = self._templates.manifest
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
                          library_kwargs=lib_kwargs)
    server.start()


if __name__ == "__main__":
    main()
