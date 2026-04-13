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
    Compute and return X-ray beam illuminated volumes as JSON.

Usage
-----
    from loop_sim.server.camera_server import CameraServer
    from loop_sim.scene.scene import load
    from loop_sim.motors.goniometer import Goniometer

    scene = load("scene_files/hampton_300um.yaml")
    server = CameraServer(scene, host="0.0.0.0", port=8080)
    server.start()   # blocks; Ctrl-C to stop
"""
import io
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from ..motors.goniometer import Goniometer
from ..renderer.microscope import render as microscope_render
from ..renderer.beam        import beam_volumes_json

_MJPEG_BOUNDARY = b"--myboundary"


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
        elif path == "/beam":
            self._handle_beam()
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length).decode()
        params = dict(urllib.parse.parse_qsl(body))
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/motor":
            params.update(dict(urllib.parse.parse_qsl(parsed.query)))
            self._handle_motor(params)
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
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=myboundary"
        )
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while True:
                jpeg = self.server._get_jpeg()
                frame = (
                    _MJPEG_BOUNDARY + b"\r\n"
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n".encode()
                    + b"\r\n"
                    + jpeg
                    + b"\r\n"
                )
                self.wfile.write(frame)
                self.wfile.flush()
                time.sleep(self.server._frame_interval)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_motor(self, params):
        motor_keys = {"tx", "ty", "tz", "rotx", "roty", "rotz", "zoom"}
        updates = {k: float(v) for k, v in params.items() if k in motor_keys}
        if updates:
            self.server._goniometer.set(**updates)
            self.server._invalidate()
        state = self.server._goniometer.get()
        body  = json.dumps(state, indent=2).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_beam(self):
        body = beam_volumes_json(self.server._scene,
                                 self.server._goniometer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CameraServer(HTTPServer):
    """
    Parameters
    ----------
    scene      : Scene
    host       : str
    port       : int
    n_cond     : int — condenser rays per pixel (1=fast preview, 7=smooth)
    fps_limit  : float — max frame rate for MJPEG stream
    """

    def __init__(self, scene, host="0.0.0.0", port=8080,
                 n_cond=7, fps_limit=5.0):
        super().__init__((host, port), _Handler)
        self._scene          = scene
        self._goniometer     = Goniometer(scene.geometry)
        self._n_cond         = n_cond
        self._frame_interval = 1.0 / fps_limit
        self._jpeg_cache     = None
        self._cache_dirty    = True
        self._lock           = threading.Lock()
        self._bg_thread      = None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _invalidate(self):
        with self._lock:
            self._cache_dirty = True

    def _render_now(self):
        _, jpeg = microscope_render(self._scene, self._goniometer,
                                    n_cond=self._n_cond)
        with self._lock:
            self._jpeg_cache  = jpeg
            self._cache_dirty = False
        return jpeg

    def _get_jpeg(self):
        with self._lock:
            dirty = self._cache_dirty
            cached = self._jpeg_cache
        if dirty or cached is None:
            return self._render_now()
        return cached

    # ------------------------------------------------------------------
    # Background render thread
    # ------------------------------------------------------------------

    def _bg_render_loop(self):
        """Continuously re-render when the cache is dirty."""
        while True:
            with self._lock:
                dirty = self._cache_dirty
            if dirty:
                self._render_now()
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def goniometer(self):
        return self._goniometer

    def start(self, background=False):
        """
        Start serving.

        background=True → runs in a daemon thread and returns immediately.
        background=False → blocks (use Ctrl-C to stop).
        """
        # Kick off background render thread
        t = threading.Thread(target=self._bg_render_loop, daemon=True)
        t.start()
        self._bg_thread = t

        host, port = self.server_address
        print(f"loop-sim camera server on http://{host}:{port}")
        print(f"  MJPEG : http://{host}:{port}/axis-cgi/mjpg/video.cgi")
        print(f"  Snap  : http://{host}:{port}/axis-cgi/jpg/image.cgi")
        print(f"  Motor : http://{host}:{port}/motor?tx=0.1&rotz=45&zoom=2")
        print(f"  Beam  : http://{host}:{port}/beam")

        if background:
            st = threading.Thread(target=self.serve_forever, daemon=True)
            st.start()
        else:
            try:
                self.serve_forever()
            except KeyboardInterrupt:
                print("\nStopping server.")
                self.server_close()
