#!/usr/bin/env python
"""
Phase-gate soak for the flag-gated compiled preview path.

Boots the REAL GPU camera server on an ephemeral port (fps_limit=15, compile
preview on), attaches 2 raw-socket MJPEG consumers, and drives a scripted
CONTINUOUS move pattern (long rotations + big pans) that keeps _anim_active busy
the whole time so every produced frame is a compiled n_cond=1 preview. It reports
client fps, renders/s, median/p90 preview render ms, and any exceptions.

Exit status:
  0  healthy — no crash and median preview render ms <= --budget
  1  crash / exception observed, or median render ms > --budget

Not a pytest test (boots real threads, sockets, and the GPU engine). Run:

    ~/miniconda3/envs/loopsim/bin/python investigation/soak_server.py --seconds 45
"""
import argparse
import os
import socket
import statistics
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_BOUNDARY = b"--myboundary"


def _connect_host(host):
    """A dial-able host: 0.0.0.0 isn't connectable, loop back to localhost."""
    return "127.0.0.1" if host in ("0.0.0.0", "", "::") else host


class MjpegConsumer(threading.Thread):
    """Raw-socket MJPEG reader; counts boundary markers as a frame proxy."""

    def __init__(self, host, port, stop_evt, errors):
        super().__init__(daemon=True)
        self.host, self.port = _connect_host(host), port
        self.stop_evt = stop_evt
        self.errors = errors
        self.frames = 0

    def run(self):
        try:
            s = socket.create_connection((self.host, self.port), timeout=10)
            s.sendall(b"GET /axis-cgi/mjpg/video.cgi HTTP/1.0\r\n\r\n")
            s.settimeout(1.0)
            buf = b""
            while not self.stop_evt.is_set():
                try:
                    data = s.recv(65536)
                except socket.timeout:
                    continue
                if not data:
                    break
                buf += data
                parts = buf.split(_BOUNDARY)
                self.frames += len(parts) - 1
                buf = parts[-1]
            s.close()
        except Exception as exc:                      # noqa: BLE001
            self.errors.append(f"consumer: {exc!r}")


def _http_get(host, port, path, errors):
    try:
        s = socket.create_connection((_connect_host(host), port), timeout=10)
        s.sendall(f"GET {path} HTTP/1.0\r\nHost: x\r\n\r\n".encode())
        s.settimeout(5.0)
        while True:
            try:
                if not s.recv(65536):
                    break
            except socket.timeout:
                break
        s.close()
    except Exception as exc:                          # noqa: BLE001
        errors.append(f"driver GET {path}: {exc!r}")


def _driver(host, port, stop_evt, errors):
    """Keep _anim_active busy with long rotations + big pans, re-issued so the
    animation never settles until we stop."""
    patterns = [
        "/move?drotx=3600&speed=1",       # 10 s rotation
        "/move?drotz=3600&speed=1",
        "/move?panx=8&pany=0&speed=0.5",  # ~32 s pan across 8 screen widths
        "/move?droty=3600&speed=1",
        "/move?panx=-8&pany=4&speed=0.5",
    ]
    i = 0
    while not stop_evt.is_set():
        _http_get(host, port, patterns[i % len(patterns)], errors)
        i += 1
        stop_evt.wait(3.0)                # re-issue (preempt) every 3 s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default=os.path.join(
        REPO_ROOT, "scene_files", "hampton_300um.yaml"))
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--budget", type=float, default=100.0,
                    help="max acceptable median preview render ms (default 100)")
    ap.add_argument("--fps-limit", type=float, default=15.0)
    args = ap.parse_args()

    import torch
    from loop_sim.scene.scene import load
    from loop_sim.server.camera_server import CameraServer

    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable — the compiled preview path needs a GPU.")
        return 1

    scene = load(args.scene, device="cpu")
    srv = CameraServer(scene, host="127.0.0.1", port=0, fps_limit=args.fps_limit,
                       engine="auto", preview_mode=True, compile_preview=True)

    # Instrument _render_frame to time each PREVIEW render end-to-end (the encode
    # tail's D2H copy synchronises the CUDA stream, so wall time is honest).
    preview_ms = []
    errors = []
    _orig_render_frame = srv._render_frame

    def _timed_render_frame():
        was_preview = srv._anim_active and srv._preview_mode
        t0 = time.perf_counter()
        try:
            jpeg = _orig_render_frame()
        except Exception as exc:                      # noqa: BLE001
            errors.append(f"render: {exc!r}")
            raise
        dt = (time.perf_counter() - t0) * 1000.0
        if was_preview:
            preview_ms.append(dt)
        return jpeg
    srv._render_frame = _timed_render_frame

    print(f"soak: booting server (warmup compiles ~30-60 s) on ephemeral port ...")
    t_boot = time.perf_counter()
    srv.start(background=True)                         # warmup (compile) happens here
    host, port = srv.server_address
    print(f"soak: server up on {host}:{port} after {time.perf_counter()-t_boot:.1f}s "
          f"(compiled_ok={srv._compiled_ok})")

    stop_evt = threading.Event()
    consumers = [MjpegConsumer(host, port, stop_evt, errors) for _ in range(2)]
    for c in consumers:
        c.start()
    drv = threading.Thread(target=_driver, args=(host, port, stop_evt, errors),
                           daemon=True)
    drv.start()

    rc0 = srv._render_count
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < args.seconds:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    elapsed = time.perf_counter() - t0

    stop_evt.set()
    time.sleep(1.5)                                   # let consumers drain
    renders = srv._render_count - rc0
    client_frames = sum(c.frames for c in consumers)

    srv.shutdown()
    srv.server_close()

    med = statistics.median(preview_ms) if preview_ms else float("nan")
    p90 = (statistics.quantiles(preview_ms, n=10)[8]
           if len(preview_ms) >= 10 else max(preview_ms) if preview_ms else float("nan"))

    print("\n===== soak report =====")
    print(f"duration            : {elapsed:.1f} s")
    print(f"compiled_ok         : {srv._compiled_ok}")
    print(f"preview renders     : {len(preview_ms)}")
    print(f"renders/s (all)     : {renders/elapsed:.2f}")
    print(f"client fps (2 cons) : {client_frames/elapsed:.2f} "
          f"({client_frames} frames total)")
    print(f"preview render ms   : median={med:.1f}  p90={p90:.1f}  "
          f"(budget {args.budget:.0f})")
    print(f"exceptions          : {len(errors)}")
    for e in errors[:10]:
        print(f"    - {e}")

    ok = (not errors) and preview_ms and med <= args.budget
    if not srv._compiled_ok:
        print("WARN: compiled path was disabled (warmup or runtime fallback) — "
              "ran eager; not a crash but the fusion win was not exercised.")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    # Hard-exit with a deterministic code: torch's CUDA/daemon-thread teardown
    # can raise a benign "terminate called" abort during normal interpreter
    # shutdown, which must not masquerade as a phase-gate crash.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
