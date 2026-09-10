"""Shared test setup.

`ensure_dynamo()` before anything imports `engine_torch`: on torch 2.0.1 (the
beamline's pt env) `torch._dynamo` is not bound as an attribute, and
`engine_torch` decorates two methods with `@torch._dynamo.disable` at class-body
time, so the import fails outright.  The entry points (render.py, camera_server.py) call it
themselves; this covers the test suite, which imports `engine_torch` directly in
seven files.  Harmless on torch >= 2.1, where it is a `hasattr` check.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.renderer.torch_compat import ensure_dynamo  # noqa: E402

ensure_dynamo()
