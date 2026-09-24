"""Shared test setup.

Calls `ensure_dynamo()` before anything imports `engine_torch` (see
`loop_sim/renderer/torch_compat.py` for why).  The entry points (render.py,
camera_server.py) call it themselves; this covers the test suite, which
imports `engine_torch` directly in seven files.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.renderer.torch_compat import ensure_dynamo  # noqa: E402

ensure_dynamo()
