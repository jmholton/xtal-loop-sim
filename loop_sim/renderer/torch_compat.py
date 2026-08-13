"""Make `torch._dynamo` reachable before `engine_torch` is imported.

THE FAILURE THIS FIXES, observed on voltron 2026-08-13:

    File "loop_sim/renderer/engine_torch.py", line 467, in TTube
        @torch._dynamo.disable
    AttributeError: module 'torch' has no attribute '_dynamo'

`engine_torch` decorates two methods with `@torch._dynamo.disable`, and a
decorator runs when the class body executes -- i.e. at IMPORT time, before any
code has had a chance to import the submodule.  On torch 2.6 (the dev box) that
works because `torch/__init__.py` binds `_dynamo` itself.  On torch 2.0.1 (the
beamline's `/programs/pytorch/envs/pt`) it does not: `torch._dynamo` exists as a
module but is only bound as an attribute once something does an explicit
`import torch._dynamo`.  So the whole GPU path was unimportable on the
deployment machine while being perfectly healthy in development -- the exact
class of bug a first real deploy exists to find.

WHY THIS IS A SEPARATE FILE, AND NOT A LINE IN `engine_torch.py`

`engine_torch.py` is one of the five sources `frame_library._RENDER_SOURCES`
hashes into `render_sha`, so editing it marks every frame library stale and
re-arms the launch-path rebuild (7.5 h for the droplet scene) -- for a change
that cannot alter a single pixel.  `renderer/` is enumerated file by file rather
than globbed, so a new module here is outside the hash, the same reason
`pin_projection.py` lives here.  `tests/test_frame_library.py` asserts both stay
out.

WHY NOT `loop_sim/__init__.py`, WHICH WOULD BE UNMISSABLE

Because it would drag torch into the torch-free path.  Measured: `import torch`
is 1.63 s, and `from loop_sim.scene.scene import load` currently leaves torch
unimported entirely -- `requirements.txt` omits torch on purpose so the numpy
reference renderer runs without it.  Paying 1.6 s and torch's memory on every
CPU render, to fix a GPU-only import, is the wrong trade.  So this is called
explicitly by the handful of places that are about to import `engine_torch`.
"""


def ensure_dynamo():
    """Bind `torch._dynamo`, or install a no-op stub.  Never raises.

    Call before importing `engine_torch`.  Idempotent and cheap after the first
    call -- Python caches the submodule import.

    The stub path matters on the beamline too: DECISIONS "TITAN V measured"
    records that the pt env's torch 2.0.1 has an Inductor `pkg_resources` bug,
    so `import torch._dynamo` can itself fail rather than merely being absent.
    A no-op is the correct behaviour there, not a degraded one: `disable` exists
    to stop dynamo tracing a function and `maybe_mark_dynamic` to hint a dynamic
    shape, and with no working dynamo there is nothing to stop or hint.  The
    library build runs eager anyway, and `camera_server` already treats a failed
    compile as a fall back to eager.
    """
    try:
        import torch
    except ImportError:
        return False                      # no torch at all: the CPU path
    if hasattr(torch, "_dynamo"):
        return True                       # torch >= 2.1 binds it for us
    try:
        import torch._dynamo              # noqa: F401  (the binding IS the point)
        return True
    except Exception:
        import types
        stub = types.ModuleType("torch._dynamo")

        def disable(fn=None, *_a, **_kw):
            # Usable bare (`@disable`) or called (`@disable()`), matching the
            # real decorator's two spellings.
            return fn if fn is not None else (lambda f: f)

        def maybe_mark_dynamic(*_a, **_kw):
            return None

        stub.disable = disable
        stub.maybe_mark_dynamic = maybe_mark_dynamic
        torch._dynamo = stub
        return False
