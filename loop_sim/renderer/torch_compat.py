"""Bind `torch._dynamo` before `engine_torch` is imported, or install a no-op stub.

Works around torch 2.0.1 (the beamline's `/programs/pytorch/envs/pt`), which
does not bind `_dynamo` as a `torch` attribute until something explicitly
imports it, so `engine_torch`'s `@torch._dynamo.disable` decorators raise
`AttributeError` at import time.  Kept out of `engine_torch.py` because that
file is hashed into `render_sha` (`frame_library._RENDER_SOURCES`) and a
change here must not mark every frame library stale.  Call `ensure_dynamo()`
before importing `engine_torch`.
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
