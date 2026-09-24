"""
Flag-gated compiled preview path (the 10 fps lever).

The PREVIEW hot trace may run through torch.compile for a fusion win. The
exact-path invariants (settle frames, /xray, offline render, every existing
byte-exact gate) stay bitwise-identical eager -- proven by the pre-existing
settle-parity suite. Compiled PREVIEW frames MAY flip a handful of boundary
pixels; this suite bounds that divergence and proves the gating is real.

CUDA-only (compilation on CPU is forbidden here). Kept cheap: the server-wiring
test spies render_torch so nothing actually compiles.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

from loop_sim.scene.scene import load
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer import engine_torch
from loop_sim.renderer.engine_torch import TorchScene, render_torch
from loop_sim.server.camera_server import CameraServer

HAMPTON = os.path.join(REPO_ROOT, "data", "scene_files", "hampton_300um.yaml")

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="CUDA not available")


def _u8(img):
    """(H, W, 3) uint8, exactly as the server quantises before JPEG."""
    return (img * 255).clamp(0, 255).to(torch.uint8).cpu()


def _encode(img, quality=85, camera=None, sensor=None, phase=0, pin=None):
    """Deliver exactly as the server delivers.

    Routed through the server's own `encode_frame` rather than reimplementing
    the chain, so the camera-emulation stage cannot drift between the two.
    Pass the server's `_camera` and `_sensor`; None/None gives raw
    transmittance on the render's own square-pixel grid.  `pin` is where the
    specular glint goes and comes from the server's `_live_pin` for the same
    reason -- it is pose-dependent, so leaving it out is a second delivery
    implementation.
    """
    from loop_sim.server.camera_server import encode_frame
    return encode_frame(img.detach().cpu().numpy(), quality, camera, sensor,
                        phase, 0.0, pin)


# ---------------------------------------------------------------------------
# (a) bounded divergence + no eager-state contamination
# ---------------------------------------------------------------------------
@cuda_only
@pytest.mark.parametrize("pose", [{}, {"rotx": 45}])
def test_compiled_preview_bounded_divergence(pose):
    scene = load(HAMPTON, device="cpu")
    ts = TorchScene(scene, torch.device("cuda"), torch.float64)
    gono = lambda: Goniometer(scene.geometry).set(**pose)

    eager0 = _u8(render_torch(ts, gono(), n_cond=1, compiled=False))
    comp   = _u8(render_torch(ts, gono(), n_cond=1, compiled=True))
    eager1 = _u8(render_torch(ts, gono(), n_cond=1, compiled=False))

    # Eager path is untouched by compiled use (no cross-contamination of state).
    assert torch.equal(eager0, eager1), "eager render changed after compiled use"

    # Compiled preview may flip only a handful of boundary pixels.
    ndiff = int((eager0 != comp).any(dim=-1).sum().item())
    assert ndiff <= 16, f"compiled preview diverged in {ndiff}/307200 px (> 16)"


# ---------------------------------------------------------------------------
# (b) flag off => preview byte-equal to eager (gating is real)
# ---------------------------------------------------------------------------
@cuda_only
def test_compile_off_preview_is_eager_bytes():
    scene = load(HAMPTON, device="cpu")
    srv = CameraServer(scene, host="127.0.0.1", port=0, engine="auto",
                       compile_preview=False)
    try:
        srv._warmup_compiled_preview()          # no-op when compile_preview off
        assert srv._compiled_ok is False
        srv._anim_active = True                  # preview path
        served = srv._render_now()

        ts = TorchScene(scene, torch.device("cuda"), torch.float64)
        from loop_sim.server.camera_server import pose_phase
        g = Goniometer(scene.geometry).set()
        ref = _encode(render_torch(ts, g, n_cond=1, compiled=False),
                      camera=srv._camera, sensor=srv._sensor,
                      phase=pose_phase(g.get()), pin=srv._live_pin(g))
        assert served == ref
    finally:
        srv.server_close()


# ---------------------------------------------------------------------------
# (c) server wiring: preview uses compiled, settle does not (cheap -- spied)
# ---------------------------------------------------------------------------
@cuda_only
def test_server_routes_preview_compiled_settle_eager(monkeypatch):
    scene = load(HAMPTON, device="cpu")
    srv = CameraServer(scene, host="127.0.0.1", port=0, engine="auto",
                       compile_preview=True)

    seen = []
    real = engine_torch.render_torch

    def spy(tscene, gono, n_cond=1, tile_size=250_000, compiled=False):
        seen.append(compiled)
        # Never actually compile -- delegate to the eager path (cheap).
        return real(tscene, gono, n_cond=n_cond, tile_size=tile_size, compiled=False)

    # _render_frame + warmup both import render_torch from this module by name.
    monkeypatch.setattr(engine_torch, "render_torch", spy)

    try:
        # Manual warmup (cheap: spy short-circuits the real compile) flips the flag.
        srv._warmup_compiled_preview()
        assert srv._compiled_ok is True
        assert seen and all(c is True for c in seen), "warmup must request compiled=True"

        seen.clear()
        srv._anim_active = True          # preview
        srv._render_now()
        assert seen[-1] is True, "animating preview must use the compiled path"

        seen.clear()
        srv._anim_active = False         # settle
        srv._render_now()
        assert seen[-1] is False, "settled frame must stay eager/exact"
    finally:
        srv.server_close()
