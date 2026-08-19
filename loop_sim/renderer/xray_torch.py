"""
GPU-resident X-ray radiograph tracer.

Split out of engine_torch.py (2026-08-18) so an X-ray-only change never
invalidates the optical frame libraries: `frame_library.py`'s
`_RENDER_SOURCES` hashes `engine_torch.py` whole, and that file's shared
`TorchScene`/`next_interface` machinery is genuinely optical-relevant, but
`render_xray_torch`/`trace_xray` never contributed a single optical template
pixel -- they were just riding along in the same file. This module is
deliberately NOT in `_RENDER_SOURCES`; an eventual X-ray frame library hashes
it under its own, separate `render_sha` instead. See docs/DECISIONS.md
2026-08-18.

`trace_xray` takes a `TorchScene` explicitly (it was a `TorchScene` method
before the split) so it can live here without `engine_torch.py` needing to
import this module back.
"""
import numpy as np
import torch


def trace_xray(tscene, o, d, max_depth=None):
    """Straight-ray X-ray traversal (no refraction, no NA): accumulate
    optical depth ``tau = Σ mu_xray·L`` per ray. Returns (N,) tau; the
    transmission is exp(-tau).

    Mirrors `TorchScene.trace_rays`' `next_interface` stepping + compaction,
    but the ray goes straight through every interface (X-rays are undeviated
    at these indices) and we accumulate mu_xray·segment instead of optical
    Beer-Lambert + Fresnel + Snell.
    """
    from .microscope import MAX_DEPTH
    if max_depth is None:
        max_depth = MAX_DEPTH
    N = o.shape[0]
    dev, dt = o.device, o.dtype
    INF = float("inf")
    tau = torch.zeros(N, device=dev, dtype=dt)

    o = o.clone()
    cur_mat = torch.zeros(N, device=dev, dtype=torch.long)   # 0 = background
    gidx = torch.arange(N, device=dev)                       # active-ray indices

    for _ in range(max_depth):
        if gidx.numel() == 0:
            break
        og, dg = o[gidx], d[gidx]
        t_next, _normals, mat_out = tscene.next_interface(og, dg, t_min=1e-6)
        hit = t_next < INF
        if not bool(hit.any()):
            break
        gh = gidx[hit]
        cur_h = cur_mat[gh]
        # Beer-Lambert optical depth over the segment just travelled
        tau[gh] = tau[gh] + tscene.mat_muxray[cur_h] * t_next[hit]
        # advance straight to the interface; direction unchanged
        o[gh] = og[hit] + t_next[hit].unsqueeze(1) * dg[hit]
        cur_mat[gh] = (mat_out[hit] + 1).clamp(0, tscene.K - 1)
        gidx = gh

    return tau


# ---------------------------------------------------------------------------
# render_xray_torch: per-pixel X-ray transmission map (radiograph), GPU-resident.
# One straight ray per pixel along the beam axis; registered to the optical
# view (same focal-point grid as render_torch). Returns an (H, W) torch tensor
# of transmission T = exp(-Σ mu_xray·L) in [0, 1] (1 = transmitted, 0 = absorbed).
# ---------------------------------------------------------------------------
@torch.inference_mode()
def render_xray_torch(tscene, goniometer, tile_size=250_000):
    from ..motors.goniometer import apply_transform, apply_transform_dirs

    scene = tscene.scene
    dev, dt = tscene.dev, tscene.dt
    cam = scene.camera_cfg
    W = int(cam.get("width", 640))
    H = int(cam.get("height", 480))
    pixel_size = float(cam.get("pixel_size", 0.005))
    eff_px = pixel_size / goniometer.zoom

    g = scene.geometry
    fast = np.array(g.get("camera_fast", [1, 0, 0]), dtype=float)
    slow = np.array(g.get("camera_slow", [0, 1, 0]), dtype=float)
    beam = np.array(g.get("beam_axis", [0, 0, 1]), dtype=float)
    beam /= np.linalg.norm(beam) + 1e-30

    px = (np.arange(W) - W / 2.0) * eff_px
    py = (np.arange(H) - H / 2.0) * eff_px
    gx, gy = np.meshgrid(px, py)
    focal_pts = (gx[:, :, None] * fast + gy[:, :, None] * slow).reshape(-1, 3)
    focal_pts -= 50.0 * beam      # start 50 mm upstream, matching beam.py

    T_inv = goniometer.transform_inv()
    origins_s = apply_transform(T_inv, focal_pts)
    dirs_lab  = np.broadcast_to(beam, focal_pts.shape).copy()
    dirs_s    = apply_transform_dirs(T_inv, dirs_lab)
    dirs_s   /= np.linalg.norm(dirs_s, axis=1, keepdims=True) + 1e-30

    o_t = torch.as_tensor(origins_s, device=dev, dtype=dt)
    d_t = torch.as_tensor(dirs_s, device=dev, dtype=dt)
    M = W * H
    tau = torch.empty(M, device=dev, dtype=dt)
    for s in range(0, M, tile_size):
        e = min(s + tile_size, M)
        tau[s:e] = trace_xray(tscene, o_t[s:e], d_t[s:e])
    return torch.exp(-tau).reshape(H, W)
