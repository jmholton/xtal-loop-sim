"""
Tube primitive: a 3-D curve defined by Neville-interpolated waypoints,
swept to form a tube of constant diameter.

The curve is sampled into a chain of K capsule segments.  Ray intersection
is vectorized over all K capsules simultaneously using (batch × K) numpy
operations rather than a Python loop over capsules, giving large speedups
for scenes with many rays.

Neville's algorithm builds an interpolating polynomial that passes through
every supplied waypoint (unlike Bézier control points).

GPU acceleration
----------------
Pass ``device='cuda'`` to use PyTorch on the GPU for the hot-path
intersection computation.  The external interface is unchanged.
Batch size is automatically increased to 32768 on CUDA (vs 8192 on CPU).
"""
import numpy as np
from scipy.interpolate import CubicSpline

_INF = np.inf
_BATCH_CPU  = 8192
_BATCH_CUDA = 32768


# ---------------------------------------------------------------------------
# Neville polynomial interpolation
# ---------------------------------------------------------------------------

def neville_eval(points, t_nodes, t_query):
    """
    Evaluate a vector-valued interpolating polynomial at scalar `t_query`
    using Neville's algorithm.

    points  : (n, d)  array — data values at each node
    t_nodes : (n,)    array — parameter values at each node (must be distinct)
    t_query : float   — parameter value to evaluate at

    Returns: (d,) array
    """
    n = len(t_nodes)
    P = points.copy().astype(float)
    for k in range(1, n):
        for i in range(n - k):
            j = i + k
            denom = t_nodes[j] - t_nodes[i]
            P[i] = ((t_query - t_nodes[i]) * P[i + 1] -
                    (t_query - t_nodes[j]) * P[i]) / denom
    return P[0]


def neville_sample(waypoints, n_samples):
    """
    Sample an interpolating curve through `waypoints` at `n_samples`
    uniformly-spaced parameter values in [0, 1].

    Uses chord-length parameterization (t proportional to cumulative arc
    length) so the parameter is proportional to physical distance.

    For m ≤ 4 points: global Neville polynomial (degree ≤ 3, no Runge risk).
    For m > 4 points: CubicSpline (C2-smooth, no Runge oscillations).
    Global Neville at degree 5+ causes Runge-phenomenon knots on curved paths.

    waypoints : (m, 3) array
    n_samples : int

    Returns: (n_samples, 3) array of 3-D curve points.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    m = len(waypoints)

    # Chord-length parameterization: t proportional to cumulative arc length.
    chords  = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    cumul   = np.concatenate([[0.0], np.cumsum(chords)])
    total   = cumul[-1]
    t_nodes = cumul / total if total > 1e-30 else np.linspace(0.0, 1.0, m)

    t_query = np.linspace(0.0, 1.0, n_samples)

    if m <= 4:
        return np.stack([neville_eval(waypoints, t_nodes, t) for t in t_query])

    # CubicSpline: C2-smooth, exact through all waypoints, no Runge oscillations.
    cs = CubicSpline(t_nodes, waypoints)
    return cs(t_query)


# ---------------------------------------------------------------------------
# Tube
# ---------------------------------------------------------------------------

class Tube:
    """
    Tube defined by a list of 3-D waypoints and a uniform diameter.

    The curve is interpolated through all waypoints using Neville's algorithm
    and then approximated by a chain of K = (n_samples - 1) capsule segments.
    Ray intersection tests all K capsules simultaneously in one vectorized
    numpy pass (batched over rays).

    Parameters
    ----------
    waypoints : array-like, shape (m, 3)
        3-D control points.  The curve passes through every point.
    diameter : float
        Outer diameter of the tube (mm).
    n_samples : int
        Number of sample points along the interpolated curve (controls
        smoothness vs. speed).  Default: 50.
    device : str
        'cpu' (default, numpy) or 'cuda' (PyTorch GPU).
    """

    def __init__(self, waypoints, diameter, n_samples=50, device='cpu'):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.radius    = float(diameter) / 2.0
        self.n_samples = int(n_samples)
        self._device   = device
        self._build_chain()

    def _build_chain(self):
        pts = neville_sample(self.waypoints, self.n_samples)
        self._curve_pts = pts               # (n_samples, 3)

        # Precompute tangent at each sample point for fiber-axis export
        tangents = np.gradient(pts, axis=0)
        tnorms   = np.linalg.norm(tangents, axis=1, keepdims=True)
        self._tangents = tangents / (tnorms + 1e-30)

        # AABB for fast ray rejection (expanded by radius)
        self._bbox_lo = pts.min(axis=0) - self.radius   # (3,)
        self._bbox_hi = pts.max(axis=0) + self.radius   # (3,)

        # Pre-convert geometry to CUDA tensors for the GPU path
        if self._device != 'cpu':
            import torch
            dev = torch.device(self._device)
            def _t(a):
                return torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)
            ba   = pts[1:] - pts[:-1]              # (K, 3)
            baba = (ba * ba).sum(1)                 # (K,)
            self._p0_t   = _t(pts[:-1])
            self._ba_t   = _t(ba)
            self._baba_t = _t(baba)
            self._ax_t   = _t(ba / (np.sqrt(baba)[:, None] + 1e-30))
            self._pts_t  = _t(pts)

    # ------------------------------------------------------------------
    # Ray intersection — vectorized over (batch, K) capsules
    # ------------------------------------------------------------------

    def ray_intersect(self, origins, dirs):
        """
        Returns (t_enter, t_exit, n_enter, n_exit) for the nearest capsule hit.

        Uses AABB pre-filter to skip rays that cannot hit the tube's bounding
        box, then processes remaining rays in batches of _BATCH.
        """
        N = len(origins)
        t_enter = np.full(N, _INF)
        t_exit  = np.full(N, _INF)
        n_enter = np.zeros((N, 3))
        n_exit  = np.zeros((N, 3))

        # --- AABB slab test: filter rays that miss the bounding box ---
        lo = self._bbox_lo   # (3,)
        hi = self._bbox_hi   # (3,)

        parallel = np.abs(dirs) < 1e-12                                 # (N, 3)
        nonpar   = ~parallel
        safe_inv = np.where(nonpar, 1.0 / np.where(nonpar, dirs, 1.0), 0.0)
        t1 = (lo - origins) * safe_inv     # (N, 3) t at lo-plane
        t2 = (hi - origins) * safe_inv     # (N, 3) t at hi-plane
        t_near = np.where(nonpar, np.minimum(t1, t2), -_INF)           # (N, 3)
        t_far  = np.where(nonpar, np.maximum(t1, t2),  _INF)           # (N, 3)
        par_out = parallel & ((origins < lo) | (origins > hi))
        t_near  = np.where(par_out,  _INF, t_near)
        t_far   = np.where(par_out, -_INF, t_far)

        t_in   = t_near.max(axis=1)                                     # (N,)
        t_out  = t_far.min(axis=1)                                      # (N,)
        aabb   = (t_out > 0.0) & (t_in < t_out)

        if not np.any(aabb):
            return t_enter, t_exit, n_enter, n_exit

        idx = np.where(aabb)[0]
        o_f = origins[idx]
        d_f = dirs[idx]
        M   = len(idx)

        te_f = np.full(M, _INF)
        tx_f = np.full(M, _INF)
        ne_f = np.zeros((M, 3))
        nx_f = np.zeros((M, 3))

        _BATCH = _BATCH_CUDA if self._device != 'cpu' else _BATCH_CPU

        for start in range(0, M, _BATCH):
            end = min(start + _BATCH, M)
            te, tx, ne, nx = self._intersect_batch(o_f[start:end], d_f[start:end])
            te_f[start:end] = te
            tx_f[start:end] = tx
            ne_f[start:end] = ne
            nx_f[start:end] = nx

        t_enter[idx] = te_f
        t_exit[idx]  = tx_f
        n_enter[idx] = ne_f
        n_exit[idx]  = nx_f

        return t_enter, t_exit, n_enter, n_exit

    # ------------------------------------------------------------------
    # GPU batch intersection
    # ------------------------------------------------------------------

    def _intersect_batch_cuda(self, o, d):
        """
        PyTorch/CUDA version of _intersect_batch.
        Accepts and returns numpy arrays.
        """
        import torch
        dev = self._p0_t.device
        ot = torch.from_numpy(np.ascontiguousarray(o)).float().to(dev)   # (B, 3)
        dt = torch.from_numpy(np.ascontiguousarray(d)).float().to(dev)   # (B, 3)

        p0   = self._p0_t    # (K, 3)
        ba   = self._ba_t    # (K, 3)
        baba = self._baba_t  # (K,)
        ax   = self._ax_t    # (K, 3)
        pts  = self._pts_t   # (N_s, 3)
        r    = self.radius
        B    = len(o)
        K    = len(p0)
        N_s  = len(pts)
        INF  = float('inf')

        # oa[b, k] = o[b] - p0[k]  →  (B, K, 3)
        oa = ot.unsqueeze(1) - p0.unsqueeze(0)

        bard = dt @ ba.t()                              # (B, K)
        baoa = (oa * ba.unsqueeze(0)).sum(-1)           # (B, K)
        rdoa = (dt.unsqueeze(1) * oa).sum(-1)           # (B, K)
        oaoa = (oa * oa).sum(-1)                        # (B, K)

        baba_bk = baba.unsqueeze(0)   # (1, K)

        # Cylinder barrel quadratic
        a_ = baba_bk - bard ** 2
        b_ = baba_bk * rdoa - baoa * bard
        c_ = baba_bk * oaoa - baoa ** 2 - r * r * baba_bk
        h_ = b_ * b_ - a_ * c_

        small_a = a_.abs() < 1e-12
        safe_a  = torch.where(small_a, torch.ones_like(a_), a_)
        inv_a   = torch.where(small_a, torch.zeros_like(a_), 1.0 / safe_a)
        v_cyl   = (h_ >= 0.0) & ~small_a

        sq     = torch.where(v_cyl, torch.sqrt(h_.clamp(min=0.0)), torch.zeros_like(h_))
        raw_te = (-b_ - sq) * inv_a
        raw_tx = (-b_ + sq) * inv_a

        # Clip to span: projection ye must lie in [0, baba]
        ye    = baoa + raw_te * bard
        yx    = baoa + raw_tx * bard
        in_seg = baba_bk > 0
        te_k  = torch.where(v_cyl & (ye >= 0.0) & (ye <= baba_bk) & in_seg,
                             raw_te, torch.full_like(raw_te, INF))
        tx_k  = torch.where(v_cyl & (yx >= 0.0) & (yx <= baba_bk) & in_seg,
                             raw_tx, torch.full_like(raw_tx, INF))
        te_k  = torch.where(te_k < tx_k, te_k, torch.full_like(te_k, INF))

        # Sphere caps at every sample point  →  (B, N_s)
        vc    = ot.unsqueeze(1) - pts.unsqueeze(0)           # (B, N_s, 3)
        b_sp  = (vc * dt.unsqueeze(1)).sum(-1)               # (B, N_s)
        c_sp  = (vc * vc).sum(-1) - r * r                    # (B, N_s)
        h_sp  = b_sp * b_sp - c_sp
        v_sp  = h_sp >= 0.0
        sq_sp = torch.where(v_sp, torch.sqrt(h_sp.clamp(min=0.0)), torch.zeros_like(h_sp))
        te_sp = torch.where(v_sp, -b_sp - sq_sp, torch.full_like(b_sp, INF))
        tx_sp = torch.where(v_sp, -b_sp + sq_sp, torch.full_like(b_sp, INF))
        te_sp = torch.where(te_sp < tx_sp, te_sp, torch.full_like(te_sp, INF))

        # Best entry: cylinders (0..K-1) then spheres (K..K+N_s-1)
        te_all = torch.cat([te_k, te_sp], dim=1)    # (B, K+N_s)
        best_k = te_all.argmin(1)                    # (B,)
        bi     = torch.arange(B, device=dev)
        best_te = te_all[bi, best_k]                 # (B,)

        # Best exit — prefer cylinder; fall back to sphere
        tx_cyl_after = torch.where(tx_k  > best_te.unsqueeze(1),
                                   tx_k,  torch.full_like(tx_k,  INF))
        tx_sp_after  = torch.where(tx_sp > best_te.unsqueeze(1),
                                   tx_sp, torch.full_like(tx_sp, INF))
        exit_cyl_k   = tx_cyl_after.argmin(1)
        exit_sp_k    = tx_sp_after.argmin(1)
        best_tx_cyl  = tx_cyl_after[bi, exit_cyl_k]
        best_tx_sp   = tx_sp_after [bi, exit_sp_k]
        use_cyl_exit = best_tx_cyl < INF
        best_tx      = torch.where(use_cyl_exit, best_tx_cyl, best_tx_sp)
        exit_k       = torch.where(use_cyl_exit, exit_cyl_k, exit_sp_k + K)

        # ------------------------------------------------------------------
        # Normals — fully vectorized (compute for all B rays, zero misses)
        # ------------------------------------------------------------------
        hit_mask = best_te < INF    # (B,)

        def _normals_for(t_vals, k_idx):
            """Compute (B, 3) normals given t-values and segment indices."""
            P   = ot + t_vals.unsqueeze(1) * dt         # (B, 3) hit points
            is_cyl = k_idx < K                          # (B,) bool

            # Cylinder normal (safe index into [0, K-1])
            k_c   = k_idx.clamp(0, K - 1)
            p0_k  = p0[k_c]                             # (B, 3)
            ax_k  = ax[k_c]                             # (B, 3)
            proj  = ((P - p0_k) * ax_k).sum(-1, keepdim=True)  # (B, 1)
            C_pt  = p0_k + proj * ax_k                  # (B, 3) closest point on axis
            rad_c = P - C_pt
            n_cyl = rad_c / (rad_c.norm(dim=1, keepdim=True) + 1e-30)

            # Sphere normal (safe index into [0, N_s-1])
            k_s   = (k_idx - K).clamp(0, N_s - 1)
            ctr   = pts[k_s]                            # (B, 3)
            rad_s = P - ctr
            n_sph = rad_s / (rad_s.norm(dim=1, keepdim=True) + 1e-30)

            # Select and zero out misses
            n = torch.where(is_cyl.unsqueeze(1), n_cyl, n_sph)
            n = torch.where(hit_mask.unsqueeze(1), n, torch.zeros_like(n))
            return n

        ne_t = _normals_for(best_te, best_k)
        nx_t = _normals_for(best_tx, exit_k)

        return (best_te.cpu().numpy(), best_tx.cpu().numpy(),
                ne_t.cpu().numpy(),   nx_t.cpu().numpy())

    def _intersect_batch(self, o, d):
        """Intersect a batch of B rays against the swept tube surface.

        Dispatches to CUDA if device is set.

        The tube surface is the union of:
          • K cylinder barrels — one per spline span, clipped to their span.
          • N_s sphere caps   — one at every sample point (fills the elbow gap
            at each curved junction).

        Sphere cap normals (P − center)/r equal the adjacent cylinder barrel
        normals at the boundary circle, so the surface is C0-smooth everywhere.

        Exit is resolved by preferring a cylinder barrel tx (avoids a sphere
        junction exit cutting off a longer cylinder path); sphere tx is the
        fallback for rays that enter and exit purely through an elbow cap.
        """
        if self._device != 'cpu':
            return self._intersect_batch_cuda(o, d)

        B    = len(o)
        r    = self.radius
        pts  = self._curve_pts      # (n_samples, 3)
        p0   = pts[:-1]             # (K, 3)
        ba   = pts[1:] - p0        # (K, 3)
        baba = (ba * ba).sum(1)    # (K,)
        ax   = ba / (np.sqrt(baba)[:, None] + 1e-30)   # (K, 3) unit tangent
        K    = len(p0)
        N_s  = len(pts)             # number of sphere caps

        # oa[b, k] = o[b] - p0[k]  →  (B, K, 3)
        oa   = o[:, None, :] - p0[None, :, :]

        bard = d @ ba.T                           # (B, K)
        baoa = np.einsum("bki,ki->bk", oa, ba)   # (B, K)
        rdoa = np.einsum("bi,bki->bk", d, oa)    # (B, K)
        oaoa = np.einsum("bki,bki->bk", oa, oa)  # (B, K)

        baba_bk = baba[None, :]   # (1, K)

        # Cylinder barrel quadratic
        a_ = baba_bk - bard ** 2
        b_ = baba_bk * rdoa - baoa * bard
        c_ = baba_bk * oaoa - baoa ** 2 - r * r * baba_bk
        h_ = b_ * b_ - a_ * c_

        small_a = np.abs(a_) < 1e-12
        inv_a   = np.where(small_a, 0.0, 1.0 / np.where(small_a, 1.0, a_))
        v_cyl   = (h_ >= 0.0) & ~small_a

        sq     = np.where(v_cyl, np.sqrt(np.maximum(h_, 0.0)), 0.0)
        raw_te = (-b_ - sq) * inv_a
        raw_tx = (-b_ + sq) * inv_a

        # Clip to span: projection ye must lie in [0, baba]
        ye     = baoa + raw_te * bard
        yx     = baoa + raw_tx * bard
        in_seg = baba_bk > 0
        te_k   = np.where(v_cyl & (ye >= 0.0) & (ye <= baba_bk) & in_seg,
                          raw_te, _INF)
        tx_k   = np.where(v_cyl & (yx >= 0.0) & (yx <= baba_bk) & in_seg,
                          raw_tx, _INF)
        te_k   = np.where(te_k < tx_k, te_k, _INF)

        # Sphere caps at every sample point  →  (B, N_s)
        vc    = o[:, None, :] - pts[None, :, :]   # (B, N_s, 3)
        b_sp  = np.einsum('bsi,bi->bs', vc, d)    # (B, N_s)
        c_sp  = (vc * vc).sum(2) - r * r          # (B, N_s)
        h_sp  = b_sp * b_sp - c_sp
        v_sp  = h_sp >= 0.0
        sq_sp = np.where(v_sp, np.sqrt(np.maximum(h_sp, 0.0)), 0.0)
        te_sp = np.where(v_sp, -b_sp - sq_sp, _INF)
        tx_sp = np.where(v_sp, -b_sp + sq_sp, _INF)
        te_sp = np.where(te_sp < tx_sp, te_sp, _INF)

        # Best entry: cylinders (0..K-1) then spheres (K..K+N_s-1)
        te_all  = np.concatenate([te_k, te_sp], axis=1)   # (B, K+N_s)
        best_k  = np.argmin(te_all, axis=1)               # (B,)
        bi      = np.arange(B)
        best_te = te_all[bi, best_k]                       # (B,)

        # Best exit — prefer cylinder; fall back to sphere if no cylinder exits.
        tx_cyl_after = np.where(tx_k  > best_te[:, None], tx_k,  _INF)
        tx_sp_after  = np.where(tx_sp > best_te[:, None], tx_sp, _INF)
        exit_cyl_k   = np.argmin(tx_cyl_after, axis=1)
        exit_sp_k    = np.argmin(tx_sp_after,  axis=1)
        best_tx_cyl  = tx_cyl_after[bi, exit_cyl_k]
        best_tx_sp   = tx_sp_after [bi, exit_sp_k]

        use_cyl_exit = best_tx_cyl < _INF
        best_tx = np.where(use_cyl_exit, best_tx_cyl, best_tx_sp)
        # encode exit index in same te_all space: cylinders 0..K-1, spheres K..
        exit_k  = np.where(use_cyl_exit, exit_cyl_k, exit_sp_k + K)

        # Normals
        hit_mask = best_te < _INF
        ne = np.zeros((B, 3))
        nx = np.zeros((B, 3))

        if np.any(hit_mask):
            hi = np.where(hit_mask)[0]

            for t_vals, k_all, n_arr in [
                    (best_te[hi], best_k[hi],  ne),
                    (best_tx[hi], exit_k[hi],  nx)]:

                valid = hi[t_vals < _INF]
                if len(valid) == 0:
                    continue
                k_v = k_all[t_vals < _INF]
                t_v = t_vals[t_vals < _INF]
                P_v = o[valid] + t_v[:, None] * d[valid]

                is_cyl = k_v < K

                if np.any(is_cyl):
                    idx_c = valid[is_cyl]
                    k_c   = k_v[is_cyl]
                    P_c   = P_v[is_cyl]
                    p0s   = p0[k_c]
                    axs   = ax[k_c]
                    proj  = ((P_c - p0s) * axs).sum(1)
                    C_t   = p0s + proj[:, None] * axs
                    rad   = P_c - C_t
                    n_arr[idx_c] = rad / (
                        np.linalg.norm(rad, axis=1, keepdims=True) + 1e-30)

                if np.any(~is_cyl):
                    idx_s   = valid[~is_cyl]
                    centers = pts[k_v[~is_cyl] - K]
                    rad     = P_v[~is_cyl] - centers
                    n_arr[idx_s] = rad / (
                        np.linalg.norm(rad, axis=1, keepdims=True) + 1e-30)

        return best_te, best_tx, ne, nx

    # ------------------------------------------------------------------
    # Fiber axis helpers (for X-ray beam reporter)
    # ------------------------------------------------------------------

    def tangent_at_fraction(self, frac):
        """Return the unit tangent vector at curve fraction frac ∈ [0, 1]."""
        idx = np.clip(int(frac * (self.n_samples - 1)), 0, self.n_samples - 2)
        return self._tangents[idx]

    def segment_tangents(self):
        """
        Return (n_samples-1, 3) array: one tangent vector per capsule segment,
        pointing along the curve.
        """
        return (self._tangents[:-1] + self._tangents[1:]) / 2.0

    def segment_midpoints(self):
        """Return (n_samples-1, 3) midpoint of each capsule segment."""
        return (self._curve_pts[:-1] + self._curve_pts[1:]) / 2.0
