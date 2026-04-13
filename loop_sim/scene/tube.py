"""
Tube primitive: a 3-D curve defined by Neville-interpolated waypoints,
swept to form a tube of constant diameter.

The curve is sampled into a chain of K capsule segments.  Ray intersection
is vectorized over all K capsules simultaneously using (batch × K) numpy
operations rather than a Python loop over capsules, giving large speedups
for scenes with many rays.

Neville's algorithm builds an interpolating polynomial that passes through
every supplied waypoint (unlike Bézier control points).
"""
import numpy as np

_INF = np.inf
_BATCH = 8192   # rays processed per iteration in ray_intersect


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

    Uses Neville's algorithm with chord-length parameterization.  Keep
    m ≤ 8 waypoints per tube; a global degree-(m-1) polynomial is smooth
    and exact through all waypoints for small m, but exhibits Runge-phenomenon
    oscillations for m > ~10 with uniform nodes.

    waypoints : (m, 3) array  — m ≤ 8 strongly recommended
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
    return np.stack([neville_eval(waypoints, t_nodes, t) for t in t_query])


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
    """

    def __init__(self, waypoints, diameter, n_samples=50):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.radius    = float(diameter) / 2.0
        self.n_samples = int(n_samples)
        self._build_chain()

    def _build_chain(self):
        pts = neville_sample(self.waypoints, self.n_samples)
        self._curve_pts = pts               # (n_samples, 3)
        K = self.n_samples - 1

        self._p0   = pts[:-1].copy()        # (K, 3) — segment start
        self._p1   = pts[1:].copy()         # (K, 3) — segment end
        self._ba   = self._p1 - self._p0   # (K, 3) — segment vectors
        self._baba = (self._ba * self._ba).sum(axis=1)  # (K,) = |ba|²
        norms      = np.sqrt(self._baba)
        self._ax   = self._ba / (norms[:, None] + 1e-30)        # (K, 3) unit tangent

        # Precompute tangent at each sample point for fiber-axis export
        tangents = np.gradient(pts, axis=0)
        tnorms   = np.linalg.norm(tangents, axis=1, keepdims=True)
        self._tangents = tangents / (tnorms + 1e-30)

        # AABB for fast ray rejection (expanded by radius)
        self._bbox_lo = pts.min(axis=0) - self.radius   # (3,)
        self._bbox_hi = pts.max(axis=0) + self.radius   # (3,)

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
        # Correctly handles rays parallel to each axis.
        lo = self._bbox_lo   # (3,)
        hi = self._bbox_hi   # (3,)

        parallel = np.abs(dirs) < 1e-12                                 # (N, 3)
        nonpar   = ~parallel
        # Safe inverse for non-parallel axes
        safe_inv = np.where(nonpar, 1.0 / np.where(nonpar, dirs, 1.0), 0.0)
        t1 = (lo - origins) * safe_inv     # (N, 3) t at lo-plane
        t2 = (hi - origins) * safe_inv     # (N, 3) t at hi-plane
        # For parallel axes: slab spans all t if origin is inside, none if outside
        t_near = np.where(nonpar, np.minimum(t1, t2), -_INF)           # (N, 3)
        t_far  = np.where(nonpar, np.maximum(t1, t2),  _INF)           # (N, 3)
        # Parallel rays outside their slab → force rejection
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

    def _intersect_batch(self, o, d):
        """Intersect one batch of B rays against all K capsules."""
        B = len(o)
        K = len(self._p0)
        r = self.radius

        p0   = self._p0    # (K, 3)
        p1   = self._p1    # (K, 3)
        ba   = self._ba    # (K, 3)
        baba = self._baba  # (K,)
        ax   = self._ax    # (K, 3)

        # oa[b, k, :] = o[b] - p0[k]  — (B, K, 3)
        oa = o[:, None, :] - p0[None, :, :]

        # (B, K) dot products — use element-wise ops instead of einsum
        bard = d @ ba.T                           # (B, K)
        baoa = np.einsum("bki,ki->bk", oa, ba)   # (B, K)
        rdoa = np.einsum("bi,bki->bk", d, oa)    # (B, K)
        oaoa = np.einsum("bki,bki->bk", oa, oa)  # (B, K)

        baba_bk = baba[None, :]   # (1, K)

        # ------- Cylinder barrel -------
        a_ = baba_bk - bard ** 2
        b_ = baba_bk * rdoa - baoa * bard
        c_ = baba_bk * oaoa - baoa ** 2 - r * r * baba_bk
        h_ = b_ * b_ - a_ * c_

        small_a = np.abs(a_) < 1e-12
        inv_a   = np.where(small_a, 0.0,
                           1.0 / np.where(small_a, 1.0, a_))
        v_cyl = (h_ >= 0.0) & ~small_a

        sq = np.where(v_cyl, np.sqrt(np.maximum(h_, 0.0)), 0.0)
        # Compute raw t values (finite even for invalid capsules) before
        # setting invalid entries to _INF, to avoid inf*0 = NaN warnings.
        raw_te = (-b_ - sq) * inv_a
        raw_tx = (-b_ + sq) * inv_a
        te_cyl = np.where(v_cyl, raw_te, _INF)
        tx_cyl = np.where(v_cyl, raw_tx, _INF)

        ye = baoa + raw_te * bard
        yx = baoa + raw_tx * bard
        in_seg = (baba_bk > 0)
        te_cyl = np.where(v_cyl & (ye >= 0.0) & (ye <= baba_bk) & in_seg,
                          te_cyl, _INF)
        tx_cyl = np.where(v_cyl & (yx >= 0.0) & (yx <= baba_bk) & in_seg,
                          tx_cyl, _INF)

        # ------- Hemisphere cap at p0 -------
        # oc0 = oa  (ray offset from p0)
        b0   = rdoa
        c0   = oaoa - r * r
        disc0 = b0 * b0 - c0
        v0   = disc0 >= 0.0
        sq0  = np.where(v0, np.sqrt(np.maximum(disc0, 0.0)), 0.0)
        raw_te0 = -b0 - sq0
        raw_tx0 = -b0 + sq0
        te0  = np.where(v0, raw_te0, _INF)
        tx0  = np.where(v0, raw_tx0, _INF)
        yte0 = baoa + raw_te0 * bard
        ytx0 = baoa + raw_tx0 * bard
        te0  = np.where(v0 & (yte0 <= 0.0), te0, _INF)
        tx0  = np.where(v0 & (ytx0 <= 0.0), tx0, _INF)

        # ------- Hemisphere cap at p1 -------
        # oc1 = oa - ba[k]  =  o - p1[k]
        oc1  = oa - ba[None, :, :]          # (B, K, 3)
        b1   = np.einsum("bi,bki->bk", d, oc1)
        c1   = np.einsum("bki,bki->bk", oc1, oc1) - r * r
        disc1 = b1 * b1 - c1
        v1   = disc1 >= 0.0
        sq1  = np.where(v1, np.sqrt(np.maximum(disc1, 0.0)), 0.0)
        raw_te1 = -b1 - sq1
        raw_tx1 = -b1 + sq1
        te1  = np.where(v1, raw_te1, _INF)
        tx1  = np.where(v1, raw_tx1, _INF)
        yte1 = baoa + raw_te1 * bard
        ytx1 = baoa + raw_tx1 * bard
        te1  = np.where(v1 & (yte1 >= baba_bk), te1, _INF)
        tx1  = np.where(v1 & (ytx1 >= baba_bk), tx1, _INF)

        # ------- Combine across surfaces -------
        # Entry: earliest surface hit; exit: earliest valid exit surface
        te_k = np.minimum(np.minimum(te_cyl, te0), te1)   # (B, K)
        tx_k = np.minimum(np.minimum(tx_cyl, tx0), tx1)   # (B, K)

        # Invalidate capsules where te >= tx (no valid intersection interval)
        te_k = np.where(te_k < tx_k, te_k, _INF)

        # ------- Best capsule per ray -------
        best_k  = np.argmin(te_k, axis=1)    # (B,)
        bi      = np.arange(B)
        best_te = te_k[bi, best_k]            # (B,)
        best_tx = tx_k[bi, best_k]            # (B,)

        # ------- Compute normals for hit rays -------
        hit_mask = best_te < _INF
        ne = np.zeros((B, 3))
        nx = np.zeros((B, 3))

        if np.any(hit_mask):
            hi    = np.where(hit_mask)[0]    # local hit indices
            k_hi  = best_k[hi]               # winning capsule index per hit ray

            for which, t_vals, n_arr in [
                    ("enter", best_te[hi], ne),
                    ("exit",  best_tx[hi], nx)]:

                pts = o[hi] + t_vals[:, None] * d[hi]   # (n_hit, 3)

                if which == "enter":
                    t_cyl_h = te_cyl[hi, k_hi]
                    t_c0_h  = te0[hi, k_hi]
                else:
                    t_cyl_h = tx_cyl[hi, k_hi]
                    t_c0_h  = tx0[hi, k_hi]

                from_cyl  = np.abs(t_cyl_h - t_vals) < 1e-8
                from_cap0 = (~from_cyl) & (np.abs(t_c0_h - t_vals) < 1e-8)
                from_cap1 = ~from_cyl & ~from_cap0

                normals = np.zeros((len(hi), 3))

                if np.any(from_cyl):
                    k_fc  = k_hi[from_cyl]
                    p0s   = p0[k_fc]
                    axs   = ax[k_fc]
                    pts_c = pts[from_cyl]
                    proj  = ((pts_c - p0s) * axs).sum(axis=1)

                    # Smooth normal: interpolate spline tangent along capsule
                    # instead of using the constant capsule-axis direction.
                    # This gives continuously-varying normals across junctions.
                    frac      = np.clip(proj / (np.sqrt(baba[k_fc]) + 1e-30),
                                        0.0, 1.0)
                    smooth_ax = (self._tangents[k_fc]     * (1.0 - frac[:, None]) +
                                 self._tangents[k_fc + 1] * frac[:, None])
                    smooth_ax /= (np.linalg.norm(smooth_ax, axis=1, keepdims=True)
                                  + 1e-30)

                    proj2  = ((pts_c - p0s) * smooth_ax).sum(axis=1)
                    radial = pts_c - p0s - proj2[:, None] * smooth_ax
                    normals[from_cyl] = radial / (
                        np.linalg.norm(radial, axis=1, keepdims=True) + 1e-30)

                if np.any(from_cap0):
                    normals[from_cap0] = (pts[from_cap0] - p0[k_hi[from_cap0]]) / r

                if np.any(from_cap1):
                    normals[from_cap1] = (pts[from_cap1] - p1[k_hi[from_cap1]]) / r

                n_arr[hi] = normals

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
