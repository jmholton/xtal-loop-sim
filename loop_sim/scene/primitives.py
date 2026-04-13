"""
Analytic ray-primitive intersection, vectorized over N rays.

Each primitive implements:
    ray_intersect(origins, dirs) -> (t_enter, t_exit, n_enter, n_exit)

where:
    origins  : (N, 3) float64 — ray origins
    dirs     : (N, 3) float64 — unit direction vectors
    t_enter  : (N,)  float64 — entry parameter (np.inf if no hit)
    t_exit   : (N,)  float64 — exit parameter  (np.inf if no hit)
    n_enter  : (N, 3) float64 — outward normal at entry point
    n_exit   : (N, 3) float64 — outward normal at exit point

The outward normal always points away from the primitive interior.
At entry the normal faces the incident ray; at exit it faces away.

A ray hits the primitive over the interval [t_enter, t_exit] when
t_enter < t_exit and t_exit > 0.
"""
import numpy as np

_INF = np.inf


def _no_hit(N):
    inf = np.full(N, _INF)
    zero = np.zeros((N, 3))
    return inf, inf, zero, zero


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------

class Sphere:
    """Sphere centred at `centre` with radius `radius`."""

    def __init__(self, centre=(0., 0., 0.), radius=1.0):
        self.centre = np.asarray(centre, dtype=float)
        self.radius = float(radius)

    def ray_intersect(self, origins, dirs):
        N = len(origins)
        oc = origins - self.centre  # (N, 3)
        a = np.einsum("ij,ij->i", dirs, dirs)           # should be 1 for unit dirs
        b = 2.0 * np.einsum("ij,ij->i", oc, dirs)
        c = np.einsum("ij,ij->i", oc, oc) - self.radius ** 2
        disc = b * b - 4.0 * a * c

        hit = disc >= 0.0
        t_enter = np.full(N, _INF)
        t_exit  = np.full(N, _INF)
        n_enter = np.zeros((N, 3))
        n_exit  = np.zeros((N, 3))

        if np.any(hit):
            sq = np.where(hit, np.sqrt(np.maximum(disc, 0.0)), 0.0)
            a2 = 2.0 * a
            te = (-b - sq) / a2
            tx = (-b + sq) / a2
            t_enter[hit] = te[hit]
            t_exit[hit]  = tx[hit]
            pts_e = origins[hit] + te[hit, None] * dirs[hit]
            pts_x = origins[hit] + tx[hit, None] * dirs[hit]
            n_enter[hit] = (pts_e - self.centre) / self.radius
            n_exit[hit]  = (pts_x - self.centre) / self.radius

        return t_enter, t_exit, n_enter, n_exit


# ---------------------------------------------------------------------------
# HalfSpace  (n · x <= offset  is interior)
# ---------------------------------------------------------------------------

class HalfSpace:
    """
    The interior is the set of points p where dot(normal, p) <= offset.
    normal must be a unit vector.
    """

    def __init__(self, normal=(0., 1., 0.), offset=0.0):
        self.normal = np.asarray(normal, dtype=float)
        self.normal /= np.linalg.norm(self.normal)
        self.offset = float(offset)

    def ray_intersect(self, origins, dirs):
        N = len(origins)
        n = self.normal
        # Ray: p(t) = origin + t * dir
        # dot(n, p(t)) = offset  =>  t = (offset - dot(n, origin)) / dot(n, dir)
        denom = dirs @ n                      # (N,)
        num   = self.offset - origins @ n     # (N,)

        t_enter = np.full(N, _INF)
        t_exit  = np.full(N, _INF)
        n_enter = np.zeros((N, 3))
        n_exit  = np.zeros((N, 3))

        # Interior of the half-space: {p : n · p <= offset}
        # num = offset - n·o  (positive ↔ origin is inside)
        inside_origin = num >= 0.0                  # (N,)

        # Ray parallel to plane
        parallel         = np.abs(denom) < 1e-12
        inside_parallel  = parallel &  inside_origin
        outside_parallel = parallel & ~inside_origin

        t_plane = np.where(~parallel, num / denom, 0.0)

        # Build intervals:
        #   origin inside  + denom > 0 (exiting toward boundary): [-inf, t_plane]
        #   origin inside  + denom < 0 (going deeper inside):     [-inf, +inf]
        #   origin outside + denom < 0 (entering):                [t_plane, +inf]
        #   origin outside + denom > 0 (moving away, no hit):     [+inf, -inf]
        t_e = np.full(N, -_INF)
        t_x = np.full(N,  _INF)

        exiting  = ~parallel &  inside_origin & (denom > 0)   # inside  → exit at t_plane
        entering = ~parallel & ~inside_origin & (denom < 0)   # outside → enter at t_plane
        no_hit   = ~parallel & ~inside_origin & (denom > 0)   # outside, moving away

        t_x[exiting]  = t_plane[exiting]
        t_e[entering] = t_plane[entering]
        t_e[no_hit]   = _INF
        t_x[no_hit]   = -_INF
        t_e[outside_parallel] = _INF
        t_x[outside_parallel] = -_INF

        t_enter[:] = t_e
        t_exit[:]  = t_x

        # Outward surface normal (pointing away from interior = +n direction)
        outward = self.normal
        n_enter[entering]       = outward
        n_enter[inside_parallel] = outward
        n_exit[exiting]         = outward

        return t_enter, t_exit, n_enter, n_exit


# ---------------------------------------------------------------------------
# Cylinder  (infinite along local Z, then capped by two HalfSpaces externally
#            via CSG — or just the infinite barrel here)
# ---------------------------------------------------------------------------

class InfiniteCylinder:
    """
    Infinite cylinder aligned with `axis`, passing through `centre`,
    with the given `radius`.  No end caps — combine with HalfSpaces via CSG.
    """

    def __init__(self, centre=(0., 0., 0.), axis=(0., 1., 0.), radius=1.0):
        self.centre = np.asarray(centre, dtype=float)
        self.axis   = np.asarray(axis,   dtype=float)
        self.axis  /= np.linalg.norm(self.axis)
        self.radius = float(radius)

    def ray_intersect(self, origins, dirs):
        N = len(origins)
        a = self.axis
        oc = origins - self.centre

        # Project out the axial component
        # d_perp = dir - (dir·a) a
        # oc_perp = oc  - (oc·a)  a
        d_a  = dirs   @ a              # (N,)
        oc_a = oc     @ a              # (N,)
        d_perp  = dirs   - d_a[:, None]  * a   # (N, 3)
        oc_perp = oc     - oc_a[:, None] * a   # (N, 3)

        A = np.einsum("ij,ij->i", d_perp, d_perp)
        B = 2.0 * np.einsum("ij,ij->i", oc_perp, d_perp)
        C = np.einsum("ij,ij->i", oc_perp, oc_perp) - self.radius ** 2
        disc = B * B - 4.0 * A * C

        hit = (disc >= 0.0) & (A > 1e-14)
        t_enter = np.full(N, _INF)
        t_exit  = np.full(N, _INF)
        n_enter = np.zeros((N, 3))
        n_exit  = np.zeros((N, 3))

        if np.any(hit):
            sq = np.sqrt(np.maximum(disc[hit], 0.0))
            A2 = 2.0 * A[hit]
            te = (-B[hit] - sq) / A2
            tx = (-B[hit] + sq) / A2
            t_enter[hit] = te
            t_exit[hit]  = tx

            def perp_normal(t_vals, idx):
                pts = origins[idx] + t_vals[:, None] * dirs[idx]
                proj = (pts - self.centre) @ a
                axial = proj[:, None] * a
                radial = (pts - self.centre) - axial
                nrm = radial / (np.linalg.norm(radial, axis=1, keepdims=True) + 1e-30)
                return nrm

            n_enter[hit] = perp_normal(te, hit)
            n_exit[hit]  = perp_normal(tx, hit)

        return t_enter, t_exit, n_enter, n_exit


class Cylinder:
    """
    Finite cylinder: barrel clipped to height `height` centred on `centre`,
    aligned with `axis`, with hemispherical caps (use Capsule instead) or
    flat caps via intersection with two HalfSpaces.

    This version: just the flat-capped finite cylinder via CSG internally.
    """

    def __init__(self, centre=(0., 0., 0.), axis=(0., 1., 0.),
                 radius=1.0, height=2.0):
        self.centre = np.asarray(centre, dtype=float)
        ax = np.asarray(axis, dtype=float)
        self.axis   = ax / np.linalg.norm(ax)
        self.radius = float(radius)
        self.height = float(height)
        half = height / 2.0
        self._barrel = InfiniteCylinder(centre, axis, radius)
        self._cap_lo = HalfSpace( self.axis,  self.axis @ self.centre + half)
        self._cap_hi = HalfSpace(-self.axis, -self.axis @ self.centre + half)

    def ray_intersect(self, origins, dirs):
        from .csg import Intersection
        shape = Intersection(
            self._barrel,
            Intersection(self._cap_lo, self._cap_hi)
        )
        return shape.ray_intersect(origins, dirs)


# ---------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------

class Ellipsoid:
    """
    Ellipsoid aligned with the coordinate axes, centred at `centre`,
    with semi-axes `radii = (a, b, c)`.
    """

    def __init__(self, centre=(0., 0., 0.), radii=(1., 1., 1.)):
        self.centre = np.asarray(centre, dtype=float)
        self.radii  = np.asarray(radii,  dtype=float)

    def ray_intersect(self, origins, dirs):
        N = len(origins)
        r = self.radii
        # Scale space so ellipsoid → unit sphere
        oc = (origins - self.centre) / r   # (N, 3)
        ds = dirs / r                       # (N, 3)

        a = np.einsum("ij,ij->i", ds, ds)
        b = 2.0 * np.einsum("ij,ij->i", oc, ds)
        c = np.einsum("ij,ij->i", oc, oc) - 1.0
        disc = b * b - 4.0 * a * c

        hit = disc >= 0.0
        t_enter = np.full(N, _INF)
        t_exit  = np.full(N, _INF)
        n_enter = np.zeros((N, 3))
        n_exit  = np.zeros((N, 3))

        if np.any(hit):
            sq  = np.sqrt(np.maximum(disc[hit], 0.0))
            a2  = 2.0 * a[hit]
            te  = (-b[hit] - sq) / a2
            tx  = (-b[hit] + sq) / a2
            t_enter[hit] = te
            t_exit[hit]  = tx

            def ellipsoid_normal(t_vals, idx):
                pts = origins[idx] + t_vals[:, None] * dirs[idx]
                # Gradient of (x/a)²+(y/b)²+(z/c)²-1 is 2*(x/a², y/b², z/c²)
                nrm = (pts - self.centre) / (r * r)
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
                return nrm

            n_enter[hit] = ellipsoid_normal(te, hit)
            n_exit[hit]  = ellipsoid_normal(tx, hit)

        return t_enter, t_exit, n_enter, n_exit


# ---------------------------------------------------------------------------
# Box (axis-aligned)
# ---------------------------------------------------------------------------

class Box:
    """Axis-aligned box from `lo` to `hi`."""

    def __init__(self, lo=(-1., -1., -1.), hi=(1., 1., 1.)):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def ray_intersect(self, origins, dirs):
        N = len(origins)
        inv_d = np.where(np.abs(dirs) > 1e-15,
                         1.0 / np.where(np.abs(dirs) > 1e-15, dirs, 1.0),
                         np.sign(dirs) * _INF)

        t1 = (self.lo - origins) * inv_d   # (N, 3)
        t2 = (self.hi - origins) * inv_d

        t_lo = np.minimum(t1, t2)
        t_hi = np.maximum(t1, t2)

        t_enter_val = np.max(t_lo, axis=1)   # (N,)
        t_exit_val  = np.min(t_hi, axis=1)

        hit = t_enter_val <= t_exit_val

        t_enter = np.where(hit, t_enter_val, _INF)
        t_exit  = np.where(hit, t_exit_val,  _INF)

        # Normals: which face was hit?
        n_enter = np.zeros((N, 3))
        n_exit  = np.zeros((N, 3))

        if np.any(hit):
            # Entry normal: the axis whose t_lo component equals t_enter_val
            for axis in range(3):
                mask_e = hit & (np.abs(t_lo[:, axis] - t_enter_val) < 1e-10)
                sgn_e  = np.sign(origins[:, axis] - (self.lo[axis] + self.hi[axis]) / 2.0)
                n_enter[mask_e, axis] = -sgn_e[mask_e]  # points outward

                mask_x = hit & (np.abs(t_hi[:, axis] - t_exit_val) < 1e-10)
                sgn_x  = np.sign(origins[:, axis] - (self.lo[axis] + self.hi[axis]) / 2.0)
                n_exit[mask_x, axis]  =  sgn_x[mask_x]

        return t_enter, t_exit, n_enter, n_exit


# ---------------------------------------------------------------------------
# Capsule  (cylinder with hemispherical end caps, from p0 to p1)
# ---------------------------------------------------------------------------

class Capsule:
    """
    Capsule = cylinder of radius r between points p0 and p1, capped by
    hemispheres of radius r at each end.
    """

    def __init__(self, p0, p1, radius):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.radius = float(radius)
        seg = self.p1 - self.p0
        self._len = np.linalg.norm(seg)
        self._axis = seg / (self._len + 1e-30)

    def ray_intersect(self, origins, dirs):
        N = len(origins)
        # SDF-based closest-point-on-segment approach via quadratic
        # See Inigo Quilez: https://iquilezles.org/articles/intersectors/
        p0 = self.p0
        ax = self._axis
        L  = self._len
        r  = self.radius

        ba = self.p1 - p0           # segment vector
        oa = origins - p0           # (N, 3)

        baba = ba @ ba              # scalar: L²
        bard = (dirs @ ba)          # (N,)
        baoa = oa @ ba              # (N,)
        rdoa = np.einsum("ij,ij->i", dirs, oa)   # (N,)
        oaoa = np.einsum("ij,ij->i", oa,   oa)   # (N,)

        a_ = baba - bard * bard
        b_ = baba * rdoa - baoa * bard
        c_ = baba * oaoa - baoa * baoa - r * r * baba
        h_ = b_ * b_ - a_ * c_

        t_enter = np.full(N, _INF)
        t_exit  = np.full(N, _INF)
        n_enter = np.zeros((N, 3))
        n_exit  = np.zeros((N, 3))

        # --- Infinite cylinder part ---
        cyl_hit = (h_ >= 0.0) & (np.abs(a_) > 1e-12)
        if np.any(cyl_hit):
            sq  = np.sqrt(np.maximum(h_[cyl_hit], 0.0))
            a2  = a_[cyl_hit]

            for sign, t_arr, n_arr in [(-1, t_enter, n_enter), (1, t_exit, n_exit)]:
                tc = (-b_[cyl_hit] + sign * sq) / a2
                # Check that hit is within capsule length
                yc = baoa[cyl_hit] + tc * bard[cyl_hit]
                valid = (yc >= 0.0) & (yc <= baba)
                # Map back to global indices
                global_idx = np.where(cyl_hit)[0][valid]
                tc_valid   = tc[valid]
                t_arr[global_idx] = tc_valid
                pts = origins[global_idx] + tc_valid[:, None] * dirs[global_idx]
                proj = (pts - p0) @ ax
                axial = proj[:, None] * ax
                radial = pts - p0 - axial
                n_arr[global_idx] = radial / (
                    np.linalg.norm(radial, axis=1, keepdims=True) + 1e-30
                )

        # --- Sphere caps ---
        for cap_centre, cap_sign in [(p0, -1), (self.p1, 1)]:
            oc  = origins - cap_centre
            b2  = np.einsum("ij,ij->i", dirs, oc)
            c2  = np.einsum("ij,ij->i", oc,   oc) - r * r
            disc = b2 * b2 - c2
            cap_hit = disc >= 0.0
            if np.any(cap_hit):
                sq2 = np.sqrt(np.maximum(disc[cap_hit], 0.0))
                for sign, t_arr, n_arr in [(-1, t_enter, n_enter), (1, t_exit, n_exit)]:
                    tc = -b2[cap_hit] + sign * sq2
                    # Only use this if on the correct hemisphere
                    pts = origins[cap_hit] + tc[:, None] * dirs[cap_hit]
                    proj = (pts - p0) @ ax
                    on_cap = (cap_sign * (proj - L / 2.0 - L / 2.0) >= 0.0) if cap_sign == 1 \
                             else (proj <= 0.0)
                    # Simpler: cap0 when proj <= 0, cap1 when proj >= L
                    if cap_sign == -1:
                        on_cap = proj <= 0.0
                    else:
                        on_cap = proj >= L
                    global_idx = np.where(cap_hit)[0][on_cap]
                    tc_valid   = tc[on_cap]
                    # Only override if this t is closer
                    update = t_arr[global_idx] > tc_valid
                    t_arr[global_idx[update]] = tc_valid[update]
                    nrm = (pts[on_cap][update] - cap_centre) / r
                    n_arr[global_idx[update]] = nrm

        return t_enter, t_exit, n_enter, n_exit
