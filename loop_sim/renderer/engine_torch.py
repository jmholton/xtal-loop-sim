"""
GPU-resident torch rendering engine (Phase 2).

This module is a torch reimplementation of the numpy intersection/render path,
parametric over device ('cpu'|'cuda') and dtype (float32|float64). The numpy
path (scene/primitives.py, scene/csg.py, renderer/microscope.py) is preserved
verbatim as the canonical ground-truth REFERENCE; this engine is validated
against it by a three-rung differential ladder:

    rung 1  numpy-f64        -- frozen reference / executable spec
    rung 2  torch-cpu-f64    -- must equal rung 1 to ~1e-10 (proves the port; no GPU)
    rung 3  torch-cuda       -- must equal rung 1 within uint8 tolerance (production)

Each torch shape mirrors the corresponding numpy primitive's ray_intersect
contract exactly:
    ray_intersect(o, d) -> (t_enter, t_exit, n_enter, n_exit)   [all torch tensors]
    o, d   : (N, 3) on (device, dtype)
    t_*    : (N,)   +inf where no hit (HalfSpace may return +-inf interval ends)
    n_*    : (N, 3) outward normals; zero where no hit

Built incrementally: 2a = analytic primitives (this file's first cut).
"""
import numpy as np
import torch

_EPS_PARALLEL = 1e-12


def _t(a, dev, dt):
    return torch.as_tensor(np.asarray(a, dtype=float), device=dev, dtype=dt)


def _snell_refract_t(d, n_hat, n1, n2):
    """Torch port of microscope._snell_refract. NaN rows = TIR."""
    flip = (d * n_hat).sum(-1) > 0
    n_hat = torch.where(flip.unsqueeze(1), -n_hat, n_hat)
    cos_i = (-(d * n_hat).sum(-1)).clamp(-1.0, 1.0)
    r = n1 / n2
    sin2_t = r ** 2 * (1.0 - cos_i ** 2)
    tir = sin2_t > 1.0
    cos_t = torch.sqrt(torch.clamp(1.0 - sin2_t, min=0.0))
    d_ref = r.unsqueeze(1) * d + (r * cos_i - cos_t).unsqueeze(1) * n_hat
    d_ref = d_ref / (d_ref.norm(dim=1, keepdim=True) + 1e-30)
    nan = torch.full_like(d_ref, float("nan"))
    return torch.where(tir.unsqueeze(1), nan, d_ref)


def _fresnel_T_t(n1, n2, cos_i):
    """Torch port of microscope._fresnel_T (unpolarised transmittance)."""
    sin2_t = (n1 / n2) ** 2 * (1.0 - cos_i ** 2)
    tir = sin2_t >= 1.0
    cos_t = torch.sqrt(torch.clamp(1.0 - sin2_t, min=0.0))
    rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t + 1e-30)) ** 2
    rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t + 1e-30)) ** 2
    T = 1.0 - (rs + rp) / 2.0
    return torch.where(tir, torch.zeros_like(T), T)


def _aabb_survivors(o, d, lo, hi):
    """Boolean (N,) mask of rays that could hit the AABB [lo, hi] (float64 slab
    test, matching numpy Tube/SurfaceMesh.ray_intersect). Conservative -> exact
    results once the heavy kernel runs on survivors."""
    f64 = torch.float64
    of, df = o.to(f64), d.to(f64)
    INF = float("inf")
    big = df.abs() >= 1e-12
    inv = torch.where(big, 1.0 / torch.where(big, df, torch.ones_like(df)),
                      torch.zeros_like(df))
    t1 = (lo - of) * inv
    t2 = (hi - of) * inv
    t_near = torch.where(big, torch.minimum(t1, t2), torch.full_like(df, -INF))
    t_far = torch.where(big, torch.maximum(t1, t2), torch.full_like(df, INF))
    par_out = (~big) & ((of < lo) | (of > hi))
    t_near = torch.where(par_out, torch.full_like(df, INF), t_near)
    t_far = torch.where(par_out, torch.full_like(df, -INF), t_far)
    t_in = t_near.max(dim=1).values
    t_out = t_far.min(dim=1).values
    return (t_out > 0.0) & (t_in < t_out)


# ---------------------------------------------------------------------------
# Sphere
# ---------------------------------------------------------------------------
class TSphere:
    def __init__(self, centre, radius, dev, dt):
        self.centre = _t(centre, dev, dt)
        self.radius = float(radius)
        self.dev, self.dt = dev, dt

    def ray_intersect(self, o, d, compiled=False):
        oc = o - self.centre
        a = (d * d).sum(-1)
        b = 2.0 * (oc * d).sum(-1)
        c = (oc * oc).sum(-1) - self.radius ** 2
        disc = b * b - 4.0 * a * c
        hit = disc >= 0.0
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        a2 = 2.0 * a
        inf = torch.full_like(a, float("inf"))
        te = torch.where(hit, (-b - sq) / a2, inf)
        tx = torch.where(hit, (-b + sq) / a2, inf)
        zeros = torch.zeros_like(o)
        hit3 = hit.unsqueeze(1)
        ne = torch.where(hit3, (o + te.unsqueeze(1) * d - self.centre) / self.radius, zeros)
        nx = torch.where(hit3, (o + tx.unsqueeze(1) * d - self.centre) / self.radius, zeros)
        return te, tx, ne, nx


# ---------------------------------------------------------------------------
# HalfSpace  (interior: n . x <= offset)
# ---------------------------------------------------------------------------
class THalfSpace:
    def __init__(self, normal, offset, dev, dt):
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        self.normal = _t(n, dev, dt)
        self.offset = float(offset)
        self.dev, self.dt = dev, dt

    def ray_intersect(self, o, d, compiled=False):
        n = self.normal
        denom = (d * n).sum(-1)
        num = self.offset - (o * n).sum(-1)
        inside = num >= 0.0
        parallel = denom.abs() < _EPS_PARALLEL
        nonpar = ~parallel
        safe = torch.where(parallel, torch.ones_like(denom), denom)
        t_plane = torch.where(nonpar, num / safe, torch.zeros_like(denom))

        exiting = nonpar & inside & (denom > 0)
        entering = nonpar & (~inside) & (denom < 0)
        no_hit = nonpar & (~inside) & (denom > 0)
        inside_parallel = parallel & inside
        outside_parallel = parallel & (~inside)

        pos_inf = torch.full_like(denom, float("inf"))
        neg_inf = torch.full_like(denom, float("-inf"))
        t_e = neg_inf.clone()
        t_x = pos_inf.clone()
        t_x = torch.where(exiting, t_plane, t_x)
        t_e = torch.where(entering, t_plane, t_e)
        t_e = torch.where(no_hit, pos_inf, t_e)
        t_x = torch.where(no_hit, neg_inf, t_x)
        t_e = torch.where(outside_parallel, pos_inf, t_e)
        t_x = torch.where(outside_parallel, neg_inf, t_x)

        outward = n.expand_as(o)
        zeros = torch.zeros_like(o)
        ne = torch.where((entering | inside_parallel).unsqueeze(1), outward, zeros)
        nx = torch.where(exiting.unsqueeze(1), outward, zeros)
        return t_e, t_x, ne, nx


# ---------------------------------------------------------------------------
# InfiniteCylinder
# ---------------------------------------------------------------------------
class TInfiniteCylinder:
    def __init__(self, centre, axis, radius, dev, dt):
        ax = np.asarray(axis, dtype=float)
        ax = ax / np.linalg.norm(ax)
        self.centre = _t(centre, dev, dt)
        self.axis = _t(ax, dev, dt)
        self.radius = float(radius)
        self.dev, self.dt = dev, dt

    def ray_intersect(self, o, d, compiled=False):
        a = self.axis
        oc = o - self.centre
        d_a = (d * a).sum(-1, keepdim=True)
        oc_a = (oc * a).sum(-1, keepdim=True)
        d_perp = d - d_a * a
        oc_perp = oc - oc_a * a
        A = (d_perp * d_perp).sum(-1)
        B = 2.0 * (oc_perp * d_perp).sum(-1)
        C = (oc_perp * oc_perp).sum(-1) - self.radius ** 2
        disc = B * B - 4.0 * A * C
        hit = (disc >= 0.0) & (A > 1e-14)
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        A2 = torch.where(hit, 2.0 * A, torch.ones_like(A))
        inf = torch.full_like(A, float("inf"))
        te = torch.where(hit, (-B - sq) / A2, inf)
        tx = torch.where(hit, (-B + sq) / A2, inf)

        zeros = torch.zeros_like(o)
        hit3 = hit.unsqueeze(1)

        def radial_normal(t):
            p = o + t.unsqueeze(1) * d
            rel = p - self.centre
            proj = (rel * a).sum(-1, keepdim=True)
            radial = rel - proj * a
            nrm = radial / (radial.norm(dim=1, keepdim=True) + 1e-30)
            return torch.where(hit3, nrm, zeros)

        return te, tx, radial_normal(te), radial_normal(tx)


# ---------------------------------------------------------------------------
# Ellipsoid
# ---------------------------------------------------------------------------
class TEllipsoid:
    def __init__(self, centre, radii, dev, dt):
        self.centre = _t(centre, dev, dt)
        self.radii = _t(radii, dev, dt)
        self.dev, self.dt = dev, dt

    def ray_intersect(self, o, d, compiled=False):
        r = self.radii
        oc = (o - self.centre) / r
        ds = d / r
        a = (ds * ds).sum(-1)
        b = 2.0 * (oc * ds).sum(-1)
        c = (oc * oc).sum(-1) - 1.0
        disc = b * b - 4.0 * a * c
        hit = disc >= 0.0
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        a2 = 2.0 * a
        inf = torch.full_like(a, float("inf"))
        te = torch.where(hit, (-b - sq) / a2, inf)
        tx = torch.where(hit, (-b + sq) / a2, inf)
        zeros = torch.zeros_like(o)
        hit3 = hit.unsqueeze(1)

        def ell_normal(t):
            p = o + t.unsqueeze(1) * d
            nrm = (p - self.centre) / (r * r)
            nrm = nrm / (nrm.norm(dim=1, keepdim=True) + 1e-30)
            return torch.where(hit3, nrm, zeros)

        return te, tx, ell_normal(te), ell_normal(tx)


# ---------------------------------------------------------------------------
# Box (axis-aligned)
# ---------------------------------------------------------------------------
class TBox:
    def __init__(self, lo, hi, dev, dt):
        self.lo = _t(lo, dev, dt)
        self.hi = _t(hi, dev, dt)
        self.dev, self.dt = dev, dt

    def ray_intersect(self, o, d, compiled=False):
        big = abs(d) > 1e-15
        safe_d = torch.where(big, d, torch.ones_like(d))
        inf = torch.full_like(d, float("inf"))
        inv_d = torch.where(big, 1.0 / safe_d, torch.sign(d) * inf)
        t1 = (self.lo - o) * inv_d
        t2 = (self.hi - o) * inv_d
        t_lo = torch.minimum(t1, t2)
        t_hi = torch.maximum(t1, t2)
        t_enter_val = t_lo.max(dim=1).values
        t_exit_val = t_hi.min(dim=1).values
        hit = t_enter_val <= t_exit_val
        infN = torch.full_like(t_enter_val, float("inf"))
        te = torch.where(hit, t_enter_val, infN)
        tx = torch.where(hit, t_exit_val, infN)

        centre = (self.lo + self.hi) / 2.0
        sgn = torch.sign(o - centre)
        hit1 = hit.unsqueeze(1)
        match_e = (t_lo - t_enter_val.unsqueeze(1)).abs() < 1e-10
        match_x = (t_hi - t_exit_val.unsqueeze(1)).abs() < 1e-10
        zeros = torch.zeros_like(o)
        ne = torch.where(match_e & hit1, -sgn, zeros)
        nx = torch.where(match_x & hit1, sgn, zeros)
        return te, tx, ne, nx


# ---------------------------------------------------------------------------
# Capsule (cylinder + two hemispherical caps, p0->p1)
# ---------------------------------------------------------------------------
class TCapsule:
    def __init__(self, p0, p1, radius, dev, dt):
        self.p0 = _t(p0, dev, dt)
        self.p1 = _t(p1, dev, dt)
        self.radius = float(radius)
        seg = self.p1 - self.p0
        self._len = float(torch.linalg.vector_norm(seg).item())
        self.dev, self.dt = dev, dt

    def ray_intersect(self, o, d, compiled=False):
        p0, p1, r, L = self.p0, self.p1, self.radius, self._len
        ax = (p1 - p0) / (L + 1e-30)
        ba = p1 - p0
        oa = o - p0
        baba = (ba * ba).sum()
        bard = (d * ba).sum(-1)
        baoa = (oa * ba).sum(-1)
        rdoa = (d * oa).sum(-1)
        oaoa = (oa * oa).sum(-1)
        a_ = baba - bard * bard
        b_ = baba * rdoa - baoa * bard
        c_ = baba * oaoa - baoa * baoa - r * r * baba
        h_ = b_ * b_ - a_ * c_

        inf = torch.full_like(bard, float("inf"))
        zeros = torch.zeros_like(o)

        cyl_ok = (h_ >= 0.0) & (a_.abs() > 1e-12)
        sq = torch.sqrt(torch.clamp(h_, min=0.0))
        safe_a = torch.where(cyl_ok, a_, torch.ones_like(a_))
        tc_e = (-b_ - sq) / safe_a
        tc_x = (-b_ + sq) / safe_a
        yc_e = baoa + tc_e * bard
        yc_x = baoa + tc_x * bard
        valid_e = cyl_ok & (yc_e >= 0.0) & (yc_e <= baba)
        valid_x = cyl_ok & (yc_x >= 0.0) & (yc_x <= baba)
        t_enter = torch.where(valid_e, tc_e, inf)
        t_exit = torch.where(valid_x, tc_x, inf)

        def cyl_normal(t, valid):
            p = o + t.unsqueeze(1) * d
            rel = p - p0
            proj = (rel * ax).sum(-1, keepdim=True)
            radial = rel - proj * ax
            nrm = radial / (radial.norm(dim=1, keepdim=True) + 1e-30)
            return torch.where(valid.unsqueeze(1), nrm, zeros)

        n_enter = cyl_normal(t_enter, valid_e)
        n_exit = cyl_normal(t_exit, valid_x)

        # caps: p0 (proj<=0) and p1 (proj>=L); each may pull t inward (min)
        for cap_centre, is_p1 in [(p0, False), (p1, True)]:
            oc = o - cap_centre
            b2 = (d * oc).sum(-1)
            c2 = (oc * oc).sum(-1) - r * r
            disc = b2 * b2 - c2
            cap_ok = disc >= 0.0
            sq2 = torch.sqrt(torch.clamp(disc, min=0.0))
            for tc_cap, t_cur, n_cur, set_t, set_n in [
                (-b2 - sq2, t_enter, n_enter, "e", None),
                (-b2 + sq2, t_exit, n_exit, "x", None),
            ]:
                pts = o + tc_cap.unsqueeze(1) * d
                proj = ((pts - p0) * ax).sum(-1)
                on_cap = (proj >= L) if is_p1 else (proj <= 0.0)
                better = cap_ok & on_cap & (tc_cap < t_cur)
                ncap = (pts - cap_centre) / r
                if set_t == "e":
                    t_enter = torch.where(better, tc_cap, t_enter)
                    n_enter = torch.where(better.unsqueeze(1), ncap, n_enter)
                else:
                    t_exit = torch.where(better, tc_cap, t_exit)
                    n_exit = torch.where(better.unsqueeze(1), ncap, n_exit)

        return t_enter, t_exit, n_enter, n_exit


# ---------------------------------------------------------------------------
# CSG (interval arithmetic on t) — mirrors scene/csg.py
# ---------------------------------------------------------------------------
class TIntersection:
    def __init__(self, children):
        self.children = list(children)

    def ray_intersect(self, o, d, compiled=False):
        te, tx, ne, nx = self.children[0].ray_intersect(o, d, compiled=compiled)
        for c in self.children[1:]:
            te2, tx2, ne2, nx2 = c.ray_intersect(o, d, compiled=compiled)
            use_e = te2 > te
            te = torch.where(use_e, te2, te)
            ne = torch.where(use_e.unsqueeze(1), ne2, ne)
            use_x = tx2 < tx
            tx = torch.where(use_x, tx2, tx)
            nx = torch.where(use_x.unsqueeze(1), nx2, nx)
        miss = te >= tx
        inf = torch.full_like(te, float("inf"))
        te = torch.where(miss, inf, te)
        tx = torch.where(miss, inf, tx)
        return te, tx, ne, nx


class TUnion:
    def __init__(self, children):
        self.children = list(children)

    def ray_intersect(self, o, d, compiled=False):
        te, tx, ne, nx = self.children[0].ray_intersect(o, d, compiled=compiled)
        for c in self.children[1:]:
            te2, tx2, ne2, nx2 = c.ray_intersect(o, d, compiled=compiled)
            miss_a = te >= tx
            miss_b = te2 >= tx2
            use_b_entry = (te2 < te) & (~miss_b)
            te = torch.where(use_b_entry, te2, te)
            ne = torch.where(use_b_entry.unsqueeze(1), ne2, ne)
            use_b_exit = (tx2 > tx) & (~miss_b)
            tx = torch.where(use_b_exit, tx2, tx)
            nx = torch.where(use_b_exit.unsqueeze(1), nx2, nx)
            mh = miss_a & (~miss_b)
            te = torch.where(mh, te2, te)
            tx = torch.where(mh, tx2, tx)
            ne = torch.where(mh.unsqueeze(1), ne2, ne)
            nx = torch.where(mh.unsqueeze(1), nx2, nx)
        return te, tx, ne, nx


class TDifference:
    def __init__(self, A, B):
        self.A, self.B = A, B

    def ray_intersect(self, o, d, compiled=False):
        te_a, tx_a, ne_a, nx_a = self.A.ray_intersect(o, d, compiled=compiled)
        te_b, tx_b, ne_b, nx_b = self.B.ray_intersect(o, d, compiled=compiled)
        te, tx, ne, nx = te_a.clone(), tx_a.clone(), ne_a.clone(), nx_a.clone()
        inf = torch.full_like(te, float("inf"))
        miss_a = te_a >= tx_a
        hit_b = te_b < tx_b

        clip_start = hit_b & (~miss_a) & (te_b <= te_a) & (tx_b > te_a) & (tx_b < tx_a)
        te = torch.where(clip_start, tx_b, te)
        ne = torch.where(clip_start.unsqueeze(1), -nx_b, ne)

        clip_end = hit_b & (~miss_a) & (te_b > te_a) & (te_b < tx_a) & (tx_b >= tx_a)
        tx = torch.where(clip_end, te_b, tx)
        nx = torch.where(clip_end.unsqueeze(1), -ne_b, nx)

        contains = hit_b & (~miss_a) & (te_b <= te_a) & (tx_b >= tx_a)
        te = torch.where(contains, inf, te)
        tx = torch.where(contains, inf, tx)
        te = torch.where(miss_a, inf, te)
        tx = torch.where(miss_a, inf, tx)
        return te, tx, ne, nx


# ---------------------------------------------------------------------------
# Tube (swept curve = K cylinder barrels + N_s sphere caps) -- resident port of
# numpy Tube._intersect_batch.  Intersection math is done in float64 regardless
# of engine dtype (the Phase-1 correctness fix: float32 catastrophically cancels
# in c_ = baba*oaoa - baoa**2 - r^2*baba at the 50 mm lever arm).
# ---------------------------------------------------------------------------
class TTube:
    def __init__(self, curve_pts, radius, dev, dt):
        pts = np.asarray(curve_pts, dtype=float)
        ba = pts[1:] - pts[:-1]
        baba = (ba * ba).sum(1)
        self.dev, self.dt = dev, dt
        self.r = float(radius)
        self.K = len(pts) - 1
        self.N_s = len(pts)
        # geometry kept in float64 (intersection precision); store dt too for cast-out
        f64 = torch.float64
        self._pts = _t(pts, dev, f64)
        self._p0 = _t(pts[:-1], dev, f64)
        self._ba = _t(ba, dev, f64)
        self._baba = _t(baba, dev, f64)
        self._ax = _t(ba / (np.sqrt(baba)[:, None] + 1e-30), dev, f64)
        self._bbox_lo = _t(pts.min(0) - self.r, dev, f64)
        self._bbox_hi = _t(pts.max(0) + self.r, dev, f64)
        self._kernel_c = None   # lazy torch.compile handle (preview path only)

    def _compiled_kernel(self):
        """Dynamic-batch torch.compile of the heavy tube _kernel (CUDA only).

        The AABB cull's survivor count is data-dependent, so the tube math
        cannot live inside the outer compiled next_interface graph without
        specializing on the per-pose survivor count (a recompile per distinct
        count until dynamo's cache cap, then permanent eager -- the fusion
        loss behind slow /motor-driven previews). Compiling the kernel on its
        own with dynamic=True over the survivor dim compiles ONCE and reuses.
        """
        if self._kernel_c is None:
            self._kernel_c = torch.compile(self._kernel, mode="default",
                                           dynamic=True)
        return self._kernel_c

    @torch._dynamo.disable
    def ray_intersect(self, o, d, compiled=False):
        # AABB cull: run the heavy (B,K,3) kernel only on bbox survivors.
        # Byte-identical (culled rays geometrically miss -> inf either way).
        # dynamo-disabled: the compiled next_interface graph breaks cleanly
        # here, so the data-dependent survivor count never specializes the
        # outer graph; on the preview path the tube math is fused by the
        # separately-compiled kernel below. `compiled` is threaded explicitly
        # through the call chain (no shared mutable state), so eager callers
        # (settle, /xray, parity tests) provably run the eager kernel.
        N = o.shape[0]
        te = torch.full((N,), float("inf"), device=o.device, dtype=self.dt)
        tx = torch.full((N,), float("inf"), device=o.device, dtype=self.dt)
        ne = torch.zeros((N, 3), device=o.device, dtype=self.dt)
        nx = torch.zeros((N, 3), device=o.device, dtype=self.dt)
        surv = _aabb_survivors(o, d, self._bbox_lo, self._bbox_hi)
        # nonzero is the single host sync here (numel on the materialized index
        # tensor is a free shape read; the old bool(surv.any()) pre-check was a
        # second, redundant sync).
        idx = surv.nonzero(as_tuple=False).squeeze(1)
        if idx.numel():
            o_s, d_s = o[idx], d[idx]
            if compiled and self.dev.type == "cuda":
                # Metadata hint: treat the survivor dim as dynamic so the very
                # first compile is already batch-size-agnostic.
                torch._dynamo.maybe_mark_dynamic(o_s, 0)
                torch._dynamo.maybe_mark_dynamic(d_s, 0)
                kte, ktx, kne, knx = self._compiled_kernel()(o_s, d_s)
            else:
                kte, ktx, kne, knx = self._kernel(o_s, d_s)
            te[idx] = kte
            tx[idx] = ktx
            ne[idx] = kne
            nx[idx] = knx
        return te, tx, ne, nx

    def _kernel(self, o, d):
        f64 = torch.float64
        of, df = o.to(f64), d.to(f64)
        p0, ba, baba, ax, pts, r = self._p0, self._ba, self._baba, self._ax, self._pts, self.r
        B, K, N_s = of.shape[0], self.K, self.N_s
        INF = float("inf")

        oa = of.unsqueeze(1) - p0.unsqueeze(0)          # (B,K,3)
        bard = df @ ba.t()                               # (B,K)
        baoa = (oa * ba.unsqueeze(0)).sum(-1)
        rdoa = (df.unsqueeze(1) * oa).sum(-1)
        oaoa = (oa * oa).sum(-1)
        baba_bk = baba.unsqueeze(0)

        a_ = baba_bk - bard ** 2
        b_ = baba_bk * rdoa - baoa * bard
        c_ = baba_bk * oaoa - baoa ** 2 - r * r * baba_bk
        h_ = b_ * b_ - a_ * c_
        small_a = a_.abs() < 1e-12
        inv_a = torch.where(small_a, torch.zeros_like(a_),
                            1.0 / torch.where(small_a, torch.ones_like(a_), a_))
        v_cyl = (h_ >= 0.0) & (~small_a)
        sq = torch.where(v_cyl, torch.sqrt(torch.clamp(h_, min=0.0)), torch.zeros_like(h_))
        raw_te = (-b_ - sq) * inv_a
        raw_tx = (-b_ + sq) * inv_a
        ye = baoa + raw_te * bard
        yx = baoa + raw_tx * bard
        in_seg = baba_bk > 0
        big = torch.full_like(raw_te, INF)
        te_k = torch.where(v_cyl & (ye >= 0.0) & (ye <= baba_bk) & in_seg, raw_te, big)
        tx_k = torch.where(v_cyl & (yx >= 0.0) & (yx <= baba_bk) & in_seg, raw_tx, big)
        te_k = torch.where(te_k < tx_k, te_k, big)

        vc = of.unsqueeze(1) - pts.unsqueeze(0)          # (B,N_s,3)
        b_sp = (vc * df.unsqueeze(1)).sum(-1)
        c_sp = (vc * vc).sum(-1) - r * r
        h_sp = b_sp * b_sp - c_sp
        v_sp = h_sp >= 0.0
        sq_sp = torch.where(v_sp, torch.sqrt(torch.clamp(h_sp, min=0.0)), torch.zeros_like(h_sp))
        big_sp = torch.full_like(b_sp, INF)
        te_sp = torch.where(v_sp, -b_sp - sq_sp, big_sp)
        tx_sp = torch.where(v_sp, -b_sp + sq_sp, big_sp)
        te_sp = torch.where(te_sp < tx_sp, te_sp, big_sp)

        te_all = torch.cat([te_k, te_sp], dim=1)         # (B, K+N_s)
        bi = torch.arange(B, device=of.device)
        best_k = te_all.argmin(1)
        best_te = te_all[bi, best_k]
        tx_cyl_after = torch.where(tx_k > best_te.unsqueeze(1), tx_k, big)
        tx_sp_after = torch.where(tx_sp > best_te.unsqueeze(1), tx_sp, big_sp)
        exit_cyl_k = tx_cyl_after.argmin(1)
        exit_sp_k = tx_sp_after.argmin(1)
        best_tx_cyl = tx_cyl_after[bi, exit_cyl_k]
        best_tx_sp = tx_sp_after[bi, exit_sp_k]
        use_cyl_exit = best_tx_cyl < INF
        best_tx = torch.where(use_cyl_exit, best_tx_cyl, best_tx_sp)
        exit_k = torch.where(use_cyl_exit, exit_cyl_k, exit_sp_k + K)

        def normals_for(t_vals, k_idx):
            m = torch.isfinite(t_vals)
            P = of + t_vals.unsqueeze(1) * df
            is_cyl = k_idx < K
            k_c = k_idx.clamp(0, K - 1)
            p0_k, ax_k = p0[k_c], ax[k_c]
            proj = ((P - p0_k) * ax_k).sum(-1, keepdim=True)
            rad_c = P - (p0_k + proj * ax_k)
            n_cyl = rad_c / (rad_c.norm(dim=1, keepdim=True) + 1e-30)
            k_s = (k_idx - K).clamp(0, N_s - 1)
            rad_s = P - pts[k_s]
            n_sph = rad_s / (rad_s.norm(dim=1, keepdim=True) + 1e-30)
            n = torch.where(is_cyl.unsqueeze(1), n_cyl, n_sph)
            return torch.where(m.unsqueeze(1), n, torch.zeros_like(n))

        ne = normals_for(best_te, best_k)
        nx = normals_for(best_tx, exit_k)
        return (best_te.to(self.dt), best_tx.to(self.dt), ne.to(self.dt), nx.to(self.dt))


# ---------------------------------------------------------------------------
# SurfaceMesh (Moller-Trumbore) -- resident port of numpy SurfaceMesh.ray_intersect.
# Intersection in float64 (same precision rationale). Brute-force over all faces;
# a GPU broad-phase is deferred to 2f/2g (fine at current mesh sizes).
# ---------------------------------------------------------------------------
class TSurfaceMesh:
    def __init__(self, vertices, faces, dev, dt):
        v = np.asarray(vertices, dtype=float)
        f = np.asarray(faces, dtype=int)
        v0 = v[f[:, 0]]; v1 = v[f[:, 1]]; v2 = v[f[:, 2]]
        e1 = v1 - v0; e2 = v2 - v0
        N = np.cross(e1, e2)
        fn = N / (np.linalg.norm(N, axis=1, keepdims=True) + 1e-30)
        f64 = torch.float64
        self._v0 = _t(v0, dev, f64); self._e1 = _t(e1, dev, f64); self._e2 = _t(e2, dev, f64)
        self._fn = _t(fn, dev, f64)
        self._eps = 1e-8
        self.dev, self.dt = dev, dt

    def _mt_batch(self, o, d):
        v0, e1, e2 = self._v0, self._e1, self._e2
        B, F = o.shape[0], v0.shape[0]
        INF = float("inf"); eps = self._eps
        h = torch.linalg.cross(d.unsqueeze(1).expand(-1, F, -1),
                               e2.unsqueeze(0).expand(B, -1, -1), dim=2)
        a = (e1.unsqueeze(0) * h).sum(-1)
        par = a.abs() < eps
        inv_a = torch.where(par, torch.zeros_like(a),
                            1.0 / torch.where(par, torch.ones_like(a), a))
        s = o.unsqueeze(1) - v0.unsqueeze(0)
        u = inv_a * (s * h).sum(-1)
        q = torch.linalg.cross(s, e1.unsqueeze(0).expand(B, -1, -1), dim=2)
        v = inv_a * (d.unsqueeze(1) * q).sum(-1)
        t = inv_a * (e2.unsqueeze(0) * q).sum(-1)
        miss = par | (u < 0) | (u > 1) | (v < 0) | ((u + v) > 1)
        t_fwd = torch.where(miss | (t < eps), torch.full_like(t, INF), t)
        t_fwd_r = torch.where(miss | (t < eps), torch.full_like(t, -INF), t)
        t_bwd = torch.where(miss | (t > -eps), torch.full_like(t, INF), -t)
        bi = torch.arange(B, device=o.device)
        fi_min = t_fwd.argmin(1); fi_max = t_fwd_r.argmax(1); fi_bwd = t_bwd.argmin(1)
        t_min = t_fwd[bi, fi_min]; t_max = t_fwd_r[bi, fi_max]
        has_bwd = t_bwd[bi, fi_bwd] < INF
        t_back = torch.where(has_bwd, -t_bwd[bi, fi_bwd], torch.zeros_like(t_min))
        return t_min, t_max, fi_min, fi_max, t_back, fi_bwd

    def ray_intersect(self, o, d, compiled=False):
        f64 = torch.float64
        of, df = o.to(f64), d.to(f64)
        INF = float("inf")
        t_min, t_max, fi_min, fi_max, t_back, fi_bwd = self._mt_batch(of, df)

        has_fwd = t_min < INF
        has_bwd = t_back < 0.0
        inside = has_fwd & has_bwd & (t_min >= t_max)
        outside = has_fwd & (~inside)
        te_all = torch.where(inside, t_back, t_min)
        tx_all = torch.where(inside, t_min, t_max)
        grazing = outside & (t_min >= t_max)
        tx_all = torch.where(grazing, t_min, tx_all)
        valid = inside | outside

        big = torch.full_like(t_min, INF)
        t_enter = torch.where(valid, te_all, big)
        t_exit = torch.where(valid, tx_all, big)

        fi_e = torch.where(inside, fi_bwd, fi_min)
        fn_e = self._fn[fi_e]
        cos_e = (df * fn_e).sum(-1, keepdim=True)
        ne = torch.where(cos_e < 0, fn_e, -fn_e)
        zeros = torch.zeros_like(fn_e)
        ne = torch.where(valid.unsqueeze(1), ne, zeros)

        fi_x = torch.where(inside, fi_min, fi_max)
        fn_x = self._fn[fi_x]
        cos_x = (df * fn_x).sum(-1, keepdim=True)
        nx = torch.where(cos_x > 0, fn_x, -fn_x)
        nx = torch.where(valid.unsqueeze(1), nx, zeros)
        return (t_enter.to(self.dt), t_exit.to(self.dt), ne.to(self.dt), nx.to(self.dt))


# ---------------------------------------------------------------------------
# TNull: inert stand-in for provably-hitless shapes (e.g. a radius-0 Sphere,
# the default hampton scene's placeholder "solvent" object).
#
# A zero-radius sphere can never yield an interface: disc = 4[(oc.d)^2 -
# (d.d)(oc.oc)] <= 0 by Cauchy-Schwarz, so entry == exit at best and the
# strict te < tx tests in next_interface (and the material-interval probe)
# reject it identically in numpy and torch. Substituting it out at build time
# removes its kernels from every depth iteration while keeping the object
# index (and so the material tables) aligned -- byte-exact by construction.
# ---------------------------------------------------------------------------
class TNull:
    def __init__(self, dev, dt):
        self.dev, self.dt = dev, dt

    def ray_intersect(self, o, d, compiled=False):
        inf = torch.full(o.shape[:1], float("inf"), device=o.device, dtype=o.dtype)
        zeros = torch.zeros_like(o)
        return inf, inf, zeros, zeros


# ---------------------------------------------------------------------------
# Builder: numpy shape -> torch shape
# ---------------------------------------------------------------------------
_PRIMITIVE_BUILDERS = {
    "Sphere": lambda s, dev, dt: (TSphere(s.centre, s.radius, dev, dt)
                                  if s.radius > 0 else TNull(dev, dt)),
    "HalfSpace": lambda s, dev, dt: THalfSpace(s.normal, s.offset, dev, dt),
    "InfiniteCylinder": lambda s, dev, dt: TInfiniteCylinder(s.centre, s.axis, s.radius, dev, dt),
    "Ellipsoid": lambda s, dev, dt: TEllipsoid(s.centre, s.radii, dev, dt),
    "Box": lambda s, dev, dt: TBox(s.lo, s.hi, dev, dt),
    "Capsule": lambda s, dev, dt: TCapsule(s.p0, s.p1, s.radius, dev, dt),
}


def build_torch_shape(shape, dev, dt):
    """Build the torch mirror of a numpy shape tree (primitives + CSG + Cylinder).

    Tube and SurfaceMesh are added in sub-step 2c. The recursion reuses each
    numpy node's own sub-shapes/parameters, so the torch tree is parameter-
    identical to the numpy one by construction.
    """
    name = type(shape).__name__
    if name in _PRIMITIVE_BUILDERS:
        return _PRIMITIVE_BUILDERS[name](shape, dev, dt)
    if name == "Intersection":
        return TIntersection([build_torch_shape(c, dev, dt) for c in shape.children])
    if name == "Union":
        return TUnion([build_torch_shape(c, dev, dt) for c in shape.children])
    if name == "Difference":
        return TDifference(build_torch_shape(shape.A, dev, dt),
                           build_torch_shape(shape.B, dev, dt))
    if name == "Cylinder":
        # Composite: barrel ∩ (cap_lo ∩ cap_hi) — reuse the numpy sub-shapes.
        barrel = build_torch_shape(shape._barrel, dev, dt)
        caps = TIntersection([build_torch_shape(shape._cap_lo, dev, dt),
                              build_torch_shape(shape._cap_hi, dev, dt)])
        return TIntersection([barrel, caps])
    if name == "Tube":
        return TTube(shape._curve_pts, shape.radius, dev, dt)
    if name == "SurfaceMesh":
        return TSurfaceMesh(shape.vertices, shape.faces, dev, dt)
    if name == "ThinShell":
        # ThinShell wraps an internal SurfaceMesh and delegates ray_intersect to it.
        return TSurfaceMesh(shape._mesh.vertices, shape._mesh.faces, dev, dt)
    raise NotImplementedError(f"torch port of {name} not implemented yet")


# ---------------------------------------------------------------------------
# TorchScene: resident scene + next_interface (mirrors scene.Scene.next_interface)
# ---------------------------------------------------------------------------
class TorchScene:
    """A loaded numpy Scene mirrored as resident torch shapes, with a torch
    next_interface() that matches scene.py's numpy version. The numpy Scene is
    kept for geometry/camera/material metadata used by the render path (2e)."""

    def __init__(self, scene, dev, dt):
        self.dev, self.dt = dev, dt
        self.scene = scene
        # Lazily-built torch.compile handle for the preview hot path (CUDA only).
        self._compiled_ni = None
        self.shapes = [build_torch_shape(ob.shape, dev, dt) for ob in scene.objects]
        self.n_obj = len(self.shapes)
        # Material lookup tables, K = 1 + n_obj (index 0 = background).
        # n_background = 1.0 (matches _trace_rays default); mu/colour from background.
        bg, objs = scene.background, scene.objects
        self.K = 1 + len(objs)
        self.mat_n = _t([1.0] + [o.material.n for o in objs], dev, dt)
        self.mat_mu = _t([bg.mu_optical] + [o.material.mu_optical for o in objs], dev, dt)
        self.mat_col = _t([list(bg.color)] + [list(o.material.color) for o in objs], dev, dt)
        # X-ray linear attenuation coefficient (mm⁻¹) per material, same index
        # convention (0 = background). Used by trace_xray / render_xray_torch.
        self.mat_muxray = _t([bg.mu_xray] + [o.material.mu_xray for o in objs], dev, dt)

    def _compiled_next_interface(self):
        """Lazily build a torch.compile()d next_interface for the PREVIEW path.

        Compiled only on CUDA (Inductor's CPU backend historically miscompiled
        the mesh/CSG path here), with mode="default" — reduce-overhead's implicit
        CUDA-graph capture is not thread-safe in the single-flight server ("already
        recording to mempool"/CUBLAS crashes). dynamic=True keeps the symbolic
        batch dim from recompiling as depth-compaction shrinks the active set.
        On CPU (or first-build failure upstream) callers fall back to eager.
        """
        if self.dev.type != "cuda":
            return self.next_interface
        if self._compiled_ni is None:
            self._compiled_ni = torch.compile(
                self.next_interface, mode="default", dynamic=True)
        return self._compiled_ni

    def next_interface(self, o, d, t_min=1e-6, compiled=False):
        """Returns (best_t (N,), best_n (N,3), mat_out_oi (N,) long; -1 = background).

        `compiled` is forwarded to every shape's ray_intersect: only TTube acts
        on it (its heavy kernel gets a separately-compiled dynamic-batch
        variant on the preview path); everything else ignores it, so eager
        callers stay byte-exact by construction."""
        N = o.shape[0]
        dev, dt = o.device, o.dtype
        INF = float("inf")
        best_t = torch.full((N,), INF, device=dev, dtype=dt)
        best_n = torch.zeros((N, 3), device=dev, dtype=dt)
        te_list, tx_list = [], []
        for oi, shape in enumerate(self.shapes):
            te, tx, ne, nx = shape.ray_intersect(o, d, compiled=compiled)
            te_list.append(te)
            tx_list.append(tx)
            # entry, then exit using the entry-updated best_t (matches numpy order)
            valid_e = (te > t_min) & (te < tx) & (te < best_t)
            best_t = torch.where(valid_e, te, best_t)
            best_n = torch.where(valid_e.unsqueeze(1), ne, best_n)
            valid_x = (tx > t_min) & (te < tx) & (tx < best_t)
            best_t = torch.where(valid_x, tx, best_t)
            best_n = torch.where(valid_x.unsqueeze(1), nx, best_n)

        all_te = torch.stack(te_list, dim=1)   # (N, n_obj)
        all_tx = torch.stack(tx_list, dim=1)
        hit_mask = best_t < INF
        # material after crossing: first object whose interval contains best_t + 1e-4
        t_probe = (best_t + 1e-4).unsqueeze(1)
        inside = (all_te < t_probe) & (t_probe < all_tx)            # (N, n_obj) bool
        nobj = self.n_obj
        idx = torch.arange(nobj, device=dev).unsqueeze(0)
        first_true = torch.where(inside, idx, torch.full_like(idx, nobj)).min(dim=1).values
        has_any = first_true < nobj
        am = first_true.clamp(max=max(nobj - 1, 0)).to(torch.long)
        neg1 = torch.full((N,), -1, device=dev, dtype=torch.long)
        mat_out_oi = torch.where(hit_mask & has_any, am, neg1)
        return best_t, best_n, mat_out_oi

    def trace_rays(self, o, d, na_obj, opt_axis_sample, max_depth=None, color_mu=None,
                   compiled=False):
        """Masked-active-ray port of microscope._trace_rays. Returns (N,3) RGB in [0,1].

        Uses a boolean `alive` mask instead of compaction (semantically identical
        since next_interface is per-ray independent; enables fixed shapes for
        CUDA-graph capture in 2g). The NA cutoff is applied to a ray at the depth
        it exits, exactly as numpy does.

        `compiled=True` routes the per-depth interface query through the
        torch.compile()d next_interface (preview-only fusion win); eager otherwise.
        """
        from .microscope import MAX_DEPTH, _COLOR_MU
        if max_depth is None:
            max_depth = MAX_DEPTH
        if color_mu is None:
            color_mu = _COLOR_MU
        ni_fn = self._compiled_next_interface() if compiled else self.next_interface

        def next_interface(og, dg, t_min=1e-6):
            return ni_fn(og, dg, t_min=t_min, compiled=compiled)

        N = o.shape[0]
        dev, dt = o.device, o.dtype
        INF = float("inf")
        intensity = torch.ones((N, 3), device=dev, dtype=dt)
        opt_axis = _t(opt_axis_sample, dev, dt)
        cos_na = float(np.cos(np.arcsin(np.clip(na_obj, 0.0, 1.0))))

        o = o.clone()
        d = d.clone()
        cur_mat = torch.zeros(N, device=dev, dtype=torch.long)   # 0 = background
        gidx = torch.arange(N, device=dev)                       # active-ray indices

        # Compaction (mirrors numpy _trace_rays): each depth processes only the
        # still-active rays. The active set collapses by depth ~2-3, so depths
        # 1.. are cheap -- ~10x less work than masking the full frame every depth.
        # All per-depth math is where-masked over the whole active set (pure
        # elementwise -> bit-exact for every kept lane), so the ONLY host sync
        # per depth is the single keep-compaction at the bottom; the CPU queues
        # a whole depth's kernels without stalling on bool(any())/mask indexing.
        for _ in range(max_depth):
            if gidx.numel() == 0:
                break
            og, dg = o[gidx], d[gidx]
            t_next, normals, mat_out = next_interface(og, dg, t_min=1e-6)
            hit = t_next < INF

            # NA cutoff for rays that exited the scene (using their exit direction)
            cos_exit = (dg * opt_axis).sum(-1)
            na_cut = (~hit) & (cos_exit < cos_na)

            cur_g = cur_mat[gidx]
            inten_g = intensity[gidx]
            # Beer-Lambert over the segment just travelled (current material)
            mu_per_ch = self.mat_mu[cur_g].unsqueeze(1) + color_mu * (1.0 - self.mat_col[cur_g])
            new_int = inten_g * torch.exp(-mu_per_ch * t_next.unsqueeze(1))

            new_orig = og + t_next.unsqueeze(1) * dg
            n1 = self.mat_n[cur_g]
            nos = (mat_out + 1).clamp(0, self.K - 1)
            n2 = self.mat_n[nos]
            cos_i = (dg * normals).sum(-1).abs()
            new_int = new_int * _fresnel_T_t(n1, n2, cos_i).unsqueeze(1)

            new_dirs = _snell_refract_t(dg, normals, n1, n2)
            tir = torch.isnan(new_dirs).any(dim=1)

            # One indexed intensity write per depth: hit lanes take the
            # Beer-Lambert*Fresnel product, killed lanes (NA cutoff / TIR) take
            # exact zeros, exited-and-collected lanes write back their old bits.
            # (no-hit lanes' new_int may hold inf/nan garbage -- never selected.)
            out_int = torch.where(hit.unsqueeze(1), new_int, inten_g)
            kill = na_cut | (hit & tir)
            out_int = torch.where(kill.unsqueeze(1), torch.zeros_like(out_int), out_int)
            intensity[gidx] = out_int

            # Single compaction (the depth's one host sync): survivors are
            # hit & ~TIR, so new_dirs NaN rows never enter surviving state; an
            # all-dead depth compacts to zero and exits at the next loop top.
            kidx = (hit & ~tir).nonzero(as_tuple=False).squeeze(1)
            gk = gidx[kidx]
            o[gk] = new_orig[kidx]
            d[gk] = new_dirs[kidx]
            cur_mat[gk] = nos[kidx]
            gidx = gk

        return intensity

    def trace_xray(self, o, d, max_depth=None):
        """Straight-ray X-ray traversal (no refraction, no NA): accumulate
        optical depth ``tau = Σ mu_xray·L`` per ray. Returns (N,) tau; the
        transmission is exp(-tau).

        Mirrors trace_rays' next_interface stepping + compaction, but the ray
        goes straight through every interface (X-rays are undeviated at these
        indices) and we accumulate mu_xray·segment instead of optical
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
            t_next, _normals, mat_out = self.next_interface(og, dg, t_min=1e-6)
            hit = t_next < INF
            if not bool(hit.any()):
                break
            gh = gidx[hit]
            cur_h = cur_mat[gh]
            # Beer-Lambert optical depth over the segment just travelled
            tau[gh] = tau[gh] + self.mat_muxray[cur_h] * t_next[hit]
            # advance straight to the interface; direction unchanged
            o[gh] = og[hit] + t_next[hit].unsqueeze(1) * dg[hit]
            cur_mat[gh] = (mat_out[hit] + 1).clamp(0, self.K - 1)
            gidx = gh

        return tau


# ---------------------------------------------------------------------------
# render: torch port of microscope.render. Reuses the numpy ray-grid/condenser
# setup verbatim (tiny, per-frame) so any divergence is isolated to the trace.
# Returns an (H, W, 3) torch tensor in [0, 1]; condenser rays still looped here
# (batched into one resident trace in 2f).
# ---------------------------------------------------------------------------
@torch.inference_mode()
def render_torch(tscene, goniometer, n_cond=1, tile_size=250_000, compiled=False):
    from .microscope import _condenser_offsets
    from ..motors.goniometer import apply_transform

    scene = tscene.scene
    dev, dt = tscene.dev, tscene.dt
    cam = scene.camera_cfg
    W = int(cam.get("width", 640))
    H = int(cam.get("height", 480))
    pixel_size = float(cam.get("pixel_size", 0.005))
    eff_px = pixel_size / goniometer.zoom
    na_obj = float(cam.get("na_objective", 0.10))
    na_cond = float(cam.get("na_condenser", 0.07))

    g = scene.geometry
    fast = np.array(g.get("camera_fast", [1, 0, 0]), dtype=float)
    slow = np.array(g.get("camera_slow", [0, 1, 0]), dtype=float)
    opt_axis = np.array(g.get("optical_axis", [0, 0, -1]), dtype=float)
    opt_axis /= np.linalg.norm(opt_axis)

    px = (np.arange(W) - W / 2.0) * eff_px
    py = (np.arange(H) - H / 2.0) * eff_px
    gx, gy = np.meshgrid(px, py)
    focal_pts = (gx[:, :, None] * fast + gy[:, :, None] * slow).reshape(-1, 3)
    T_inv = goniometer.transform_inv()
    focal_pts_s = apply_transform(T_inv, focal_pts)
    opt_axis_s = T_inv[:3, :3] @ opt_axis
    opt_axis_s /= np.linalg.norm(opt_axis_s) + 1e-30

    # Build ALL n_cond condenser ray sets and trace them in ONE resident pass,
    # tiled to bound peak memory (the (B,K,3) tube intermediate scales with B).
    # All condenser rays share opt_axis_s, so a single trace_rays handles them.
    offsets = _condenser_offsets(n_cond, na_cond)
    WH = W * H
    o_all = np.empty((n_cond * WH, 3))
    d_all = np.empty((n_cond * WH, 3))
    for k in range(n_cond):
        ox, oy, _ = offsets[k]
        illum_dir = opt_axis + ox * fast + oy * slow
        illum_dir /= np.linalg.norm(illum_dir)
        illum_dir_s = T_inv[:3, :3] @ illum_dir
        illum_dir_s /= np.linalg.norm(illum_dir_s) + 1e-30
        t_up = 50.0 / max(abs(np.dot(illum_dir, opt_axis)), 1e-6)
        o_all[k * WH:(k + 1) * WH] = focal_pts_s - t_up * illum_dir_s
        d_all[k * WH:(k + 1) * WH] = illum_dir_s

    o_t = torch.as_tensor(o_all, device=dev, dtype=dt)
    d_t = torch.as_tensor(d_all, device=dev, dtype=dt)
    M = n_cond * WH
    # Never split one condenser sample across tiles: at 640x480 this traces
    # n_cond=1 in a single pass (2 -> 1 tiles) and aligns n_cond=7 to seven
    # sample-sized tiles (9 -> 7). Byte-exact -- per-ray results are tile-
    # independent -- and peak memory grows only ~25% (measured, well clear of
    # the WSL2 VRAM spill cliff).
    tile_size = max(tile_size, WH)
    out = torch.empty((M, 3), device=dev, dtype=dt)
    for s in range(0, M, tile_size):
        e = min(s + tile_size, M)
        out[s:e] = tscene.trace_rays(o_t[s:e], d_t[s:e], na_obj, opt_axis_s,
                                     compiled=compiled)
    # average over the condenser dimension
    return out.reshape(n_cond, WH, 3).mean(dim=0).reshape(H, W, 3)


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
        tau[s:e] = tscene.trace_xray(o_t[s:e], d_t[s:e])
    return torch.exp(-tau).reshape(H, W)
