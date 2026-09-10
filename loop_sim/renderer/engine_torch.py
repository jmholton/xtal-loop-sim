"""
GPU-resident torch rendering engine (Phase 2).

This module is a torch reimplementation of the numpy intersection/render path,
parametric over device ('cpu'|'cuda') and dtype (float32|float64). The numpy
path (scene/primitives.py, scene/csg.py, renderer/microscope.py) is preserved
verbatim as the canonical ground-truth REFERENCE; this engine is validated
against it by a three-rung differential ladder:

    rung 1  numpy-f64        -- frozen reference / executable spec
    rung 2  torch-cpu-f64    -- must equal rung 1 to ~1e-10 (proves the port; no GPU)
    rung 3  torch-cuda       -- must equal rung 1 within uint8 tolerance (the served path)

Each torch shape mirrors the corresponding numpy primitive's ray_intersect
contract exactly:
    ray_intersect(o, d) -> (t_enter, t_exit, n_enter, n_exit)   [all torch tensors]
    o, d   : (N, 3) on (device, dtype)
    t_*    : (N,)   +inf where no hit (HalfSpace may return +-inf interval ends)
    n_*    : (N, 3) outward normals; zero where no hit

"""
import os
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
# CSG (interval arithmetic on t) -- mirrors scene/csg.py
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
# Intersection in float64 (same precision rationale).
#
# AABB-culled since 2026-08-11, which is a PARITY RESTORATION rather than a new
# approximation: the numpy SurfaceMesh.ray_intersect this class ports has always
# run the same slab test and fed only survivors to Moller-Trumbore.  This class
# was the one that diverged, brute-forcing every ray against every face.  That
# cost is why a droplet scene was 162x more expensive per pixel than a tube one
# and why `fit_tile_size` had to shrink the trace tile to ~6800 rays: the
# measured law is 160 B per ray per face, and F was never reduced.
#
# The cull cannot change a pixel.  Every point of every triangle lies inside the
# vertex AABB by construction, so a ray missing the box provably misses all
# faces, and the brute-force path returned INF for exactly those rays.
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
        # Bounds over the REFERENCED triangle corners, not over `vertices` --
        # an unreferenced vertex would inflate the box and cost survivors for
        # nothing.  Both are conservative; this one is tighter.
        corners = v[f].reshape(-1, 3)
        self._bbox_lo = _t(corners.min(axis=0), dev, f64)
        self._bbox_hi = _t(corners.max(axis=0), dev, f64)

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

    @torch._dynamo.disable
    def ray_intersect(self, o, d, compiled=False):
        # AABB cull, mirroring TTube.ray_intersect: run the heavy (B,F,3)
        # Moller-Trumbore only on bbox survivors.  dynamo-disabled for the same
        # reason TTube is -- the data-dependent survivor count would otherwise
        # specialize the outer compiled next_interface graph on every pose.
        N = o.shape[0]
        te = torch.full((N,), float("inf"), device=o.device, dtype=self.dt)
        tx = torch.full((N,), float("inf"), device=o.device, dtype=self.dt)
        ne = torch.zeros((N, 3), device=o.device, dtype=self.dt)
        nx = torch.zeros((N, 3), device=o.device, dtype=self.dt)
        surv = _aabb_survivors(o, d, self._bbox_lo, self._bbox_hi)
        idx = surv.nonzero(as_tuple=False).squeeze(1)
        # Chunk the survivors so the (B,F,3) working set is bounded HERE rather
        # than by shrinking the caller's tile.  Per-ray results are independent
        # of how rays are grouped -- the same argument that makes the outer tile
        # loop byte-exact -- so this cannot change a pixel.  Bounding it locally
        # is what lets fit_tile_size stop paying the mesh term for every ray in
        # the frame when only ~0.3% of them reach a face.
        step = _mesh_survivor_chunk(self._v0.shape[0], self.dev)
        for s in range(0, idx.numel(), step):
            sub = idx[s:s + step]
            kte, ktx, kne, knx = self._intersect_all(o[sub], d[sub])
            te[sub] = kte
            tx[sub] = ktx
            ne[sub] = kne
            nx[sub] = knx
        return te, tx, ne, nx

    def _intersect_all(self, o, d):
        """The un-culled intersection.  Unchanged from the pre-cull body, so a
        survivor gets bit-for-bit what brute force gave it."""
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
# A zero-radius sphere never yields an interface (Cauchy-Schwarz forces entry
# == exit), so both engines reject it identically; substituting TNull at
# build time removes its kernels from every depth iteration while keeping the
# object index, and so the material tables, aligned -- byte-exact by
# construction.
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
        # Composite: barrel ∩ (cap_lo ∩ cap_hi) -- reuse the numpy sub-shapes.
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
        # convention (0 = background). Used by xray_torch.py's trace_xray.
        self.mat_muxray = _t([bg.mu_xray] + [o.material.mu_xray for o in objs], dev, dt)

    def _compiled_next_interface(self):
        """Lazily build a torch.compile()d next_interface for the PREVIEW path.

        Compiled only on CUDA (Inductor's CPU backend historically miscompiled
        the mesh/CSG path here), with mode="default" -- reduce-overhead's implicit
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


# ---------------------------------------------------------------------------
# Automatic tile sizing.
#
# Peak memory is linear in the number of rays in flight, with a scene-dependent
# slope: a tube scene costs ~1.1 GiB per million rays, a mesh scene scales as
# tile_rays * faces * 24 B and is orders of magnitude steeper. So the slope is
# measured per scene rather than assumed.
#
# The sizing is deliberately PREDICTIVE, not try-and-retry. Under WSL2 there is
# no OOM to back off from: past the card's capacity the Windows driver silently
# spills to host RAM and the render crawls 10-50x instead of failing, so a retry
# loop would hang rather than recover. See docs/RUNBOOK.md.
# ---------------------------------------------------------------------------
_TILE_MIN = 32_768
# Big enough that a camera-sized frame (640x480 = 307k rays per condenser
# sample) is still a single pass. The old code clamped the tile up to W*H, so a
# smaller default would quietly split those frames and shift every recorded
# benchmark: measured +8.3% at n_cond=7 with a 250k tile. Anything rendering
# larger than this -- the library builder -- passes tile_size=None for the
# VRAM-aware size instead.
_TILE_DEFAULT = 1_000_000

# Memory law: every Moller-Trumbore temporary in TSurfaceMesh._mt_batch is
# (B, F) or (B, F, 3), so mesh intersection memory is linear in (rays x faces)
# at 160 B per ray per face (measured 2026-08-07, RTX 4080 SUPER, stable
# across tile sizes and scenes). Scenes with no mesh carry no such term
# (0.5 KB/ray total) and keep the flat _TILE_DEFAULT above.
#
# This tile (`fit_tile_size`) bounds the ray-grid and condenser terms, which
# scale with total_rays alone. It does not bound the mesh term: that is
# `_mesh_survivor_chunk`'s job, sized against AABB survivors rather than the
# whole tile (see the TSurfaceMesh header above).
_MESH_BYTES_PER_RAY_FACE = 160
# Floor for the computed tile. Below this the Python-level tile loop starts to
# dominate; a 2880-face scene on a 16 GB card lands near 20k, so this only binds
# on much heavier meshes, where being slow beats spilling.
_TILE_FIT_MIN = 4_096

# Floor for the survivor chunk. Survivors are ~0.3% of a frame on the shipped
# droplet scene, so this almost never binds -- it exists for a pose that puts
# the mesh across the whole field (deep zoom into the drop).
_MESH_CHUNK_MIN = 2_048

# ABSOLUTE ceiling on the survivor working set, and the reason it is absolute:
# sizing purely as a fraction of FREE VRAM takes whatever the card happens to
# have, which on an idle 16 GB card came to ~8 GB and drove nvidia-smi to
# 13.7 GB mid-build -- inside the WSL2 spill zone and enough to make a shared
# desktop unusable. The cull already makes survivors scarce (~2.7k rays on the
# build frame, 0.3% of it), so a bigger budget buys no speed and only takes the
# card away from whatever else is using it. 2 GiB holds ~2.4k survivors against
# the shipped 5472-face droplet; more survivors simply take more chunks.
_MESH_CHUNK_MAX_BYTES = 2 << 30

# Held back from every budget for the CUDA context, cuBLAS workspaces and the
# caching allocator's slack. nvidia-smi routinely reads ~1 GB above torch's own
# max_memory_allocated, and on a shared node that gap is what stops one build
# from evicting a neighbour.
_VRAM_HEADROOM_BYTES = 1 << 30

# Resident cost of one ray in flight, measured on the tube scene at
# ~1.1 GiB per million rays (docs/DECISIONS.md 2026-07-31). Used only to pick a
# STARTING tile; the real bound comes from check_render_fits measuring a frame.
_BYTES_PER_RAY_RESIDENT = 1200

# Cost per OUTPUT pixel of the buffers no tile can shrink (the condenser
# accumulator, the ray grid, the PSF round-trip). Measured on a simulated 12 GB
# card by varying supersample alone: 5.94 / 7.14 / 8.83 GB at 14.4 / 32.4 /
# 57.5 Mpx is a slope of ~67 MB per Mpx. This is what makes a too-large render
# unfixable by tiling, and it is what turns a refusal into a NUMBER.
_BYTES_PER_PIXEL_UNTILEABLE = 70


def _mesh_survivor_chunk(faces, dev, vram_fraction=0.25):
    """How many AABB survivors TSurfaceMesh may push through Moller-Trumbore
    at once.

    Since 2026-08-11 the mesh bounds its own working set here instead of the
    caller shrinking the whole trace tile to suit it. The old arrangement made
    every ray in the frame pay the mesh's memory law even though the AABB cull
    now rejects ~99.7% of them before a single triangle is touched -- which
    forced ~133 passes over the build frame and cost 10x more than the
    brute-force intersection it was protecting against.

    A quarter of free VRAM, capped absolutely: this budget nests inside a caller
    that has already sized its own tile, so the two must not both claim the same
    headroom, and the cap keeps a long build from annexing the whole card.
    """
    if getattr(dev, "type", None) != "cuda" or faces <= 0:
        return 1 << 30
    free, _total = torch.cuda.mem_get_info()
    budget = min(free * vram_fraction, float(_MESH_CHUNK_MAX_BYTES))
    per_ray = faces * _MESH_BYTES_PER_RAY_FACE
    fits = int(budget // per_ray)
    # The floor is a PREFERENCE and must never override the budget. Applying it
    # unconditionally is what made a 50,976-face droplet reserve 16.1 GB and
    # spill: the budget asked for 263 rays, the floor forced 2048, and
    # 2048 x 50976 x 160 B = 16.7 GB. It crosses over at ~6,553 faces, which is
    # why meshes up to 5,472 never showed it. Below that many faces the floor
    # is free; above it, honour the budget and take more chunks instead.
    return max(1, min(fits, max(_MESH_CHUNK_MIN, fits)))


def fit_tile_size(tscene, total_rays, vram_fraction=0.80):
    """Largest tile that fits in free VRAM, computed rather than measured.

    Costs microseconds and runs no trial renders, so it is safe as a default:
    nothing here perturbs the memory counters the benchmark harnesses report,
    and nothing allocates the very block it is trying to avoid. plan_tile_size
    remains available for callers that explicitly ask for tile_size=None and
    want the measured answer.

    This tile bounds the ray-grid and condenser-accumulator terms, which scale
    with total_rays; it does not bound the mesh's own survivor-chunk memory
    (`_mesh_survivor_chunk`, 160 B per ray per face), which AABB-culls and
    chunks independently of this tile.

    This is a starting size, not a guarantee: the measured peak does not fit a
    clean linear model, so the guarantee comes from `check_render_fits`, which
    renders one frame and reads the real peak before a build commits.
    """
    if tscene.dev.type != "cuda":
        return total_rays
    affordable = int(memory_budget(vram_fraction) // _BYTES_PER_RAY_RESIDENT)
    return int(max(_TILE_FIT_MIN, min(total_rays, _TILE_DEFAULT, affordable)))


def memory_budget(vram_fraction=0.80):
    """Bytes this process may use, from what is ACTUALLY free right now.

    THE single budget authority -- every consumer sizes against this rather
    than reading `mem_get_info` itself, so there is one place to audit and one
    place to override.

    Not from `total_memory`: voltron is a shared 8-GPU node, so another tenant's
    allocation must reduce ours rather than being discovered as an OOM at
    frame 300 of 360. `_VRAM_HEADROOM_BYTES` is held back for the CUDA context
    and allocator slack -- `nvidia-smi` routinely reads a GB or more above
    torch's own `max_memory_allocated`, because the caching allocator reserves
    and never returns.

    `LOOPSIM_VRAM_BUDGET_GB` overrides the measurement. Two uses: capping a
    build so it leaves room for someone else on a shared card, and testing
    small-card behaviour on a large one -- `set_per_process_memory_fraction`
    cannot do the latter, because it constrains torch's allocator while
    `mem_get_info` keeps reporting the real device.
    """
    if not torch.cuda.is_available():
        return None
    override = os.environ.get("LOOPSIM_VRAM_BUDGET_GB")
    if override:
        return max(0.0, float(override) * 2**30 - _VRAM_HEADROOM_BYTES) * vram_fraction
    free, _total = torch.cuda.mem_get_info()
    return max(0, free - _VRAM_HEADROOM_BYTES) * vram_fraction


def install_vram_ceiling(vram_fraction=0.80):
    """Make the budget a HARD limit the allocator enforces, not advice.

    Without this the budget is only consulted by code that chooses to; anything
    that miscalculates sails past it and, on WSL2, spills to host RAM instead of
    failing -- a 10-50x slowdown that looks like a hang. With it, an overrun is
    a loud `torch.OutOfMemoryError` at the budget, which callers can catch,
    shrink and retry.

    Returns the ceiling in bytes, or None on CPU. Deliberately left INSTALLED:
    the build that follows a preflight needs the same protection the preflight
    had. `release_vram_ceiling()` undoes it.
    """
    if not torch.cuda.is_available():
        return None
    budget = memory_budget(vram_fraction)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), 0)
    return budget


def release_vram_ceiling():
    """Undo `install_vram_ceiling` (process-wide state; tests restore with it)."""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(1.0, 0)


class RenderTooLargeError(RuntimeError):
    """A render cannot fit this GPU at any tile size.

    Carries the arithmetic and a concrete suggestion, because the person who
    hits this is building a library on a beamline node and needs to know what
    to change, not that something was too big.
    """


def check_render_fits(tscene, n_cond=7, psf=True,
                      vram_fraction=0.80, supersample=None, progress=None):
    """Render ONE frame and verify the real peak fits, before a build commits.

    This is the guarantee, and it is deliberately a MEASUREMENT rather than a
    model: peak memory here comes from several terms (mesh temporaries, the
    resident ray arrays, and O(W x H) buffers that no tile shrinks) whose sum
    did not fit a clean linear fit when measured, so a predictive formula would
    be a guess with a safety factor. One frame costs seconds against a build
    that costs hours.

    Shrinks the tile and retries while that can help. Raises
    `RenderTooLargeError` when it cannot -- with the largest `--supersample`
    that would fit, derived by scaling the measured peak.
    """
    if tscene.dev.type != "cuda":
        return None
    from ..motors.goniometer import Goniometer
    # Enforce the budget before measuring against it: the probe below RENDERS,
    # so without a hard ceiling the measurement itself can overshoot -- on the
    # dev box it spilled to 13.8 GB while "checking" whether 12 GB was enough.
    # Left installed on success so the build inherits the same protection.
    budget = install_vram_ceiling(vram_fraction)
    # Dimensions come from the SCENE, never from arguments. An earlier version
    # took width/height as parameters and never applied them -- it rendered at
    # whatever the scene's camera said and reported the caller's numbers, so a
    # 40000x20000 request measured a 640x480 frame at 0.20 GB and cheerfully
    # approved it. The probe must measure the render that is actually about to
    # run, so there is exactly one source of truth for its size.
    cam = tscene.scene.camera_cfg
    width, height = int(cam["width"]), int(cam["height"])
    total_rays = width * height
    tile = fit_tile_size(tscene, total_rays, vram_fraction)
    gono = Goniometer(tscene.scene.geometry)

    def probe(t):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            render_torch(tscene, gono, n_cond=n_cond, psf=psf, tile_size=t)
            return torch.cuda.max_memory_allocated()
        except torch.OutOfMemoryError:
            return float("inf")

    # At most TWO probes, not a halving ladder. Each probe renders the whole
    # frame, so a ladder costs a full render per rung -- measured at 421 s for
    # eight rungs on a 129 Mpx frame, all of them doomed. The second probe goes
    # straight to the smallest tile: if the untileable O(W x H) terms already
    # exceed the budget there, no intermediate tile can help either.
    peak = probe(tile)
    if peak > budget and tile > _TILE_FIT_MIN:
        # Scale the tile by how far over we were, rather than dropping to the
        # floor. The cost curve is steep at small tiles and flat at large ones
        # (measured at 14.4 Mpx: 1M->6M tiles span 18.6->17.1 s, only 8%, while
        # 133 tiny passes cost 10x on a 0.9 Mpx frame), so overshooting
        # downward turns a memory problem into a speed problem. An OOM gives no
        # peak to scale from, so that case halves instead.
        if peak == float("inf"):
            nxt = tile // 2
        else:
            nxt = int(tile * (budget / peak) * 0.9)
        tile = max(_TILE_FIT_MIN, min(tile - 1, nxt))
        if progress:
            progress(f"[preflight] over budget; retrying at tile {tile}")
        peak = probe(tile)
        if peak > budget and tile > _TILE_FIT_MIN:
            tile = _TILE_FIT_MIN          # last resort before refusing
            peak = probe(tile)
    if peak <= budget:
        if progress:
            progress(f"[preflight] {width}x{height} n_cond={n_cond}: "
                     f"{peak/2**30:.2f} GB peak against a {budget/2**30:.2f} GB "
                     f"budget, tile {tile} -- fits")
        torch.cuda.empty_cache()
        return tile
    # Even the smallest tile does not fit: the untileable O(W x H) terms
    # dominate, so the only remedy is fewer output pixels.
    torch.cuda.empty_cache()
    msg = (f"render of {width}x{height} ({total_rays/1e6:.1f} Mpx, "
           f"n_cond={n_cond}) needs more memory than this GPU has: peak "
           f"{peak/2**30:.2f} GB against a {budget/2**30:.2f} GB budget "
           f"(free VRAM minus headroom, x{vram_fraction}). Tiling cannot "
           f"help -- the cost that does not fit scales with OUTPUT PIXELS, "
           f"which no tile size reduces.")
    if supersample and supersample > 1:
        if peak != float("inf"):
            ok = max(1, int(supersample * (budget / peak) ** 0.5))
            msg += (f" At this scene's settings --supersample {ok} is the "
                    f"largest that fits (you asked for {supersample}).")
        else:
            # OOM gives no peak to scale from, so fall back on the measured
            # per-output-pixel cost. Pixels scale as supersample^2.
            # 0.75 because this constant is a floor on the true per-pixel
            # cost (it omits the scene's own residency), and a suggestion that
            # still does not fit is worse than none. Clamped strictly below the
            # request: an estimate that equals what we are refusing is wrong on
            # its face, and the first version printed exactly that.
            max_px = 0.75 * budget / _BYTES_PER_PIXEL_UNTILEABLE
            ok = max(1, int(supersample * (max_px / max(1, total_rays)) ** 0.5))
            ok = min(ok, supersample - 1)
            msg += (f" At this scene's settings --supersample {ok} is the "
                    f"largest that fits (you asked for {supersample}).")
    raise RenderTooLargeError(msg)


def _probe_peak(fn):
    """Marginal bytes `fn` adds to the allocator's RESERVED pool.

    Reserved, not allocated: the card is filled by what the caching allocator
    holds, and a fragmenting workload (the mesh path, whose Moller-Trumbore
    temporaries vary in size with the AABB survivor count) reserves
    substantially more than it allocates. Budgeting against `allocated`
    under-counts that gap and lets a mesh scene sail past the card's capacity --
    which under WSL2 means a silent spill to host RAM, not an OOM.

    NOTE: this RESETS the process-global peak-memory counters. Torch offers no
    way to restore them, so anything reporting peak memory around a render
    (`bench_frame.py`, `acceptance_voltron.py`) must not run while a calibration
    is happening. Both are safe because they take the default tile size
    (`fit_tile_size`, which calculates and never probes), not because they pass
    an explicit `tile_size` -- neither does. That is one of the reasons the
    probing ramp cannot be made the default; changing it back would corrupt the
    VRAM figures in the TITAN V GO/NO-GO harness with no test to catch it.
    """
    torch.cuda.empty_cache()          # start from a clean pool so the delta is real
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_reserved()
    fn()
    torch.cuda.synchronize()
    return max(torch.cuda.max_memory_reserved() - before, 0)


def plan_tile_size(tscene, o_t, d_t, na_obj, opt_axis_s, total_rays,
                   vram_fraction=0.80):
    """
    Choose a trace tile size that fits in free VRAM with headroom.

    Measures by a doubling ramp rather than extrapolating a slope from two tiny
    probes. Extrapolation was the wrong tool: fitting on 64k rays and solving
    for ~10M is a 150x stretch, and on the mesh path -- where the
    Moller-Trumbore temporaries fragment the pool -- it under-predicted by
    ~2.7 GB and sailed the render straight past the card's capacity. Under WSL2
    that is not an OOM, it is a silent spill to host RAM and a 10x slowdown.

    Each rung is *measured* and the ramp stops before a rung that would exceed
    the budget, so no probe can itself trigger the failure it is sizing to
    avoid. Returns `total_rays` unchanged on CPU (no device memory to budget).
    """
    if tscene.dev.type != "cuda":
        return total_rays

    free, _total = torch.cuda.mem_get_info()
    budget = free * vram_fraction

    def _sample(n):
        """n rays spread across the whole frame, not a contiguous prefix.

        The first n rays of a wide frame are its top few rows, which for most
        scenes is pure background: they die at the first depth and understate
        the per-ray cost. The stride is recomputed per rung -- a single fixed
        stride caps the sample at one size, and then every larger rung measures
        the same rays, reports a flat cost, and the ramp doubles to the top.
        """
        stride = max(1, total_rays // n)
        return o_t[::stride][:n], d_t[::stride][:n]

    best = _TILE_MIN
    n = _TILE_MIN
    while n <= total_rays:
        o_s, d_s = _sample(n)
        if o_s.shape[0] < n:           # cannot actually assemble this rung
            break
        try:
            cost = _probe_peak(
                lambda: tscene.trace_rays(o_s, d_s, na_obj, opt_axis_s))
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
        if cost > budget:
            break
        best = n
        if cost * 2.0 > budget or n >= total_rays:
            break                      # the next rung would not fit
        n *= 2

    torch.cuda.empty_cache()
    return max(_TILE_MIN, min(best, total_rays))


# ---------------------------------------------------------------------------
# render: torch port of microscope.render. Reuses the numpy ray-grid/condenser
# setup verbatim (tiny, per-frame) so any divergence is isolated to the trace.
# Returns an (H, W, 3) torch tensor in [0, 1].
#
# Condenser samples are traced one at a time and accumulated, so the resident
# ray arrays are W*H rather than n_cond*W*H -- peak memory is independent of
# n_cond. Within a sample the trace is tiled, which bounds the TRACE working
# set at any resolution; note the resident arrays (o_t, d_t, accum) are still
# O(W*H) and tiling cannot shrink them -- about 1 GB at a 14 Mpx template --
# so peak memory is reduced by tiling, not made resolution-independent.
# Per-ray results are tile-independent (verified byte-exact down to 1000-ray
# tiles), which is what makes both safe.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def render_torch(tscene, goniometer, n_cond=1, tile_size="fit",
                 compiled=False, vram_fraction=0.80, psf=True):
    """Render one frame.

    tile_size: "fit" (default) sizes the trace tile from the scene's mesh size
    and free VRAM by calculation -- no trial renders, microseconds, and for a
    scene with no mesh it is exactly the old flat _TILE_DEFAULT. None/"auto"
    uses plan_tile_size's measured doubling ramp instead; an int is used as-is.
    """
    from .microscope import _condenser_offsets
    from ..motors.goniometer import apply_transform
    from .optics import apply_psf, psf_sigma_px, MIN_SIGMA_PX

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

    offsets = _condenser_offsets(n_cond, na_cond)
    WH = W * H
    accum = torch.zeros((WH, 3), device=dev, dtype=dt)
    plan = tile_size          # None/"auto" -> calibrate against free VRAM once

    for k in range(n_cond):
        ox, oy, _ = offsets[k]
        illum_dir = opt_axis + ox * fast + oy * slow
        illum_dir /= np.linalg.norm(illum_dir)
        illum_dir_s = T_inv[:3, :3] @ illum_dir
        illum_dir_s /= np.linalg.norm(illum_dir_s) + 1e-30
        t_up = 50.0 / max(abs(np.dot(illum_dir, opt_axis)), 1e-6)

        o_t = torch.as_tensor(focal_pts_s - t_up * illum_dir_s, device=dev, dtype=dt)
        d_t = torch.as_tensor(np.broadcast_to(illum_dir_s, (WH, 3)).copy(),
                              device=dev, dtype=dt)

        if plan == "fit":
            # Calculated, not probed: cheap enough to redo per condenser sample
            # and it never allocates the block it is sizing against.
            plan = fit_tile_size(tscene, WH, vram_fraction=vram_fraction)
        elif plan is None or plan == "auto":
            # Cache per (scene, frame size, headroom): calibrating costs two
            # probe traces, and a 360-frame sweep would otherwise pay for them
            # 360 times over.
            key = (WH, round(float(vram_fraction), 3))
            plan = getattr(tscene, "_tile_plan", {}).get(key)
            if plan is None:
                plan = plan_tile_size(tscene, o_t, d_t, na_obj, opt_axis_s, WH,
                                      vram_fraction=vram_fraction)
                if not hasattr(tscene, "_tile_plan"):
                    tscene._tile_plan = {}
                tscene._tile_plan[key] = plan
        step = max(int(plan), 1)

        for s in range(0, WH, step):
            e = min(s + step, WH)
            accum[s:e] += tscene.trace_rays(o_t[s:e], d_t[s:e], na_obj, opt_axis_s,
                                            compiled=compiled)
        del o_t, d_t

    # average over the condenser dimension (sequential sum / n, matching the
    # numpy reference in microscope.render)
    out = (accum / n_cond).reshape(H, W, 3)

    # Objective PSF.  Deliberately a numpy round-trip through the SAME helper
    # microscope.render uses: the two renders are asserted byte-identical after
    # quantisation, and a separate device-side convolution would diverge in
    # kernel truncation, normalisation and summation order.  The caller
    # transfers this result to the host immediately anyway (to encode a JPEG),
    # so the extra sync costs little in the paths that matter.  Skipped entirely
    # when the resolved sigma is sub-threshold, so coarse renders are untouched.
    if psf:
        sigma = psf_sigma_px(cam, eff_px)
        if sigma >= MIN_SIGMA_PX:
            blurred = apply_psf(out.detach().cpu().numpy(), sigma)
            out = torch.as_tensor(blurred, dtype=dt, device=dev)
    return out

# render_xray_torch / trace_xray moved to renderer/xray_torch.py (2026-08-18) --
# they never contributed a template pixel and don't belong in this file's
# _RENDER_SOURCES hash. See that module's docstring and docs/DECISIONS.md.
