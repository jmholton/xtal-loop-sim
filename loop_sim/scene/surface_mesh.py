"""
Triangulated surface mesh with Möller-Trumbore ray-triangle intersection,
vectorized over N rays using NumPy broadcasting.

The mesh represents a closed surface.  Ray intersection returns the entry
and exit t values (nearest and farthest crossings) and corresponding outward
normals.

For large meshes (thousands of triangles) an optional uniform-grid spatial
index reduces the per-ray triangle tests from O(T) to O(T/cell) on average.

GPU acceleration
----------------
Pass ``device='cuda'`` to use PyTorch on the GPU for the hot-path
Möller-Trumbore batch computation.  The external interface (numpy arrays
in/out) is unchanged; conversion happens inside ``_mt_batch_cuda``.
Batch size is automatically increased to 65536 on CUDA (vs 2048 on CPU).
"""
import numpy as np

_INF = np.inf
_EPS = 1e-8
_BATCH_CPU  = 2048
_BATCH_CUDA = 32768


class SurfaceMesh:
    """
    Parameters
    ----------
    vertices : array-like, shape (V, 3)
    faces    : array-like, shape (F, 3) — integer indices into vertices
    use_grid : bool
        Build a uniform spatial grid for faster intersection.  Recommended
        for meshes with > 500 triangles.  Ignored when device is CUDA
        (the GPU batch handles all triangles at once).
    device : str
        'cpu' (default, numpy) or 'cuda' (PyTorch GPU).
    """

    def __init__(self, vertices, faces, use_grid=True, device='cpu'):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces    = np.asarray(faces,    dtype=int)
        self._device  = device
        self._precompute()
        self._grid = None
        if device == 'cpu' and use_grid and len(self.faces) > 200:
            self._build_grid()

    def _precompute(self):
        v0 = self.vertices[self.faces[:, 0]]   # (F, 3)
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        self._v0 = v0
        self._e1 = v1 - v0    # (F, 3)
        self._e2 = v2 - v0
        # Face normals (not necessarily unit length here)
        N = np.cross(self._e1, self._e2)
        nlen = np.linalg.norm(N, axis=1, keepdims=True)
        self._face_normals = N / (nlen + 1e-30)
        # AABB for fast pre-filter
        self._bbox_lo = self.vertices.min(axis=0)
        self._bbox_hi = self.vertices.max(axis=0)

        # Pre-convert geometry to CUDA tensors for the GPU path
        if self._device != 'cpu':
            import torch
            dev = torch.device(self._device)
            def _t(a):
                return torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)
            self._v0_t = _t(self._v0)
            self._e1_t = _t(self._e1)
            self._e2_t = _t(self._e2)
            self._fn_t = _t(self._face_normals)

    def _build_grid(self, n_cells=20):
        """Build a uniform grid; store per-cell triangle lists."""
        lo = self.vertices.min(axis=0) - 1e-6
        hi = self.vertices.max(axis=0) + 1e-6
        self._grid_lo = lo
        self._grid_hi = hi
        self._grid_n  = n_cells
        cell_size = (hi - lo) / n_cells
        self._grid_cell = cell_size

        from collections import defaultdict
        grid = defaultdict(list)
        tri_lo = np.minimum(self._v0, np.minimum(self._v0 + self._e1,
                                                  self._v0 + self._e2))
        tri_hi = np.maximum(self._v0, np.maximum(self._v0 + self._e1,
                                                  self._v0 + self._e2))
        for fi in range(len(self.faces)):
            i0 = np.floor((tri_lo[fi] - lo) / cell_size).astype(int).clip(0, n_cells - 1)
            i1 = np.floor((tri_hi[fi] - lo) / cell_size).astype(int).clip(0, n_cells - 1)
            for ix in range(i0[0], i1[0] + 1):
                for iy in range(i0[1], i1[1] + 1):
                    for iz in range(i0[2], i1[2] + 1):
                        grid[(ix, iy, iz)].append(fi)
        self._grid = {k: np.array(v) for k, v in grid.items()}

    # ------------------------------------------------------------------
    # Core intersection: one ray against all (or subset of) triangles
    # ------------------------------------------------------------------

    def _intersect_ray_triangles(self, origin, direction, face_indices=None):
        """
        Möller-Trumbore: one ray against a set of triangles.
        Returns sorted (t, face_idx) array of all hits.
        """
        if face_indices is None:
            v0 = self._v0
            e1 = self._e1
            e2 = self._e2
            fi = np.arange(len(self.faces))
        else:
            fi = face_indices
            v0 = self._v0[fi]
            e1 = self._e1[fi]
            e2 = self._e2[fi]

        h = np.cross(direction, e2)             # (F, 3)
        a = np.einsum("fj,fj->f", e1, h)        # (F,)
        parallel = np.abs(a) < _EPS
        inv_a = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, a))

        s = origin - v0                          # (F, 3)
        u = inv_a * np.einsum("fj,fj->f", s, h)
        miss = parallel | (u < 0.0) | (u > 1.0)

        q = np.cross(s, e1)
        v = inv_a * np.einsum("j,fj->f", direction, q)
        miss |= (v < 0.0) | (u + v > 1.0)

        t = inv_a * np.einsum("fj,fj->f", e2, q)
        miss |= (t < _EPS)

        hits = ~miss
        if not np.any(hits):
            return np.empty(0), np.empty(0, int)
        t_hit = t[hits]
        fi_hit = fi[hits]
        order = np.argsort(t_hit)
        return t_hit[order], fi_hit[order]

    # ------------------------------------------------------------------
    # GPU batch: Möller-Trumbore on CUDA via PyTorch
    # ------------------------------------------------------------------

    def _mt_batch_cuda(self, o, d):
        """
        PyTorch/CUDA Möller-Trumbore for B rays vs F triangles.
        Accepts and returns numpy arrays; conversion happens internally.
        Returns the same 6-tuple as the numpy _mt_batch.
        """
        import torch
        dev = self._v0_t.device
        ot = torch.from_numpy(np.ascontiguousarray(o)).float().to(dev)  # (B, 3)
        dt = torch.from_numpy(np.ascontiguousarray(d)).float().to(dev)  # (B, 3)
        v0, e1, e2 = self._v0_t, self._e1_t, self._e2_t
        B, F = len(o), len(v0)
        INF = float('inf')

        # h[b,f] = cross(d[b], e2[f])  →  (B, F, 3)
        h = torch.linalg.cross(
            dt.unsqueeze(1).expand(-1, F, -1),
            e2.unsqueeze(0).expand(B, -1, -1), dim=2)
        a = (e1.unsqueeze(0) * h).sum(-1)          # (B, F)

        par  = a.abs() < _EPS
        safe = torch.where(par, torch.ones_like(a), a)
        inv_a = torch.where(par, torch.zeros_like(a), 1.0 / safe)

        s = ot.unsqueeze(1) - v0.unsqueeze(0)       # (B, F, 3)
        u = inv_a * (s * h).sum(-1)                  # (B, F)

        q = torch.linalg.cross(
            s, e1.unsqueeze(0).expand(B, -1, -1), dim=2)   # (B, F, 3)
        v = inv_a * (dt.unsqueeze(1) * q).sum(-1)   # (B, F)
        t = inv_a * (e2.unsqueeze(0) * q).sum(-1)   # (B, F)

        miss = par | (u < 0) | (u > 1) | (v < 0) | ((u + v) > 1)

        # Forward hits: t > _EPS
        t_fwd   = torch.where(miss | (t <  _EPS), torch.full_like(t,  INF), t)
        t_fwd_r = torch.where(miss | (t <  _EPS), torch.full_like(t, -INF), t)
        # Backward hit: t < -_EPS  (stored as positive magnitude)
        t_bwd   = torch.where(miss | (t > -_EPS), torch.full_like(t,  INF), -t)

        bi     = torch.arange(B, device=dev)
        fi_min = t_fwd.argmin(1)
        fi_max = t_fwd_r.argmax(1)
        fi_bwd = t_bwd.argmin(1)

        t_min   = t_fwd  [bi, fi_min]
        t_max   = t_fwd_r[bi, fi_max]
        has_bwd = t_bwd[bi, fi_bwd] < INF
        t_back  = torch.where(has_bwd, -t_bwd[bi, fi_bwd],
                              torch.zeros(B, device=dev))

        return (t_min.cpu().numpy(),  t_max.cpu().numpy(),
                fi_min.cpu().numpy(), fi_max.cpu().numpy(),
                t_back.cpu().numpy(), fi_bwd.cpu().numpy())

    # ------------------------------------------------------------------
    # Vectorized Möller-Trumbore: batch of rays vs all triangles (CPU)
    # ------------------------------------------------------------------

    def _mt_batch(self, o, d):
        """
        Möller-Trumbore for B rays vs F triangles.
        Dispatches to CUDA implementation if device is set.

        Returns
        -------
        t_min  : (B,) nearest  positive t (inf  = no forward hit)
        t_max  : (B,) farthest positive t (-inf = no forward hit)
        fi_min : (B,) face index for t_min
        fi_max : (B,) face index for t_max
        t_back : (B,) closest  negative t (0    = no backward hit)
        fi_bwd : (B,) face index for t_back
        """
        if self._device != 'cpu':
            return self._mt_batch_cuda(o, d)

        v0 = self._v0    # (F, 3)
        e1 = self._e1    # (F, 3)
        e2 = self._e2    # (F, 3)
        B  = len(o)

        # h[b,f] = cross(d[b], e2[f])
        h = np.cross(d[:, None, :], e2[None, :, :])          # (B, F, 3)
        a = (e1[None, :, :] * h).sum(axis=-1)                 # (B, F)

        parallel = np.abs(a) < _EPS
        inv_a = np.where(parallel, 0.0, 1.0 / np.where(parallel, 1.0, a))

        s = o[:, None, :] - v0[None, :, :]                   # (B, F, 3)
        u = inv_a * (s * h).sum(axis=-1)                      # (B, F)

        q = np.cross(s, e1[None, :, :])                       # (B, F, 3)
        v = inv_a * (d[:, None, :] * q).sum(axis=-1)          # (B, F)
        t = inv_a * (e2[None, :, :] * q).sum(axis=-1)         # (B, F)

        # Geometric miss: UV test and degenerate triangle (t-independent)
        miss_geom = parallel | (u < 0) | (u > 1) | (v < 0) | (u + v > 1)

        # Forward hits: t > _EPS
        t_fwd   = np.where(miss_geom | (t < _EPS),  _INF, t)    # (B, F)
        t_fwd_r = np.where(miss_geom | (t < _EPS), -_INF, t)    # (B, F)

        # Backward hit: t < -_EPS (closest to 0, stored as |t|)
        t_bwd   = np.where(miss_geom | (t > -_EPS), _INF, -t)   # (B, F) positive mag

        bi = np.arange(B)
        fi_min = t_fwd.argmin(axis=1)                            # (B,)
        fi_max = t_fwd_r.argmax(axis=1)                          # (B,)
        fi_bwd = t_bwd.argmin(axis=1)                            # (B,)

        t_min  = t_fwd  [bi, fi_min]                             # (B,)
        t_max  = t_fwd_r[bi, fi_max]                             # (B,) or -inf
        has_bwd = t_bwd[bi, fi_bwd] < _INF
        t_back  = np.where(has_bwd, -t_bwd[bi, fi_bwd], 0.0)    # (B,) negative or 0

        return t_min, t_max, fi_min, fi_max, t_back, fi_bwd

    # ------------------------------------------------------------------
    # Public ray_intersect — AABB pre-filter + vectorized MT
    # ------------------------------------------------------------------

    def ray_intersect(self, origins, dirs):
        """
        Test N rays against the mesh.  Returns entry and exit intervals.

        For a closed mesh the ray alternates between entering and exiting
        the enclosed volume.  We return:
            t_enter : nearest   crossing (smallest  positive t)
            t_exit  : farthest  crossing (largest   positive t)
            n_enter : outward face normal at entry
            n_exit  : outward face normal at exit
        """
        N = len(origins)
        t_enter = np.full(N, _INF)
        t_exit  = np.full(N, _INF)
        n_enter = np.zeros((N, 3))
        n_exit  = np.zeros((N, 3))

        # --- AABB pre-filter (same correct slab test as Tube) ---
        lo = self._bbox_lo
        hi = self._bbox_hi
        parallel = np.abs(dirs) < 1e-12
        nonpar   = ~parallel
        safe_inv = np.where(nonpar, 1.0 / np.where(nonpar, dirs, 1.0), 0.0)
        t1 = (lo - origins) * safe_inv
        t2 = (hi - origins) * safe_inv
        t_near = np.where(nonpar, np.minimum(t1, t2), -_INF)
        t_far  = np.where(nonpar, np.maximum(t1, t2),  _INF)
        par_out = parallel & ((origins < lo) | (origins > hi))
        t_near  = np.where(par_out,  _INF, t_near)
        t_far   = np.where(par_out, -_INF, t_far)
        t_in    = t_near.max(axis=1)
        t_out   = t_far.min(axis=1)
        aabb    = (t_out > 0.0) & (t_in < t_out)

        if not np.any(aabb):
            return t_enter, t_exit, n_enter, n_exit

        idx = np.where(aabb)[0]
        o_f = origins[idx]    # (M, 3)
        d_f = dirs[idx]       # (M, 3)

        # Batch size depends on device
        _BATCH = _BATCH_CUDA if self._device != 'cpu' else _BATCH_CPU
        M = len(idx)
        t_min_f  = np.full(M,  _INF)
        t_max_f  = np.full(M, -_INF)
        fi_min_f = np.zeros(M, dtype=int)
        fi_max_f = np.zeros(M, dtype=int)
        t_back_f = np.zeros(M)          # closest backward t (≤ 0); 0 = none
        fi_bwd_f = np.zeros(M, dtype=int)

        for s in range(0, M, _BATCH):
            e = min(s + _BATCH, M)
            tm, tx, fm, fx, tb, fb = self._mt_batch(o_f[s:e], d_f[s:e])
            t_min_f[s:e]  = tm
            t_max_f[s:e]  = tx
            fi_min_f[s:e] = fm
            fi_max_f[s:e] = fx
            t_back_f[s:e] = tb
            fi_bwd_f[s:e] = fb

        has_fwd  = t_min_f < _INF
        has_bwd  = t_back_f < 0.0

        # "Inside" case: ray origin is inside the closed mesh.
        inside = has_fwd & has_bwd & (t_min_f >= t_max_f)

        # Normal "outside → through" case: two distinct forward crossings
        outside = has_fwd & ~inside

        if not np.any(has_fwd | inside):
            return t_enter, t_exit, n_enter, n_exit

        # ---- Build combined entry arrays ----
        te_all = np.where(inside, t_back_f, t_min_f)   # (M,)  te < 0 when inside
        tx_all = np.where(inside, t_min_f,  t_max_f)   # (M,)  tx = exit

        # For "outside" rays with single forward hit (grazing): tx = te
        grazing = outside & (t_min_f >= t_max_f)
        tx_all  = np.where(grazing, t_min_f, tx_all)

        valid = inside | outside
        v_g   = idx[valid]
        te_v  = te_all[valid]
        tx_v  = tx_all[valid]

        t_enter[v_g] = te_v
        t_exit[v_g]  = tx_v

        # ---- Entry normals ----
        fi_e = np.where(inside[valid], fi_bwd_f[valid], fi_min_f[valid])
        fn_e = self._face_normals[fi_e]
        cos_e = (d_f[valid] * fn_e).sum(axis=1)
        n_enter[v_g] = np.where(cos_e[:, None] < 0, fn_e, -fn_e)

        # ---- Exit normals ----
        fi_x = np.where(inside[valid], fi_min_f[valid], fi_max_f[valid])
        fn_x = self._face_normals[fi_x]
        cos_x = (d_f[valid] * fn_x).sum(axis=1)
        n_exit[v_g] = np.where(cos_x[:, None] > 0, fn_x, -fn_x)

        return t_enter, t_exit, n_enter, n_exit

    @property
    def bounding_box(self):
        return self._bbox_lo, self._bbox_hi
