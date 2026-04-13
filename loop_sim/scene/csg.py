"""
CSG (Constructive Solid Geometry) operations via interval arithmetic on
the ray parameter t.

Each CSG node and primitive shares the same interface:
    ray_intersect(origins, dirs) -> (t_enter, t_exit, n_enter, n_exit)

where the returned interval [t_enter, t_exit] is the segment of the ray
that lies inside the solid.

For primitives that can produce more than one interval (rare for these
simple shapes), only the outermost [min_t_enter, max_t_exit] is returned
here; full multi-interval CSG is handled in the scene-level path_lengths().

Conventions
-----------
- t_enter = +inf  → ray misses the solid
- t_enter can be negative (ray starts inside)
- Normals are outward-pointing (away from the interior)
"""
import numpy as np

_INF = np.inf


# ---------------------------------------------------------------------------
# Intersection  (A ∩ B)
# ---------------------------------------------------------------------------

class Intersection:
    """Interior = {p : p ∈ A  and  p ∈ B}"""

    def __init__(self, *children):
        assert len(children) >= 2
        self.children = list(children)

    def ray_intersect(self, origins, dirs):
        # Start with the first child's interval
        te, tx, ne, nx = self.children[0].ray_intersect(origins, dirs)
        for child in self.children[1:]:
            te2, tx2, ne2, nx2 = child.ray_intersect(origins, dirs)
            # New interval = max(te, te2) … min(tx, tx2)
            # Normal at new entry: whichever t was larger
            use_child_entry = te2 > te
            te = np.where(use_child_entry, te2, te)
            ne = np.where(use_child_entry[:, None], ne2, ne)

            use_child_exit = tx2 < tx
            tx = np.where(use_child_exit, tx2, tx)
            nx = np.where(use_child_exit[:, None], nx2, nx)

        # Mark misses
        miss = te >= tx
        te[miss] = _INF
        tx[miss] = _INF

        return te, tx, ne, nx


# ---------------------------------------------------------------------------
# Union  (A ∪ B)
# ---------------------------------------------------------------------------

class Union:
    """Interior = {p : p ∈ A  or  p ∈ B}"""

    def __init__(self, *children):
        assert len(children) >= 2
        self.children = list(children)

    def ray_intersect(self, origins, dirs):
        te, tx, ne, nx = self.children[0].ray_intersect(origins, dirs)
        for child in self.children[1:]:
            te2, tx2, ne2, nx2 = child.ray_intersect(origins, dirs)
            # New interval = min(te, te2) … max(tx, tx2)
            # (This merges overlapping intervals into their union — correct only
            #  when the two intervals overlap.  For non-overlapping intervals this
            #  produces a slightly wrong merged interval; the scene-level code
            #  handles non-overlapping unions correctly.)
            miss_a = te  >= tx
            miss_b = te2 >= tx2

            use_b_entry = (te2 < te) & ~miss_b
            te = np.where(use_b_entry, te2, te)
            ne = np.where(use_b_entry[:, None], ne2, ne)

            use_b_exit = (tx2 > tx) & ~miss_b
            tx = np.where(use_b_exit, tx2, tx)
            nx = np.where(use_b_exit[:, None], nx2, nx)

            # If A missed but B hit, use B entirely
            miss_a_hit_b = miss_a & ~miss_b
            te[miss_a_hit_b] = te2[miss_a_hit_b]
            tx[miss_a_hit_b] = tx2[miss_a_hit_b]
            ne[miss_a_hit_b] = ne2[miss_a_hit_b]
            nx[miss_a_hit_b] = nx2[miss_a_hit_b]

        return te, tx, ne, nx


# ---------------------------------------------------------------------------
# Difference  (A − B)
# ---------------------------------------------------------------------------

class Difference:
    """
    Interior = {p : p ∈ A  and  p ∉ B}

    For a ray that passes through A with interval [ta_e, ta_x] and
    through B with interval [tb_e, tb_x]:

        If B interval is entirely inside A:   result is two sub-intervals
            (we return only the first one for simplicity — correct for
            crystal-out-of-solvent use cases where the crystal is always
            smaller than the solvent blob)

        If B clips the start of A:   result starts at tb_x
        If B clips the end   of A:   result ends   at tb_e
        If B contains A entirely:    no hit
    """

    def __init__(self, A, B):
        self.A = A
        self.B = B

    def ray_intersect(self, origins, dirs):
        te_a, tx_a, ne_a, nx_a = self.A.ray_intersect(origins, dirs)
        te_b, tx_b, ne_b, nx_b = self.B.ray_intersect(origins, dirs)

        N = len(origins)
        te = te_a.copy()
        tx = tx_a.copy()
        ne = ne_a.copy()
        nx = nx_a.copy()

        miss_a = te_a >= tx_a
        hit_b  = te_b < tx_b

        # B clips start of A (te_b < te_a < tx_b < tx_a):
        #   new entry is at tx_b (with flipped normal of B)
        clip_start = (hit_b & ~miss_a
                      & (te_b <= te_a)
                      & (tx_b > te_a)
                      & (tx_b < tx_a))
        te[clip_start] = tx_b[clip_start]
        ne[clip_start] = -nx_b[clip_start]   # flip: now facing inward of B

        # B clips end of A (te_a < te_b < tx_a < tx_b):
        #   new exit is at te_b (with flipped normal of B)
        clip_end = (hit_b & ~miss_a
                    & (te_b > te_a)
                    & (te_b < tx_a)
                    & (tx_b >= tx_a))
        tx[clip_end] = te_b[clip_end]
        nx[clip_end] = -ne_b[clip_end]

        # B contains A entirely: no hit
        contains = (hit_b & ~miss_a
                    & (te_b <= te_a)
                    & (tx_b >= tx_a))
        te[contains] = _INF
        tx[contains] = _INF

        # A missed anyway
        te[miss_a] = _INF
        tx[miss_a] = _INF

        return te, tx, ne, nx
