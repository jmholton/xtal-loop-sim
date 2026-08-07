"""
Mesh-back validation of generated scenes.

Measures the geometry actually EMITTED into a scene dict/YAML — never the
inputs that produced it — and fails loudly when it does not match what was
asked for.  Born of the silent-hemisphere episode (docs/DECISIONS.md
2026-08-07): the droplet solver failed for ordinary inputs and shipped a
plausible-looking fallback through a fidelity audit, precisely because
nothing measured the mesh back.

Checks (droplet scenes):
  - the solvent mesh is watertight (every directed edge exactly once,
    every undirected edge exactly twice) with no zero-area triangles
  - signed volume (divergence theorem) is positive and within tolerance
    of the requested volume
  - the rim is pinned on the loop fiber: every rim vertex lies within one
    fiber diameter of the loop path
  - the droplet straddles the loop plane (bulges on both sides)
  - the crystal is centred in the aperture, and is listed before the
    solvent (priority order — a crystal behind the droplet is invisible)

Failures raise SceneValidationError naming every failed check.  Soft
concerns (e.g. the crystal poking out of a thin droplet, which a real
mount does but this renderer shows with hard optical interfaces) go into
report["warnings"].

Usage
-----
    from crystal_harvester.validate import validate_scene

    report = validate_scene(scene_dict, requested_volume_mm3=0.002)
    print(report)          # measured volume, rim stats, warnings, ...
"""

import numpy as np
import yaml

from .droplet import mesh_volume_centroid


class SceneValidationError(ValueError):
    """The emitted scene geometry does not match what was requested."""


# ---------------------------------------------------------------------------
# Mesh measurements
# ---------------------------------------------------------------------------


def _watertight_failures(faces):
    """Return a list of watertightness defects (empty = closed, consistent)."""
    directed = {}
    for tri in faces:
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            directed[e] = directed.get(e, 0) + 1
    out = []
    dup = sum(1 for n in directed.values() if n != 1)
    if dup:
        out.append(f"{dup} directed edges repeat — inconsistent face winding")
    undirected = {}
    for (a, b), n in directed.items():
        k = (a, b) if a < b else (b, a)
        undirected[k] = undirected.get(k, 0) + n
    open_e = sum(1 for n in undirected.values() if n != 2)
    if open_e:
        out.append(f"{open_e} edges not shared by exactly 2 faces — mesh not closed")
    return out


def _point_polyline_dist(points, poly):
    """Min distance from each point (N,3) to a closed polyline (M,3)."""
    p0 = poly
    p1 = np.vstack([poly[1:], poly[:1]])
    seg = p1 - p0                                          # (M, 3)
    seg_len2 = np.maximum((seg ** 2).sum(axis=1), 1e-30)
    d = points[:, None, :] - p0[None, :, :]                # (N, M, 3)
    t = np.clip(np.einsum("nmj,mj->nm", d, seg) / seg_len2, 0.0, 1.0)
    closest = p0[None, :, :] + t[:, :, None] * seg[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def _fit_plane(points):
    """Best-fit plane of a point set: (centroid, unit normal)."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c)
    return c, vt[-1]


def _crystal_centre(children):
    """Least-squares centre of a half-space intersection.

    Pairs faces with opposite normals; each pair (n, o+), (-n, o-) constrains
    n . c = (o+ - o-) / 2.  Returns None if fewer than 3 independent pairs.
    """
    normals = np.array([hs["normal"] for hs in children], dtype=float)
    offsets = np.array([hs["offset"] for hs in children], dtype=float)
    rows, rhs, used = [], [], set()
    for i in range(len(children)):
        if i in used:
            continue
        for j in range(i + 1, len(children)):
            if j in used:
                continue
            if np.allclose(normals[i], -normals[j], atol=1e-6):
                rows.append(normals[i])
                rhs.append((offsets[i] - offsets[j]) / 2.0)
                used.update((i, j))
                break
    if len(rows) < 3 or np.linalg.matrix_rank(np.array(rows)) < 3:
        return None
    centre, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return centre


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_scene(scene, requested_volume_mm3=None, expect_droplet=True,
                   volume_rtol=0.02):
    """
    Validate a generated scene's emitted geometry.

    Parameters
    ----------
    scene                : scene dict, or path to a scene YAML
    requested_volume_mm3 : the --solvent-volume that was asked for; None
                           skips the volume comparison
    expect_droplet       : False for scenes that legitimately have no solvent
                           object (e.g. MiTeGen mounts); droplet checks skip
    volume_rtol          : relative tolerance on the volume comparison

    Returns a report dict of measured values and warnings.
    Raises SceneValidationError naming every failed check.
    """
    if isinstance(scene, str):
        with open(scene) as f:
            scene = yaml.safe_load(f)

    objects = scene.get("objects", [])
    by_name = {o.get("name"): (i, o) for i, o in enumerate(objects)}
    failures, warnings, report = [], [], {}

    solvent = by_name.get("solvent")
    crystal = by_name.get("crystal")
    loop = by_name.get("loop_fiber")

    if not expect_droplet:
        if solvent is not None and solvent[1]["shape"].get("type") == "surface_mesh":
            failures.append("scene has a solvent mesh but expect_droplet=False")
        if failures:
            raise SceneValidationError("; ".join(failures))
        report["warnings"] = warnings
        return report

    # --- presence ---
    if solvent is None:
        raise SceneValidationError("no 'solvent' object in scene")
    if loop is None:
        raise SceneValidationError("no 'loop_fiber' object in scene")
    shape = solvent[1]["shape"]
    if shape.get("type") != "surface_mesh":
        raise SceneValidationError(
            f"solvent shape is '{shape.get('type')}', not a surface_mesh — "
            "a degenerate fallback shape must never ship silently")

    verts = np.asarray(shape["vertices"], dtype=float)
    faces = np.asarray(shape["faces"], dtype=int)

    loop_path = np.asarray(loop[1]["shape"]["path"], dtype=float)
    fiber_d = float(loop[1]["shape"]["diameter"])
    loop_pts = loop_path[:-1] if np.allclose(loop_path[0], loop_path[-1]) \
        else loop_path
    plane_c, plane_n = _fit_plane(loop_pts)

    # --- mesh integrity ---
    failures += _watertight_failures(faces)
    va, vb, vc = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(vb - va, vc - va), axis=1)
    n_zero = int((areas < 1e-12).sum())
    if n_zero:
        failures.append(f"{n_zero} zero-area triangles")

    # --- volume ---
    volume, centroid = mesh_volume_centroid(verts, faces)
    report["volume_mm3"] = volume
    if volume <= 0:
        failures.append(f"signed volume {volume:.3e} mm^3 is not positive "
                        "(inverted or inconsistent winding)")
    elif requested_volume_mm3 is not None:
        err = abs(volume - requested_volume_mm3) / requested_volume_mm3
        report["volume_error"] = err
        if err > volume_rtol:
            failures.append(
                f"droplet volume {volume:.6f} mm^3 differs from requested "
                f"{requested_volume_mm3:.6f} by {err:.1%} (> {volume_rtol:.0%})")

    # --- placement: the rim must be pinned on the loop fiber ---
    plane_dist = (verts - plane_c) @ plane_n
    rim = verts[np.abs(plane_dist) < 1e-4]
    report["rim_vertices"] = int(len(rim))
    if len(rim) < 8:
        failures.append(f"only {len(rim)} rim vertices found in the loop plane "
                        "— droplet does not sit in the loop")
    else:
        rim_gap = _point_polyline_dist(rim, loop_pts)
        report["rim_to_fiber_mm"] = {"min": float(rim_gap.min()),
                                     "max": float(rim_gap.max())}
        if rim_gap.max() > fiber_d:
            failures.append(
                f"rim vertices up to {rim_gap.max():.4f} mm from the loop "
                f"fiber (fiber diameter {fiber_d:.4f}) — droplet is not "
                "pinned in the loop aperture")

    # --- shape: must straddle the loop plane, not sit on it ---
    h_up, h_dn = float(plane_dist.max()), float(-plane_dist.min())
    report["h_above_mm"], report["h_below_mm"] = h_up, h_dn
    if h_up < 1e-3 or h_dn < 1e-3:
        failures.append(
            f"droplet does not straddle the loop plane (extent {h_up:.4f} mm "
            f"above / {h_dn:.4f} mm below) — a one-sided dome is the old "
            "hemisphere fallback signature")

    # --- crystal ---
    if crystal is not None:
        ci, cobj = crystal
        if ci > solvent[0]:
            failures.append("crystal is listed after solvent — priority order "
                            "makes it invisible inside the droplet")
        children = cobj["shape"].get("children", [])
        centre = _crystal_centre(children) if children else None
        if centre is not None:
            report["crystal_centre"] = [float(x) for x in centre]
            off = centre - centroid
            in_plane = np.linalg.norm(off - (off @ plane_n) * plane_n)
            if in_plane > fiber_d:
                failures.append(
                    f"crystal centre is {in_plane:.4f} mm from the droplet "
                    "centre in the loop plane — crystal not in the droplet")
            half_axial = max(
                abs(float(hs["offset"]) - np.dot(hs["normal"], centre))
                for hs in children
                if abs(np.dot(hs["normal"], plane_n)) > 0.9
            ) if any(abs(np.dot(hs["normal"], plane_n)) > 0.9
                     for hs in children) else 0.0
            if half_axial > max(h_up, h_dn):
                warnings.append(
                    f"crystal half-height {half_axial:.4f} mm exceeds the "
                    f"droplet half-thickness {max(h_up, h_dn):.4f} mm — the "
                    "crystal pokes out of the solvent (a real mount does "
                    "this, but the renderer shows hard crystal/air "
                    "interfaces with no wetting film)")

    # --- mount attachment: the stem must reach into the pin's metal ---
    # The stem is glued to the pin's scored break face; a fiber that stops
    # short of the bevel plane floats in mid-air (visible on rotation).
    pin = by_name.get("pin")
    if pin is not None:
        for stem_name in ("stem_fiber_1", "stem_fiber_2"):
            stem = by_name.get(stem_name)
            if stem is None:
                continue
            end = np.asarray(stem[1]["shape"]["path"][-1], dtype=float)
            if not _point_in_csg(end, pin[1]["shape"]):
                failures.append(
                    f"{stem_name} ends at {np.round(end, 4).tolist()} outside "
                    "the pin — the mount is not attached to the metal")

    report["warnings"] = warnings
    if failures:
        raise SceneValidationError("scene validation failed: " +
                                   "; ".join(failures))
    return report


def _point_in_csg(p, shape):
    """Is point p inside a CSG intersection of cylinders and half-spaces?

    Supports exactly the node types make_pin emits; unknown types count as
    containing the point (so the check errs quiet, never spuriously loud).
    """
    t = shape.get("type")
    if t == "intersection":
        return all(_point_in_csg(p, c) for c in shape.get("children", []))
    if t == "half_space":
        return float(np.dot(shape["normal"], p)) <= shape["offset"] + 1e-9
    if t == "cylinder":
        c = np.asarray(shape["centre"], dtype=float)
        ax = np.asarray(shape["axis"], dtype=float)
        ax = ax / np.linalg.norm(ax)
        d = p - c
        axial = float(np.dot(d, ax))
        radial = float(np.linalg.norm(d - axial * ax))
        return (abs(axial) <= shape["height"] / 2.0 + 1e-9
                and radial <= shape["radius"] + 1e-9)
    return True
