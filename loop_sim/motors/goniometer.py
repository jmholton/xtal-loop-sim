"""
Goniometer: converts motor positions to an SE(3) homogeneous transform.

Motor parameters
----------------
tx, ty, tz : float (mm)   — translations along user-defined axis vectors
rotx       : float (deg)  — rotation about geometry['rotx_axis']
roty       : float (deg)  — rotation about geometry['roty_axis']
rotz       : float (deg)  — rotation about geometry['rotz_axis']
zoom       : float (≥0)   — camera zoom multiplier (not part of SE(3))

The sample transform is:
    T = R_rotz · R_roty · R_rotx · T_trans

where each rotation is applied about the user-defined axis vector using
Rodrigues' rotation formula, and T_trans translates along the user-defined
basis vectors.

The renderer transforms rays as:  ray_sample = inv(T) · ray_lab
(i.e. the camera and beam are "moved" into the sample frame rather than
moving the sample).
"""
import numpy as np


def _rodrigues(axis, angle_deg):
    """4×4 homogeneous rotation matrix for `angle_deg` about unit `axis`."""
    ax = np.asarray(axis, dtype=float)
    ax /= np.linalg.norm(ax) + 1e-30
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = ax
    R3 = np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
    ])
    M = np.eye(4)
    M[:3, :3] = R3
    return M


def _translation(tx, ty, tz, fast, slow, beam):
    """4×4 homogeneous translation along user-defined axes."""
    vec = tx * np.array(fast) + ty * np.array(slow) + tz * np.array(beam)
    M = np.eye(4)
    M[:3, 3] = vec
    return M


class Goniometer:
    """
    Converts motor positions to SE(3) sample transform.

    Parameters
    ----------
    geometry : dict
        Must contain: rotx_axis, roty_axis, rotz_axis,
                      camera_fast, camera_slow, beam_axis
    """

    def __init__(self, geometry):
        self.geometry = geometry
        self._motors = {
            "tx": 0.0, "ty": 0.0, "tz": 0.0,
            "rotx": 0.0, "roty": 0.0, "rotz": 0.0,
            "zoom": 1.0,
        }

    @property
    def zoom(self):
        return self._motors["zoom"]

    def set(self, **kwargs):
        """Update one or more motor positions and return self."""
        for k, v in kwargs.items():
            if k in self._motors:
                self._motors[k] = float(v)
        return self

    def get(self):
        """Return a copy of current motor positions."""
        return dict(self._motors)

    def transform(self):
        """
        Build and return the 4×4 SE(3) sample transform matrix.
        Rays should be multiplied by inv(transform()) to go from lab to sample frame.
        """
        g = self.geometry
        fast  = g.get("camera_fast",  [1, 0, 0])
        slow  = g.get("camera_slow",  [0, 1, 0])
        beam  = g.get("beam_axis",    [0, 0, 1])
        ax_x  = g.get("rotx_axis",    [1, 0, 0])
        ax_y  = g.get("roty_axis",    [0, 1, 0])
        ax_z  = g.get("rotz_axis",    [0, 0, 1])

        m = self._motors
        T   = _translation(m["tx"], m["ty"], m["tz"], fast, slow, beam)
        Rx  = _rodrigues(ax_x, m["rotx"])
        Ry  = _rodrigues(ax_y, m["roty"])
        Rz  = _rodrigues(ax_z, m["rotz"])

        return Rz @ Ry @ Rx @ T

    def transform_inv(self):
        """Inverse of transform() — maps lab → sample frame."""
        return np.linalg.inv(self.transform())


def apply_transform(M, points):
    """Apply 4×4 homogeneous matrix M to (N, 3) points."""
    ones = np.ones((len(points), 1))
    ph   = np.hstack([points, ones])   # (N, 4)
    out  = (M @ ph.T).T                # (N, 4)
    return out[:, :3]


def apply_transform_dirs(M, dirs):
    """Apply rotation part of 4×4 matrix M to (N, 3) direction vectors."""
    return (M[:3, :3] @ dirs.T).T
