"""Launch Jacob's camera server unchanged, with two fixes applied at runtime
(Jacob's repo is not modified):

1. Static files come from jacob_run/static/ when a copy exists there
   (index.html: panel stays to the right of the video at any window width).

2. Template crops honour the scene's handedness.  pose_crop/servable_pose put
   `ty*cos - tz*sin` on screen, which is right only when beam = fast x slow.
   James's scenes use camera_slow [0,-1,0] with beam [0,0,+1] (left-handed), so
   the tz part of every vertical move came out mirrored: at phi=0 a click was
   right, but as phi grows the vertical motion is increasingly tz and goes the
   wrong way (fully reversed at 90).  The live renderer and recenter_target use
   the real geometry and were already right.  For a left-handed scene tz is
   negated going in (and coming back out of servable_pose); the stored frames
   need no rebuild -- their window offset (centre_y ~ 0.1 um) is negligible.

cd /home/jadoughty/projects/loop_sim_MINE/xtal-loop-sim && \
    .venv/bin/python /home/jamesh/projects/loop_sim/jacob_run/serve.py [camera_server args]
"""
import os
import sys

sys.path.insert(0, os.getcwd())
from loop_sim.server import camera_server as cs  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")
_orig = cs.CameraServer._static_bytes


def _static_bytes(self, name):
    p = os.path.join(STATIC, name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    return _orig(self, name)


cs.CameraServer._static_bytes = _static_bytes

import numpy as np  # noqa: E402
import yaml  # noqa: E402

_hand = {}


def _left_handed(manifest):
    path = manifest.get("scene", "")
    if path not in _hand:
        try:
            g = yaml.safe_load(open(path)).get("geometry", {})
            f, sl, b = (np.array(g.get(k, d), float) for k, d in
                        (("camera_fast", [1, 0, 0]), ("camera_slow", [0, 1, 0]),
                         ("beam_axis", [0, 0, 1])))
            _hand[path] = float(np.cross(f, sl) @ b) < 0
        except OSError:
            _hand[path] = False
    return _hand[path]


_pose_crop, _servable_pose = cs.pose_crop, cs.servable_pose


def pose_crop(manifest, tx=0.0, ty=0.0, tz=0.0, *a, **k):
    if _left_handed(manifest):
        tz = -tz
    return _pose_crop(manifest, tx, ty, tz, *a, **k)


def servable_pose(manifest, tx=0.0, ty=0.0, tz=0.0, *a, **k):
    flip = _left_handed(manifest)
    eff, note = _servable_pose(manifest, tx, ty, -tz if flip else tz, *a, **k)
    if flip and eff is not None and "tz" in eff:
        eff = dict(eff, tz=-eff["tz"])
    return eff, note


cs.pose_crop, cs.servable_pose = pose_crop, servable_pose
cs.main()
