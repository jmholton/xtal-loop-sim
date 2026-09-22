"""The camera server's lock order, checked statically.

    _anim_cv (3) > _scene_lock (2, RLock) > _gonio_lock (1)
    _frame_cv (0) and _push_lock (0) share rank 0 as independent leaves.

Inside `with self._L:`, nothing it calls may acquire rank >= rank(L);
re-acquiring `_scene_lock` is fine (it is an RLock).

Checked statically: the inversions are call-mediated and invisible in
either function alone (a lock on `_servable` would create `_gonio_lock ->
_scene_lock` via `_set_pose_instant`), and `threading.Condition` wraps an
RLock, so a re-entrant `_anim_cv` would succeed silently rather than
hang.  Runtime backstop: test_scene_switch.py's
test_no_deadlock_under_concurrent_switch_render_and_motor.  See
docs/DECISIONS.md 2026-07-06 (_active_compiled incident).
"""
import ast
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SRC = os.path.join(REPO_ROOT, "loop_sim", "server", "camera_server.py")

RANK = {"_anim_cv": 3, "_scene_lock": 2, "_gonio_lock": 1,
        "_frame_cv": 0, "_push_lock": 0}
REENTRANT = {"_scene_lock"}
LEAF_LOCKS = {"_frame_cv", "_push_lock"}


def _lock_name(item):
    """The lock a `with` item acquires, if it is one of ours."""
    ctx = item.context_expr
    if isinstance(ctx, ast.Attribute) and ctx.attr in RANK:
        return ctx.attr
    return None


def _acquired_within(node, methods, seen):
    """Every one of our locks acquired inside `node`, following method calls."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.With):
            for item in n.items:
                name = _lock_name(item)
                if name:
                    out.add(name)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            callee = n.func.attr
            if callee in methods and callee not in seen:
                out |= _acquired_within(methods[callee], methods, seen | {callee})
    return out


def _methods(tree):
    return {fn.name: fn
            for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)
            for fn in cls.body if isinstance(fn, ast.FunctionDef)}


def test_lock_order_is_never_inverted():
    with open(SRC) as fh:
        tree = ast.parse(fh.read())
    methods = _methods(tree)

    bad = []
    for holder in ast.walk(tree):
        if not isinstance(holder, ast.With):
            continue
        for item in holder.items:
            outer = _lock_name(item)
            if outer is None:
                continue
            for stmt in holder.body:
                for inner in _acquired_within(stmt, methods, set()):
                    inverted = RANK[inner] > RANK[outer]
                    same = RANK[inner] == RANK[outer] and inner not in REENTRANT
                    if inverted or same:
                        bad.append(f"line {holder.lineno}: "
                                   f"{outer}(rank {RANK[outer]}) -> "
                                   f"{inner}(rank {RANK[inner]})")
    assert not bad, (
        "lock order inverted -- the invariant is "
        "_anim_cv > _scene_lock > _gonio_lock, _frame_cv a leaf:\n  "
        + "\n  ".join(sorted(set(bad))))


def test_frame_cv_is_a_leaf():
    """Every lock in LEAF_LOCKS must never be held while any other lock is
    taken.

    _frame_cv is the one lock a request thread grabs on every streamed
    optical frame, so anything nested under it would put stream latency
    behind scene or goniometer contention -- and _get_jpeg deliberately calls
    _render_now OUTSIDE its `with` for exactly that reason. _push_lock guards
    only the /video-trigger state, so the pusher reads its token and the
    frame slot one lock at a time.
    """
    with open(SRC) as fh:
        tree = ast.parse(fh.read())
    methods = _methods(tree)

    bad = []
    for holder in ast.walk(tree):
        if not isinstance(holder, ast.With):
            continue
        leaf = next((_lock_name(i) for i in holder.items
                    if _lock_name(i) in LEAF_LOCKS), None)
        if leaf is None:
            continue
        for stmt in holder.body:
            for inner in _acquired_within(stmt, methods, set()):
                bad.append(f"line {holder.lineno}: {leaf} -> {inner}")
    assert not bad, ("a leaf lock is not a leaf:\n  " + "\n  ".join(sorted(set(bad))))


def test_the_ranked_locks_all_exist():
    """Guard against the checks above silently passing on a renamed lock."""
    with open(SRC) as fh:
        src = fh.read()
    for name in RANK:
        assert f"self.{name}" in src, f"{name} is gone -- update RANK"
