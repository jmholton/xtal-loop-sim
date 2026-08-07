"""
The camera server's lock order, checked statically.

    _anim_cv (3)  >  _scene_lock (2, RLock)  >  _gonio_lock (1)
    _frame_cv (0) is a LEAF

Inside `with self._L:`, nothing -- directly, or through any CameraServer method
it calls -- may acquire a lock of rank >= rank(L).  Re-acquiring _scene_lock is
allowed and expected: it is an RLock, and _render_now holds it across
_render_frame, which calls _snapshot_gonio, which needs it too.

Why STATIC.  The inversions that actually happen here are call-mediated and
invisible in the body of either function involved.  The live example:
`_servable` reads self._templates, and `_set_pose_instant` calls it from inside
_gonio_lock -- so giving _servable a lock of its own would create
_gonio_lock -> _scene_lock and hang against _render_now, which holds
_scene_lock and then wants _gonio_lock.  Neither function looks wrong on its
own.  And a runtime assertion cannot cover the other half: threading.Condition
wraps an RLock, so an accidentally re-entrant _anim_cv would silently succeed
and release early rather than deadlocking, leaving nothing to observe.

tests/test_scene_switch.py::test_no_deadlock_under_concurrent_switch_render_and_motor
is the runtime backstop for anything this cannot see.

Run:  pytest tests/test_server_lock_order.py -v
"""
import ast
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SRC = os.path.join(REPO_ROOT, "loop_sim", "server", "camera_server.py")

RANK = {"_anim_cv": 3, "_scene_lock": 2, "_gonio_lock": 1, "_frame_cv": 0}
REENTRANT = {"_scene_lock"}


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
    """_frame_cv must never be held while any other lock is taken.

    It is the one lock a request thread grabs on every streamed frame, so
    anything nested under it would put stream latency behind scene or
    goniometer contention -- and _get_jpeg deliberately calls _render_now
    OUTSIDE its `with` for exactly that reason.
    """
    with open(SRC) as fh:
        tree = ast.parse(fh.read())
    methods = _methods(tree)

    bad = []
    for holder in ast.walk(tree):
        if not isinstance(holder, ast.With):
            continue
        if not any(_lock_name(i) == "_frame_cv" for i in holder.items):
            continue
        for stmt in holder.body:
            for inner in _acquired_within(stmt, methods, set()):
                bad.append(f"line {holder.lineno}: _frame_cv -> {inner}")
    assert not bad, ("_frame_cv is not a leaf:\n  " + "\n  ".join(sorted(set(bad))))


def test_the_ranked_locks_all_exist():
    """Guard against the checks above silently passing on a renamed lock."""
    with open(SRC) as fh:
        src = fh.read()
    for name in RANK:
        assert f"self.{name}" in src, f"{name} is gone -- update RANK"
