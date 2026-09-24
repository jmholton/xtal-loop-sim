# -*- coding: utf-8 -*-
"""Pose model and camera-server link for the xtalLoopSim DHS.

This is the layer that owns device state. The DHS (xtal_loop_sim_DHS.py) only
translates between DCSS messages and the calls here, and never stores a motor
position of its own.

The simulator's pose is the single source of truth. A DCSS motor is an `Axis`:
a name, the simulator pose key it drives (`rotx`, `tx`, ...), the beamline
database's scale/speed/limits, and a sign multiplier for the case where the
beamline's axis direction does not match the scene's. Two axes may share one
pose key (gonio_phi and absolute_phi are the same spindle), so a move is held
against the POSE KEY, not the motor name: commanding one answers `moving` on
the other.

Two backends behind one interface, so the same move code runs both ways:

  * CameraServerBackend - `real`: POSTs the move to the camera server and reads
    the pose back from it. The server does the animation.
  * PretendBackend      - `pretend`: an internal linear ramp with the same
    timing and the same reported trajectory, no camera server. This is what the
    unit tests drive.

Move duration is the beamline's own: |delta| * scale / speed, the time the real
motor's step generator would need at the database slew rate, with the
acceleration term ignored and a floor so a zero-length move still completes
through the normal stream.
"""
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_S = 0.1   # between position reports while a move runs
DEFAULT_MIN_MOVE_S = 0.05       # floor on a move's duration
DEFAULT_TIMEOUT_S = 3.0         # per-request connect+read timeout


def num(value: float) -> str:
    """Format a number for the DCS wire.

    DCS serialises with ' '.join(args), so every token must already be a string.
    Plain '%g' turns a slew rate of 9000000 into '9e+06'; an integral value is
    therefore written out in full and only a fractional one goes through '%g'.
    """
    value = float(value)
    if value.is_integer() and abs(value) < 1e15:
        return '%d' % value
    return '%g' % value


class Axis:
    """One DCSS motor and the simulator pose key it drives.

    `scale`, `speed`, `accel`, `backlash`, the limits and the five flags are the
    beamline database's own values and are reported verbatim in
    htos_configure_device; dcss will not accept a move until it has them.

    `sign` is +1 when the beamline axis and the scene axis point the same way
    and -1 when they oppose. `circle` is 360 for an axis dcss runs in circle
    mode (gonio_phi): its reported position is wrapped to [0, 360) while the
    pose it drives stays unwrapped, so absolute_phi can read the same spindle
    without wrapping.
    """

    def __init__(self, name: str, sim_key: str, units: str = 'mm',
                 scale: float = 1.0, speed: float = 1.0, accel: float = 0.0,
                 backlash: float = 0.0, upper: float = 0.0, lower: float = 0.0,
                 lower_on: int = 0, upper_on: int = 0, lock_on: int = 0,
                 backlash_on: int = 0, reverse_on: int = 0,
                 circle: float = 0.0, sign: float = 1.0,
                 position: float = 0.0) -> None:
        self.name = name
        self.sim_key = sim_key
        self.units = units
        self.scale = float(scale)
        self.speed = float(speed)
        self.accel = float(accel)
        self.backlash = float(backlash)
        self.upper = float(upper)
        self.lower = float(lower)
        self.lower_on = int(lower_on)
        self.upper_on = int(upper_on)
        self.lock_on = int(lock_on)
        self.backlash_on = int(backlash_on)
        self.reverse_on = int(reverse_on)
        self.circle = float(circle)
        self.sign = 1.0 if float(sign) >= 0 else -1.0
        self.position = float(position)   # the seed until the backend is read

    def report(self, sim_value: float) -> float:
        """The position dcss sees for a pose value."""
        value = float(sim_value) * self.sign
        if self.circle:
            value = value % self.circle
        return value

    def to_sim(self, target: float, current_sim: float) -> float:
        """The pose value that makes this motor read `target`.

        A circle axis is driven by the DELTA from its reported position, so the
        underlying pose keeps counting past 360 and a wrapped command never
        teleports the spindle back into the first turn. dcss picks the short way
        round for these axes itself, with stoh_correct_motor_position.
        """
        target = float(target)
        if self.circle:
            return float(current_sim) + (target - self.report(current_sim)) * self.sign
        return target * self.sign

    def duration(self, delta_sim: float, floor: float = DEFAULT_MIN_MOVE_S) -> float:
        """Seconds for a move of `delta_sim`, at the database slew rate."""
        if self.speed <= 0:
            return floor
        return max(abs(float(delta_sim)) * self.scale / self.speed, floor)

    def fmt(self, value: float) -> str:
        return num(value)

    def settings(self, position: float) -> str:
        """The 12 fields of htos_configure_device, after the motor name."""
        return ' '.join([
            num(position), num(self.upper), num(self.lower), num(self.scale),
            num(self.speed), num(self.accel), num(self.backlash),
            num(self.lower_on), num(self.upper_on), num(self.lock_on),
            num(self.backlash_on), num(self.reverse_on)])

    def __repr__(self) -> str:
        return 'Axis({}, {}, sign={:+g})'.format(self.name, self.sim_key, self.sign)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class SimBackend:
    """Where the pose lives. Never raises: a dead link reports the last pose."""

    def read(self) -> Tuple[Dict[str, float], bool]:
        """(pose, moving) - every owned pose key, and whether motion is running."""
        raise NotImplementedError

    def command_move(self, targets: Dict[str, float], duration: float) -> None:
        """Start an animated move to `targets` lasting `duration` seconds."""
        raise NotImplementedError

    def set_pose(self, targets: Dict[str, float]) -> None:
        """Place the pose at `targets` with no motion."""
        raise NotImplementedError

    def stop(self, pose: Dict[str, float]) -> None:
        """Halt every animation, leaving the pose where `pose` says it is."""
        raise NotImplementedError

    def video_trigger(self, state: str) -> None:
        """Tell the simulator the video shutter is now open or closed."""
        raise NotImplementedError


class PretendBackend(SimBackend):
    """An internal pose model with the camera server's timing and no server.

    One linear ramp per pose key, sampled on read. The trajectory a caller sees
    is the same shape the camera server produces, so a pretend run is a faithful
    rehearsal of the DCSS traffic.
    """

    def __init__(self, seed: Dict[str, float]) -> None:
        self._pose = {k: float(v) for k, v in seed.items()}
        self._ramps: Dict[str, Tuple[float, float, float, float]] = {}
        self._lock = threading.Lock()
        self.video_log: List[str] = []      # every video-trigger state, in order

    def read(self) -> Tuple[Dict[str, float], bool]:
        now = time.monotonic()
        with self._lock:
            moving = False
            for key in list(self._ramps):
                start, target, t0, dur = self._ramps[key]
                if dur <= 0 or now >= t0 + dur:
                    self._pose[key] = target
                    del self._ramps[key]
                else:
                    self._pose[key] = start + (target - start) * (now - t0) / dur
                    moving = True
            return dict(self._pose), moving

    def command_move(self, targets: Dict[str, float], duration: float) -> None:
        self.read()                     # settle anything that has already finished
        now = time.monotonic()
        with self._lock:
            for key, target in targets.items():
                self._ramps[key] = (self._pose.get(key, 0.0), float(target),
                                    now, float(duration))

    def set_pose(self, targets: Dict[str, float]) -> None:
        with self._lock:
            for key, target in targets.items():
                self._ramps.pop(key, None)
                self._pose[key] = float(target)

    def stop(self, pose: Dict[str, float]) -> None:
        with self._lock:
            self._ramps.clear()
            self._pose.update({k: float(v) for k, v in pose.items()})

    def video_trigger(self, state: str) -> None:
        self.video_log.append(state)
        _logger.info('PRETEND: video_trigger %s', state)


class CameraServerBackend(SimBackend):
    """The camera server over HTTP: it owns the pose and does the animation.

    Uses the four endpoints listed in README.md "What the camera server must
    provide" (`loop_sim/server/camera_server.py`).

    Everything is caught here so the move loop above never has to: a failed read
    reports the last known pose, standing still. The cache holds exactly the
    pose keys this DHS owns, so an abort pins those and nothing else.
    """

    def __init__(self, base_url: str, seed: Dict[str, float],
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._base = base_url.rstrip('/')
        self._timeout = float(timeout_s)
        self._pose = {k: float(v) for k, v in seed.items()}
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return self._base

    def _request(self, method: str, path: str, params: Dict[str, str]) -> Optional[dict]:
        url = self._base + path
        if params:
            url = url + '?' + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, data=b'' if method == 'POST' else None,
                                     method=method)
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read()
        if not body:
            return None
        return json.loads(body.decode('utf-8'))

    def _cache(self, positions: Optional[dict]) -> None:
        if not positions:
            return
        with self._lock:
            for key in list(self._pose):
                if key in positions:
                    self._pose[key] = float(positions[key])

    def read(self) -> Tuple[Dict[str, float], bool]:
        try:
            doc = self._request('GET', '/status', {}) or {}
        except Exception as exc:
            _logger.warning('GET %s/status failed (%r); reporting the last known '
                            'pose, standing still', self._base, exc)
            with self._lock:
                return dict(self._pose), False
        self._cache(doc.get('positions'))
        with self._lock:
            return dict(self._pose), bool(doc.get('moving', False))

    def command_move(self, targets: Dict[str, float], duration: float) -> None:
        params = {k: num(v) for k, v in targets.items()}
        params['duration'] = num(duration)
        try:
            self._request('POST', '/move', params)
        except Exception as exc:
            _logger.error('POST %s/move %s failed (%r)', self._base, params, exc)

    def set_pose(self, targets: Dict[str, float]) -> None:
        params = {k: num(v) for k, v in targets.items()}
        try:
            doc = self._request('POST', '/motor', params)
        except Exception as exc:
            _logger.error('POST %s/motor %s failed (%r)', self._base, params, exc)
            return
        self._cache(doc if isinstance(doc, dict) else None)

    def stop(self, pose: Dict[str, float]) -> None:
        # A zero-duration move to where the sample already is: the camera server
        # drops the running animation and holds that pose.
        self.command_move(pose, 0.0)

    def video_trigger(self, state: str) -> None:
        try:
            self._request('POST', '/video-trigger', {'state': state})
        except Exception as exc:
            _logger.error('POST %s/video-trigger state=%s failed (%r)',
                          self._base, state, exc)


# ---------------------------------------------------------------------------
# The link the DHS calls
# ---------------------------------------------------------------------------
class SimLink:
    """Motors, shutters and moves, over whichever backend was built.

    A move runs on its own thread: pydhsfw dispatches every message on one
    thread, so a handler that waited for a move would stall stoh_abort_all and
    the handshake.
    """

    def __init__(self, backend: SimBackend, axes: List[Axis],
                 shutters: Dict[str, str], video_shutter: str = 'video_trigger',
                 poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
                 min_move_s: float = DEFAULT_MIN_MOVE_S) -> None:
        self._backend = backend
        self._axes = {axis.name: axis for axis in axes}
        self._shutters = dict(shutters)
        self._video_shutter = video_shutter
        self._poll_interval_s = float(poll_interval_s)
        self._min_move_s = float(min_move_s)
        self._moving: Dict[str, threading.Event] = {}   # pose key -> abort flag
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state
    @property
    def backend(self) -> SimBackend:
        return self._backend

    def get_axis(self, name: str) -> Optional[Axis]:
        """The axis, or None when dcss names a motor this DHS does not own."""
        return self._axes.get(name)

    def axes(self) -> List[Axis]:
        return list(self._axes.values())

    def shutters(self) -> Dict[str, str]:
        return dict(self._shutters)

    def get_shutter(self, name: str) -> Optional[str]:
        return self._shutters.get(name)

    def position(self, name: str) -> float:
        axis = self._axes[name]
        pose, _ = self._backend.read()
        return axis.report(pose.get(axis.sim_key, axis.position))

    def position_str(self, name: str) -> str:
        return num(self.position(name))

    def is_moving(self, name: str) -> bool:
        """True while any motor on this axis's pose key has a move in flight."""
        axis = self._axes[name]
        with self._lock:
            return axis.sim_key in self._moving

    # ------------------------------------------------------------------ moves
    def start_move(self, name: str, target: float,
                   on_start: Optional[Callable[[str], None]] = None,
                   on_update: Optional[Callable[[str], None]] = None,
                   on_complete: Optional[Callable[[str, str], None]] = None,
                   duration: Optional[float] = None) -> bool:
        """Begin a move. False means the axis is already moving and nothing started.

        on_start(target_str) fires before the backend is commanded,
        on_update(pos_str) on each poll where the reported position changed, and
        on_complete(pos_str, 'normal'|'aborted') once at the end. Positions are
        already formatted for the wire.
        """
        axis = self._axes[name]
        abort = threading.Event()
        with self._lock:
            if axis.sim_key in self._moving:
                return False
            self._moving[axis.sim_key] = abort

        pose, _ = self._backend.read()
        current = pose.get(axis.sim_key, axis.position)
        sim_target = axis.to_sim(float(target), current)
        if duration is None:
            duration = axis.duration(sim_target - current, self._min_move_s)
        duration = max(float(duration), 0.0)

        if on_start:
            on_start(axis.fmt(axis.report(sim_target)))
        _logger.info('%s: move %s -> %s %s over %.3f s', axis.name,
                     axis.fmt(axis.report(current)),
                     axis.fmt(axis.report(sim_target)), axis.units, duration)
        self._backend.command_move({axis.sim_key: sim_target}, duration)

        thread = threading.Thread(
            target=self._run_move, args=(axis, abort, on_update, on_complete),
            name='move-{}'.format(name), daemon=True)
        with self._lock:
            self._threads[axis.sim_key] = thread
        thread.start()
        return True

    def _run_move(self, axis: Axis, abort: threading.Event,
                  on_update: Optional[Callable[[str], None]],
                  on_complete: Optional[Callable[[str, str], None]]) -> None:
        """Poll the backend until it stops moving, streaming what changed."""
        last = None
        aborted = False
        try:
            while True:
                if abort.wait(self._poll_interval_s):
                    aborted = True
                    break
                pose, moving = self._backend.read()
                position = axis.fmt(axis.report(pose.get(axis.sim_key, axis.position)))
                if not moving:
                    break
                if position != last:
                    last = position
                    if on_update:
                        on_update(position)
        finally:
            with self._lock:
                self._moving.pop(axis.sim_key, None)
                self._threads.pop(axis.sim_key, None)
        pose, _ = self._backend.read()
        final = axis.fmt(axis.report(pose.get(axis.sim_key, axis.position)))
        axis.position = pose.get(axis.sim_key, axis.position)
        state = 'aborted' if aborted else 'normal'
        _logger.info('%s: move %s at %s %s', axis.name, state, final, axis.units)
        if on_complete:
            on_complete(final, state)

    def join(self, timeout: float = 5.0) -> None:
        """Wait for every in-flight move thread. Tests and shutdown use this."""
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=timeout)

    def abort_all(self) -> None:
        """Stop the simulator, then flag every in-flight move so it completes aborted."""
        pose, _ = self._backend.read()
        self._backend.stop(pose)
        with self._lock:
            flags = list(self._moving.values())
        for flag in flags:
            flag.set()

    # ------------------------------------------------------------------- sets
    def set_position(self, name: str, value: float) -> str:
        """Redefine where a motor is, with no motion. Returns the new position.

        The scene follows: this DHS has no offset table, so what dcss is told
        and what the picture shows can never drift apart. The real pmac2DHS
        instead shifts a software offset and leaves the spindle alone.
        """
        axis = self._axes[name]
        pose, _ = self._backend.read()
        sim_value = axis.to_sim(float(value), pose.get(axis.sim_key, axis.position))
        self._backend.set_pose({axis.sim_key: sim_value})
        axis.position = sim_value
        return axis.fmt(axis.report(sim_value))

    def correct_position(self, name: str, correction: float) -> str:
        """Apply dcss's circle correction: the position moves by `correction`."""
        axis = self._axes[name]
        pose, _ = self._backend.read()
        current = pose.get(axis.sim_key, axis.position)
        sim_value = current + float(correction) * axis.sign
        self._backend.set_pose({axis.sim_key: sim_value})
        axis.position = sim_value
        return axis.fmt(axis.report(sim_value))

    # --------------------------------------------------------------- shutters
    def set_shutter(self, name: str, state: str) -> str:
        """Set a shutter. The video shutter also pushes to the camera server."""
        state = 'open' if state == 'open' else 'closed'
        self._shutters[name] = state
        if name == self._video_shutter:
            self._backend.video_trigger(state)
        return state


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def build_link(conf: dict, pretend: bool) -> SimLink:
    """Build the SimLink a config file describes.

    `conf` is the parsed config file; the device table lives under
    `xtal_loop_sim.motors` and `xtal_loop_sim.shutters` (README "Config keys").
    """
    section = conf.get('xtal_loop_sim', {}) or {}
    axes = []
    for name, spec in (section.get('motors') or {}).items():
        spec = dict(spec or {})
        axes.append(Axis(name, spec.pop('sim_key'), **spec))
    shutters = {name: ('open' if str(state) == 'open' else 'closed')
                for name, state in (section.get('shutters') or {}).items()}

    seed = {}
    for axis in axes:
        # Two motors on one pose key must not seed it twice; the unwrapped one
        # (absolute_phi) is the correct seed, so a wrapping axis never overwrites it.
        if axis.sim_key not in seed or not axis.circle:
            seed[axis.sim_key] = axis.position * axis.sign

    if pretend:
        backend: SimBackend = PretendBackend(seed)
    else:
        backend = CameraServerBackend(
            section.get('camera_server', 'http://localhost:8081'), seed,
            timeout_s=float(section.get('http_timeout_s', DEFAULT_TIMEOUT_S)))

    return SimLink(
        backend, axes, shutters,
        video_shutter=section.get('video_shutter', 'video_trigger'),
        poll_interval_s=float(section.get('poll_interval_s', DEFAULT_POLL_INTERVAL_S)),
        min_move_s=float(section.get('min_move_s', DEFAULT_MIN_MOVE_S)))
