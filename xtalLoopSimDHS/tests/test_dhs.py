# -*- coding: utf-8 -*-
"""Offline tests for xtal_loop_sim_DHS.py - no dcss, no camera server, no beamline.

Three layers:

  * the DCSS handlers in pretend mode, with a fake connection that records REAL
    pydhsfw serialization (str(message) runs the framework's ' '.join over
    _split_msg, so a non-str token fails HERE; a fake that stringified while
    recording would mask exactly that crash class). The assertions are the exact
    wire lines, because the wire is the contract;
  * the real-mode HTTP client against an in-process http.server implementing
    /status, /move, /motor and /video-trigger. Its job is the URL format and the
    polling loop, without loop_sim or a rendered frame in sight;
  * the production-host refusal, both as a function and as the real script
    exiting 2.

Moves run on their own threads, so these wait for the completion to arrive.

Run:
    .venv/bin/python -m pytest tests -q
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the modules under test live one level up

import xtal_loop_sim_DHS as dhs
from sim_link import Axis, CameraServerBackend, SimLink, build_link, num

from pydhsfw.dcss import (
    DcssStoCSendClientType,
    DcssStoHRegisterRealMotor,
    DcssStoHRegisterShutter,
    DcssStoHConfigureRealMotor,
    DcssStoHStartMotorMove,
    DcssStoHSetMotorPosition,
    DcssStoHCorrectMotorPosition,
    DcssStoHSetShutterState,
    DcssStoHAbortAll,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_DHS_DIR = os.path.dirname(_HERE)
LOCAL_CONFIG = os.path.join(_DHS_DIR, 'config', 'LOCAL.config')
DHS_SCRIPT = os.path.join(_DHS_DIR, 'xtal_loop_sim_DHS.py')


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeConn:
    """Records messages sent via .send() AFTER real pydhsfw serialization."""

    def __init__(self) -> None:
        self.sent: List[str] = []
        self._lock = threading.Lock()

    def send(self, message: Any) -> None:
        wire = str(message)     # the framework's ' '.join(_split_msg): non-str raises
        with self._lock:
            self.sent.append(wire)

    def lines(self) -> List[str]:
        with self._lock:
            return list(self.sent)

    def of_type(self, type_id: str) -> List[str]:
        return [line for line in self.lines() if line.split(' ')[0] == type_id]

    def wait_for(self, type_id: str, timeout: float = 10.0) -> str:
        """Block until a line of this type arrives; return the last one."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            hits = self.of_type(type_id)
            if hits:
                return hits[-1]
            time.sleep(0.01)
        raise AssertionError('no {} in {}'.format(type_id, self.lines()))


class FakeContext:
    """Just enough DcssContext for the handlers."""

    def __init__(self, conf: Optional[dict] = None, pretend: bool = True) -> None:
        conf = conf if conf is not None else load_conf()
        self.conn = FakeConn()
        self.config = conf
        self.state = {
            'url': 'dcss://localhost:14242',
            'dhs_name': conf['xtal_loop_sim']['dhs_name'],
            'link': build_link(conf, pretend),
            'pretend': pretend,
        }

    def get_connection(self, name: str) -> FakeConn:
        return self.conn

    @property
    def link(self) -> SimLink:
        return self.state['link']


def load_conf(motors: Optional[Dict[str, dict]] = None, **section: Any) -> dict:
    """The shipped LOCAL config, with the given overrides applied."""
    with open(LOCAL_CONFIG, 'r') as f:
        conf = yaml.safe_load(f)
    conf['xtal_loop_sim'].update(section)
    for name, over in (motors or {}).items():
        conf['xtal_loop_sim']['motors'][name].update(over)
    return conf


def msg(cls: Any, line: str) -> Any:
    """A real pydhsfw inbound message, parsed from its wire line."""
    return cls(line.split(' '))


# ---------------------------------------------------------------------------
# Handshake and registration
# ---------------------------------------------------------------------------
def test_client_type_reply() -> None:
    ctx = FakeContext()
    dhs.send_client_type(msg(DcssStoCSendClientType, 'stoc_send_client_type'), ctx)
    assert ctx.conn.lines() == ['htos_client_is_hardware xtalLoopSimDHS']


def test_register_real_motor_configures_all_13_fields() -> None:
    ctx = FakeContext()
    dhs.register_real_motor(
        msg(DcssStoHRegisterRealMotor, 'stoh_register_real_motor gonio_phi gonio_phi'),
        ctx)
    assert ctx.conn.lines() == [
        'htos_configure_device gonio_phi 0 360 360 8385 9000000 20000 0 0 0 0 0 0']
    # name + position upper lower scale speed accel backlash
    # lowerOn upperOn lockOn backlashOn reverseOn
    assert len(ctx.conn.lines()[0].split(' ')) == 1 + 13


def test_register_real_motor_unknown_is_ignored() -> None:
    ctx = FakeContext()
    dhs.register_real_motor(
        msg(DcssStoHRegisterRealMotor, 'stoh_register_real_motor gonio_omega omega'),
        ctx)
    assert ctx.conn.lines() == []


def test_register_shutter_configures_and_reports() -> None:
    ctx = FakeContext()
    dhs.register_shutter(
        msg(DcssStoHRegisterShutter,
            'stoh_register_shutter video_trigger closed video_trigger'), ctx)
    assert ctx.conn.lines() == [
        'htos_configure_shutter video_trigger open closed closed',
        'htos_report_shutter_state video_trigger closed']


def test_register_shutter_adopts_the_state_dcss_remembers() -> None:
    ctx = FakeContext()
    dhs.register_shutter(
        msg(DcssStoHRegisterShutter, 'stoh_register_shutter shutter open shutter'), ctx)
    assert ctx.conn.lines() == [
        'htos_configure_shutter shutter open closed open',
        'htos_report_shutter_state shutter open']
    assert ctx.link.get_shutter('shutter') == 'open'


def test_configure_real_motor_adopts_speed_and_limits() -> None:
    ctx = FakeContext()
    dhs.configure_real_motor(
        msg(DcssStoHConfigureRealMotor,
            'stoh_configure_real_motor sample_x 0 5 -5 16968 2500 50 0 1 1 0 0 0'),
        ctx)
    axis = ctx.link.get_axis('sample_x')
    assert (axis.speed, axis.accel, axis.upper, axis.lower) == (2500.0, 50.0, 5.0, -5.0)


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------
def test_gonio_phi_move_streams_and_wraps() -> None:
    ctx = FakeContext()
    # 400 deg at 8385 steps/deg and 9000000 steps/s is 0.37 s: several polls.
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move gonio_phi 400'), ctx)
    completed = ctx.conn.wait_for('htos_motor_move_completed')
    ctx.link.join()

    lines = ctx.conn.lines()
    assert lines[0] == 'htos_motor_move_started gonio_phi 40'
    updates = ctx.conn.of_type('htos_update_motor_position')
    assert updates, 'a move this long must report at least one position'
    for update in updates:
        name, position, status = update.split(' ')[1:]
        assert name == 'gonio_phi' and status == 'moving'
        assert 0.0 <= float(position) < 360.0     # circle mode: always in range
    assert completed == 'htos_motor_move_completed gonio_phi 40 normal'
    # The same spindle, unwrapped: the turn count is kept.
    assert ctx.link.position_str('absolute_phi') == '400'


def test_sample_x_move_applies_the_config_sign() -> None:
    ctx = FakeContext(load_conf(motors={'sample_x': {'sign': -1}}))
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move sample_x 0.1'), ctx)
    completed = ctx.conn.wait_for('htos_motor_move_completed')
    ctx.link.join()

    assert ctx.conn.lines()[0] == 'htos_motor_move_started sample_x 0.1'
    assert completed == 'htos_motor_move_completed sample_x 0.1 normal'
    # dcss asked for +0.1 mm; the scene went the other way.
    pose, _ = ctx.link.backend.read()
    assert abs(pose['tx'] + 0.1) < 1e-9


def test_unknown_motor_move_is_ignored() -> None:
    ctx = FakeContext()
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move gonio_omega 90'), ctx)
    assert ctx.conn.lines() == []


def test_second_move_while_moving_answers_moving() -> None:
    ctx = FakeContext()
    # 1 mm at 16968 steps/mm and 5000 steps/s is 3.4 s: still running below.
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move sample_x 1'), ctx)
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move sample_x 0.5'), ctx)
    answer = ctx.conn.of_type('htos_motor_move_completed')[0]
    assert answer.split(' ')[0::3] == ['htos_motor_move_completed', 'moving']
    assert answer.split(' ')[1] == 'sample_x'

    ctx.link.abort_all()
    ctx.link.join()


def test_a_move_on_the_shared_spindle_blocks_the_other_motor() -> None:
    ctx = FakeContext()
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move absolute_phi 3000'), ctx)
    assert ctx.link.is_moving('gonio_phi')      # the same rotx
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move gonio_phi 90'), ctx)
    assert ctx.conn.of_type('htos_motor_move_completed')[0].endswith(' moving')

    ctx.link.abort_all()
    ctx.link.join()


def test_abort_mid_move_completes_aborted() -> None:
    ctx = FakeContext()
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move sample_x 1'), ctx)
    time.sleep(0.3)
    dhs.abort_all(msg(DcssStoHAbortAll, 'stoh_abort_all soft'), ctx)
    completed = ctx.conn.wait_for('htos_motor_move_completed')
    ctx.link.join()

    assert completed.startswith('htos_motor_move_completed sample_x ')
    assert completed.endswith(' aborted')
    stopped = completed.split(' ')[2]
    assert 0.0 < float(stopped) < 1.0, 'the move must have stopped short of its target'
    # The simulator holds where it stopped rather than drifting on.
    time.sleep(0.2)
    assert ctx.link.position_str('sample_x') == stopped


# ---------------------------------------------------------------------------
# Sets, corrections, shutters, oscillation
# ---------------------------------------------------------------------------
def test_set_motor_position_sends_no_completion() -> None:
    ctx = FakeContext()
    dhs.set_motor_position(
        msg(DcssStoHSetMotorPosition, 'stoh_set_motor_position sample_y 1.5'), ctx)
    assert ctx.conn.lines() == []
    assert ctx.link.position_str('sample_y') == '1.5'
    pose, moving = ctx.link.backend.read()
    assert abs(pose['ty'] - 1.5) < 1e-9 and not moving


def test_correct_motor_position_shifts_without_moving() -> None:
    ctx = FakeContext()
    dhs.correct_motor_position(
        msg(DcssStoHCorrectMotorPosition, 'stoh_correct_motor_position gonio_phi -0.5'),
        ctx)
    assert ctx.conn.lines() == []
    assert ctx.link.position_str('gonio_phi') == '359.5'    # wrapped
    assert ctx.link.position_str('absolute_phi') == '-0.5'  # unwrapped


def test_set_shutter_state_reports_back() -> None:
    ctx = FakeContext()
    dhs.set_shutter_state(
        msg(DcssStoHSetShutterState, 'stoh_set_shutter_state video_trigger open'), ctx)
    assert ctx.conn.lines() == ['htos_report_shutter_state video_trigger open']
    assert ctx.link.backend.video_log == ['open']


def test_set_shutter_state_unknown_is_ignored() -> None:
    ctx = FakeContext()
    dhs.set_shutter_state(
        msg(DcssStoHSetShutterState, 'stoh_set_shutter_state spin_lock closed'), ctx)
    assert ctx.conn.lines() == []


def test_oscillation_opens_then_closes_the_video_trigger() -> None:
    ctx = FakeContext()
    dhs.start_oscillation(
        msg(dhs.DcssStoHStartOscillation,
            'stoh_start_oscillation gonio_phi video_trigger 1.0 0.3'), ctx)
    completed = ctx.conn.wait_for('htos_motor_move_completed')
    ctx.link.join()

    lines = ctx.conn.lines()
    assert lines[0] == 'htos_report_shutter_state video_trigger open'
    assert lines[1] == 'htos_motor_move_started gonio_phi 1'
    assert lines[-2] == 'htos_report_shutter_state video_trigger closed'
    assert completed == 'htos_motor_move_completed gonio_phi 1 normal'
    assert ctx.link.backend.video_log == ['open', 'closed']


def test_oscillation_takes_the_time_dcss_asked_for() -> None:
    ctx = FakeContext()
    start = time.monotonic()
    dhs.start_oscillation(
        msg(dhs.DcssStoHStartOscillation,
            'stoh_start_oscillation gonio_phi video_trigger 1.0 0.5'), ctx)
    ctx.conn.wait_for('htos_motor_move_completed')
    ctx.link.join()
    elapsed = time.monotonic() - start
    # The slew-rate time for 1 deg is under a millisecond; the exposure sets it.
    assert 0.5 <= elapsed < 1.5


# ---------------------------------------------------------------------------
# Production-host refusal
# ---------------------------------------------------------------------------
def test_refuse_production_host() -> None:
    assert dhs.refuse_production('bl831.als.lbl.gov', 'LOCAL')
    assert dhs.refuse_production('dataserver3.bl831.als.lbl.gov', 'LOCAL')
    assert dhs.refuse_production('bl831.als.lbl.gov', 'SIM831') is None
    assert dhs.refuse_production('localhost', 'LOCAL') is None


def test_production_config_exits_2() -> None:
    conf = load_conf()
    conf['dcss']['host'] = 'bl831.als.lbl.gov'
    with tempfile.NamedTemporaryFile('w', suffix='.config', delete=False) as f:
        yaml.safe_dump(conf, f)
        path = f.name
    try:
        run = subprocess.run(
            [sys.executable, DHS_SCRIPT, 'pretend', 'LOCAL', '-c', path],
            cwd=_DHS_DIR, capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    assert run.returncode == 2, run.stdout + run.stderr
    assert 'REFUSING TO START' in run.stdout + run.stderr


# ---------------------------------------------------------------------------
# Real mode: the camera-server client against an in-process stub
# ---------------------------------------------------------------------------
class _StubHandler(BaseHTTPRequestHandler):
    """The four endpoints the DHS needs, and nothing else."""

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def _params(self) -> Dict[str, str]:
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        length = int(self.headers.get('Content-Length', 0))
        if length:
            params.update(dict(urllib.parse.parse_qsl(
                self.rfile.read(length).decode())))
        return params

    def _json(self, doc: dict) -> None:
        body = json.dumps(doc).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _route(self, method: str) -> None:
        srv = self.server
        path = urllib.parse.urlparse(self.path).path
        params = self._params()
        srv.requests.append((method, path, params))
        if path == '/status':
            pose, moving = srv.sample()
            self._json({'positions': pose, 'target': dict(srv.target),
                        'moving': moving})
        elif path == '/move':
            self._json(srv.command_move(params))
        elif path == '/motor':
            self._json(srv.set_motor(params))
        elif path == '/video-trigger':
            srv.video.append(params.get('state', ''))
            self._json({'state': params.get('state', '')})
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        self._route('GET')

    def do_POST(self) -> None:
        self._route('POST')


class StubServer(HTTPServer):
    """A camera server with a linear animation and no pictures."""

    def __init__(self) -> None:
        super().__init__(('127.0.0.1', 0), _StubHandler)
        self.pose = {'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 'rotx': 0.0,
                     'roty': 0.0, 'rotz': 0.0, 'zoom': 1.0}
        self.target = dict(self.pose)
        self.ramp: Optional[Tuple[Dict[str, float], Dict[str, float], float, float]] = None
        self.requests: List[Tuple[str, str, Dict[str, str]]] = []
        self.video: List[str] = []

    @property
    def url(self) -> str:
        return 'http://127.0.0.1:{}'.format(self.server_address[1])

    def sample(self) -> Tuple[Dict[str, float], bool]:
        if self.ramp is None:
            return dict(self.pose), False
        start, target, t0, duration = self.ramp
        now = time.monotonic()
        if duration <= 0 or now >= t0 + duration:
            self.pose.update(target)
            self.ramp = None
            return dict(self.pose), False
        frac = (now - t0) / duration
        for key, value in target.items():
            self.pose[key] = start[key] + (value - start[key]) * frac
        return dict(self.pose), True

    def command_move(self, params: Dict[str, str]) -> Dict[str, float]:
        self.sample()
        moves = {k: float(v) for k, v in params.items() if k in self.pose}
        self.target.update(moves)
        self.ramp = (dict(self.pose), moves, time.monotonic(),
                     float(params.get('duration', 0.0)))
        return dict(self.target)

    def set_motor(self, params: Dict[str, str]) -> Dict[str, float]:
        self.ramp = None
        self.pose.update({k: float(v) for k, v in params.items() if k in self.pose})
        self.target.update(self.pose)
        return dict(self.pose)


def _stub() -> Tuple[StubServer, threading.Thread]:
    server = StubServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_real_mode_drives_the_camera_server_over_http() -> None:
    server, _ = _stub()
    try:
        ctx = FakeContext(load_conf(camera_server=server.url), pretend=False)
        assert isinstance(ctx.link.backend, CameraServerBackend)

        dhs.start_motor_move(
            msg(DcssStoHStartMotorMove, 'stoh_start_motor_move sample_x 0.2'), ctx)
        completed = ctx.conn.wait_for('htos_motor_move_completed')
        ctx.link.join()

        assert completed == 'htos_motor_move_completed sample_x 0.2 normal'
        assert ctx.conn.of_type('htos_update_motor_position'), 'the poll must stream'

        moves = [r for r in server.requests if r[1] == '/move']
        assert moves and moves[0][0] == 'POST'
        # 0.2 mm at 16968 steps/mm and 5000 steps/s.
        assert moves[0][2] == {'tx': '0.2', 'duration': num(0.2 * 16968 / 5000)}
        assert any(r[0] == 'GET' and r[1] == '/status' for r in server.requests)
        assert abs(server.pose['tx'] - 0.2) < 1e-9
    finally:
        server.shutdown()


def test_real_mode_set_and_video_trigger_use_the_right_urls() -> None:
    server, _ = _stub()
    try:
        ctx = FakeContext(load_conf(camera_server=server.url), pretend=False)
        dhs.set_motor_position(
            msg(DcssStoHSetMotorPosition, 'stoh_set_motor_position sample_z -0.75'), ctx)
        dhs.set_shutter_state(
            msg(DcssStoHSetShutterState, 'stoh_set_shutter_state video_trigger open'),
            ctx)

        assert ('POST', '/motor', {'tz': '-0.75'}) in server.requests
        assert ('POST', '/video-trigger', {'state': 'open'}) in server.requests
        assert server.video == ['open']
        assert abs(server.pose['tz'] + 0.75) < 1e-9
    finally:
        server.shutdown()


def test_real_mode_abort_pins_the_pose() -> None:
    server, _ = _stub()
    try:
        ctx = FakeContext(load_conf(camera_server=server.url), pretend=False)
        dhs.start_motor_move(
            msg(DcssStoHStartMotorMove, 'stoh_start_motor_move sample_x 1'), ctx)
        time.sleep(0.3)
        dhs.abort_all(msg(DcssStoHAbortAll, 'stoh_abort_all soft'), ctx)
        completed = ctx.conn.wait_for('htos_motor_move_completed')
        ctx.link.join()

        assert completed.endswith(' aborted')
        stops = [r for r in server.requests
                 if r[1] == '/move' and r[2].get('duration') == '0']
        assert stops, 'abort must send a zero-duration move'
        assert server.ramp is None
    finally:
        server.shutdown()


def test_real_mode_survives_a_dead_camera_server() -> None:
    """A move completes even with nothing listening: dcss is never left waiting."""
    ctx = FakeContext(load_conf(camera_server='http://127.0.0.1:1'), pretend=False)
    dhs.start_motor_move(
        msg(DcssStoHStartMotorMove, 'stoh_start_motor_move sample_x 0.1'), ctx)
    completed = ctx.conn.wait_for('htos_motor_move_completed', timeout=30)
    ctx.link.join(timeout=30)
    assert completed == 'htos_motor_move_completed sample_x 0 normal'


# ---------------------------------------------------------------------------
# Pose model
# ---------------------------------------------------------------------------
def test_axis_wraps_only_a_circle_axis() -> None:
    phi = Axis('gonio_phi', 'rotx', circle=360)
    absolute = Axis('absolute_phi', 'rotx')
    assert phi.report(400.0) == 40.0
    assert phi.report(-10.0) == 350.0
    assert absolute.report(400.0) == 400.0


def test_axis_duration_is_the_database_slew_rate() -> None:
    phi = Axis('gonio_phi', 'rotx', scale=8385, speed=9000000)
    assert abs(phi.duration(360.0) - 360 * 8385 / 9000000) < 1e-9
    assert phi.duration(0.0) == 0.05            # the floor, so a null move still completes


def test_num_never_writes_a_slew_rate_in_exponent_form() -> None:
    assert num(9000000) == '9000000'            # '%g' alone gives 9e+06
    assert num(0.302336) == '0.302336'
    assert num(360.0) == '360'


if __name__ == '__main__':
    import pytest
    raise SystemExit(pytest.main([__file__, '-q']))
