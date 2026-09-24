# -*- coding: utf-8 -*-
"""The xtalLoopSim DHS: the loop-sim camera server as a DCSS hardware server.

loop-sim serves an AXIS-compatible view of a crystal loop and takes goniometer
poses over HTTP. This DHS puts that simulator behind the DCS wire protocol, so
dcss and BluIce drive it exactly as they drive the real goniometer: it owns the
sample motors and the shutters in whichever dcss database names it as their
hardware host, and turns each DCSS motor command into a camera-server call. It
renders nothing.

It serves the same device names as pmac2, the real goniometer's DHS, but connects
under its own name, `xtalLoopSimDHS`, so it can only receive them from a database
edited to hand them over (README.md "dcss integration"); connecting as pmac2
would take the real devices with no edit. A config naming the production dcss
host is refused at startup unless the beamline is SIM831.

Device state lives in sim_link.py. The handlers below only translate:

  * real motors - stoh_register_real_motor / stoh_configure_real_motor /
                  stoh_start_motor_move / stoh_set_motor_position /
                  stoh_correct_motor_position / stoh_abort_all, replying with
                  htos_configure_device, htos_motor_move_started,
                  htos_update_motor_position and htos_motor_move_completed.
  * shutters    - stoh_register_shutter / stoh_set_shutter_state, replying with
                  htos_configure_shutter and htos_report_shutter_state.
  * oscillation - stoh_start_oscillation, the message dcss's loop-centering
                  script sends: open the shutter, turn, close it.

pydhsfw has no stoh_start_oscillation and no htos_configure_shutter, and its
htos_report_shutter_state always carries a third token; all three are defined
below rather than by editing the read-only framework (see the NOTE).

Wire contract, device table and config keys: README.md.

Run:
    .venv/bin/python xtal_loop_sim_DHS.py real LOCAL -v
    .venv/bin/python xtal_loop_sim_DHS.py pretend LOCAL -v    # no camera server
"""
import logging
import os
import signal
import sys
from typing import Callable, Optional

import yaml

from pydhsfw.messages import register_message
from pydhsfw.processors import register_message_handler, Context
from pydhsfw.dcss import (
    DcssContext,
    DcssStoCMessage,
    DcssStoCSendClientType,
    DcssHtoSClientIsHardware,
    DcssHtoSMessage,
    DcssStoHRegisterRealMotor,
    DcssStoHRegisterShutter,
    DcssStoHRegisterOperation,
    DcssStoHConfigureRealMotor,
    DcssStoHStartMotorMove,
    DcssStoHSetMotorPosition,
    DcssStoHCorrectMotorPosition,
    DcssStoHSetShutterState,
    DcssStoHAbortAll,
    DcssHtoSConfigureDevice,
    DcssHtoSMotorMoveStarted,
    DcssHtoSUpdateMotorPosition,
    DcssHtoSMotorMoveCompleted,
    DcssHtoSReportShutterState,
)
from pydhsfw.dhs import Dhs, DhsInit, DhsStart, DhsContext
from pydhsfw.connection import Connection

from sim_link import SimLink, build_link, num


# NOTE: these three belong in pydhsfw. Added here rather than by editing the
# framework checkout, which this DHS does not own.
# ---------------------------------------------------------------------------
@register_message('stoh_start_oscillation', 'dcss')
class DcssStoHStartOscillation(DcssStoCMessage):
    """Server To Hardware Start Oscillation: one shuttered turn of one motor.

    'stoh_start_oscillation <motor> <shutter> <delta_deg> <time_s>', sent by
    dcss's exposure and loop-centering scripts.
    """

    def __init__(self, split: list) -> None:
        super().__init__(split)

    @property
    def motor_name(self) -> str:
        return self.args[0]

    @property
    def shutter_name(self) -> str:
        return self.args[1]

    @property
    def delta_deg(self) -> str:
        return self.args[2]

    @property
    def time_s(self) -> str:
        return self.args[3]


@register_message('htos_configure_shutter')
class DcssHtoSConfigureShutter(DcssHtoSMessage):
    """'htos_configure_shutter <name> <open_state> <closed_state> <state>', the
    line the Tcl pmac2DHS sends for each shutter once dcss has registered it."""

    def __init__(self, shutter_name: str, open_state: str, closed_state: str,
                 state: str) -> None:
        super().__init__()
        self._split_msg = [self.get_type_id(), shutter_name, open_state,
                           closed_state, state]


class HtoSReportShutterState(DcssHtoSReportShutterState):
    """Like pydhsfw's class but two tokens, the form pmac2DHS sends and dcss
    parses: 'htos_report_shutter_state <name> <state>'. pydhsfw always appends a
    third `result` token."""

    def __init__(self, shutter_name: str, shutter_state: str) -> None:
        super().__init__(shutter_name, shutter_state, '')
        self._split_msg = [self.get_type_id(), shutter_name, shutter_state]


_logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # anchors config/, whatever the cwd
CONFIG_DIR = os.path.join(_SCRIPT_DIR, 'config')
DCSS_CONN = 'dcss_conn'             # the one connection this DHS has

# The startup mode word, like every other DHS here (`pmac1DHS.tcl real BL-831`).
MODES = ('real', 'pretend')
DEFAULT_BEAMLINE = 'LOCAL'

# The production database owns the real goniometer. A DHS that attached to it
# under this name would fight pmac2 for the sample motors.
PRODUCTION_HOST = 'bl831.als.lbl.gov'
PRODUCTION_EXEMPT_BEAMLINE = 'SIM831'

SHUTTER_OPEN = 'open'
SHUTTER_CLOSED = 'closed'


def _link(context: DcssContext) -> SimLink:
    """The pose model and camera-server link built at dhs_init."""
    return context.state['link']


def _dcss(context: Context) -> Connection:
    """The DCSS connection; `.send(msg)` talks to dcss."""
    return context.get_connection(DCSS_CONN)


def config_path(beamline: str) -> str:
    """config/<beamline>.config, beside this script."""
    return os.path.join(CONFIG_DIR, '{}.config'.format(beamline))


def refuse_production(host: str, beamline: str) -> Optional[str]:
    """The reason to refuse this dcss host, or None to go ahead.

    Only a beamline named SIM831 may name the production host, and only because
    the SIM831 instance lives there. Everything else is localhost.
    """
    if PRODUCTION_HOST in (host or '') and beamline != PRODUCTION_EXEMPT_BEAMLINE:
        return ('dcss host {!r} is the production beamline and beamline is {!r}; '
                'only {} may name {}'.format(host, beamline,
                                             PRODUCTION_EXEMPT_BEAMLINE,
                                             PRODUCTION_HOST))
    return None


# ---------------------------------------------------------------------------
# Lifecycle handlers
# ---------------------------------------------------------------------------
@register_message_handler('dhs_init')
def dhs_init(message: DhsInit, context: DhsContext) -> None:
    """Logging, config and the device table, before dcss is contacted."""
    parser = message.parser
    parser.add_argument('mode', choices=MODES, metavar='real|pretend',
                        help='real drives the camera server over HTTP; pretend '
                             'models the pose internally, with the same timing')
    parser.add_argument('beamline', nargs='?', default=DEFAULT_BEAMLINE,
                        help='config/<beamline>.config (default: {})'.format(
                            DEFAULT_BEAMLINE))
    parser.add_argument('--version', action='version', version='xtalLoopSimDHS 0.1')
    parser.add_argument('-v', '--verbose', dest='loglevel',
                        action='store_const', const=logging.INFO,
                        help='set loglevel to INFO')
    parser.add_argument('-vv', '--very-verbose', dest='loglevel',
                        action='store_const', const=logging.DEBUG,
                        help='set loglevel to DEBUG')
    parser.add_argument('-c', '--config', dest='config_file', default=None,
                        help='config file (default: config/<beamline>.config)')
    args = parser.parse_args(message.args)

    loglevel = args.loglevel or logging.INFO
    logformat = '[%(asctime)s] %(levelname)s:%(name)s:%(funcName)s():%(lineno)d - %(message)s'
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat,
        datefmt='%Y-%m-%d %H:%M:%S')

    config_file = args.config_file or config_path(args.beamline)
    _logger.info(f'Loading config: {config_file}')
    if not os.path.isfile(config_file):
        _logger.critical(f'no config file {config_file} - exiting')
        sys.exit(4)
    with open(config_file, 'r') as f:
        conf = yaml.safe_load(f) or {}

    dcss_conf = conf.get('dcss', {}) or {}
    host = dcss_conf.get('host', 'localhost')
    port = dcss_conf.get('port', 14242)
    refusal = refuse_production(host, args.beamline)
    if refusal:
        _logger.critical('REFUSING TO START: %s', refusal)
        sys.exit(2)

    section = conf.get('xtal_loop_sim', {}) or {}
    dhs_name = section.get('dhs_name', 'xtalLoopSimDHS')
    pretend = args.mode == 'pretend'
    link = build_link(conf, pretend)

    context.config = conf
    context.state = {
        'url': f'dcss://{host}:{port}',
        'dhs_name': dhs_name,
        'link': link,
        'pretend': pretend,
    }

    _logger.info('motors: %s', ', '.join(
        '{} -> {} ({}, sign {:+g})'.format(a.name, a.sim_key, a.units, a.sign)
        for a in link.axes()))
    _logger.info('shutters: %s', ', '.join(
        '{}={}'.format(n, s) for n, s in sorted(link.shutters().items())))
    if pretend:
        _logger.warning('MODE: PRETEND - the pose is modelled here and no camera '
                        'server is contacted; the DCSS traffic is the same')
    else:
        _logger.info('MODE: REAL - moves are commanded on the camera server at %s',
                     section.get('camera_server'))
    _logger.info(
        f'DCSS at {host}:{port}, DHS name {dhs_name!r} - NOTE dcss rejects a '
        f'name that is not a hardware-host row in its database, and a second '
        f'connection under the same name')


@register_message_handler('dhs_start')
def dhs_start(message: DhsStart, context: DhsContext) -> None:
    """Open the DCSS connection."""
    context.create_connection(DCSS_CONN, 'dcss', context.state['url'])
    _dcss(context).connect()


# ---------------------------------------------------------------------------
# DCSS handshake + registration
# ---------------------------------------------------------------------------
@register_message_handler('stoc_send_client_type')
def send_client_type(message: DcssStoCSendClientType, context: Context) -> None:
    """Reply with the DHS name within 1 s or dcss drops the connection."""
    name = context.state['dhs_name']
    _logger.info(f'Registering with DCSS as hardware client {name!r}')
    _dcss(context).send(DcssHtoSClientIsHardware(name))


@register_message_handler('stoh_register_real_motor')
def register_real_motor(message: DcssStoHRegisterRealMotor, context: DcssContext) -> None:
    """Report the motor's configuration. Until dcss has it the motor stays
    unconfigured and every GUI move bails with 'DHS is offline'."""
    name, hw_name = message.motor_name, message.motor_hardwareName
    link = _link(context)
    axis = link.get_axis(name)
    if axis is None:
        _logger.warning(f'DCSS registered motor {name} (hw={hw_name}), not in the '
                        f'config device table - moves will be ignored')
        return
    settings = axis.settings(link.position(name))
    _dcss(context).send(DcssHtoSConfigureDevice(name, settings))
    _logger.info(f'DCSS registered motor {name} (hw={hw_name}) -> {axis.sim_key}; '
                 f'sent htos_configure_device {name} {settings}')


@register_message_handler('stoh_register_shutter')
def register_shutter(message: DcssStoHRegisterShutter, context: DcssContext) -> None:
    """Adopt the state dcss remembers, then report the shutter back."""
    name = message.shutter_name
    link = _link(context)
    if link.get_shutter(name) is None:
        _logger.warning(f'DCSS registered shutter {name}, not in the config '
                        f'device table - state changes will be ignored')
        return
    state = link.set_shutter(name, message.shutter_status)
    conn = _dcss(context)
    conn.send(DcssHtoSConfigureShutter(name, SHUTTER_OPEN, SHUTTER_CLOSED, state))
    conn.send(HtoSReportShutterState(name, state))
    _logger.info(f'DCSS registered shutter {name} ({state})')


@register_message_handler('stoh_register_operation')
def register_operation(message: DcssStoHRegisterOperation, context: DcssContext) -> None:
    """This DHS implements no operations; log what dcss thinks it owns."""
    _logger.info(f'DCSS registered operation {message.operation_name} - '
                 f'this DHS has no handler for it')


@register_message_handler('stoh_configure_real_motor')
def configure_real_motor(message: DcssStoHConfigureRealMotor, context: DcssContext) -> None:
    """dcss pushes a motor's configuration back; adopt what changes a move."""
    axis = _link(context).get_axis(message.motor_name)
    if axis is None:
        return
    for attr, value in (('speed', message.motor_speed),
                        ('accel', message.motor_acceleration),
                        ('upper', message.motor_upperLimit),
                        ('lower', message.motor_lowerLimit)):
        try:
            setattr(axis, attr, float(value))
        except (TypeError, ValueError):
            pass
    _logger.info(
        f'DCSS configured motor {message.motor_name}: pos={message.motor_position} '
        f'limits=[{message.motor_lowerLimit}, {message.motor_upperLimit}] '
        f'scale={message.motor_scaleFactor} speed={message.motor_speed} '
        f'accel={message.motor_acceleration}')


# ---------------------------------------------------------------------------
# Motor handlers
# ---------------------------------------------------------------------------
def _move(context: DcssContext, motor_name: str, target: float,
          duration: Optional[float] = None,
          on_done: Optional[Callable[[str, str], None]] = None) -> bool:
    """Start a move and stream it to dcss. False means nothing started.

    `on_done(position, state)` runs before the completion goes out, which is how
    an oscillation closes its shutter at the end of the turn.
    """
    conn = _dcss(context)
    link = _link(context)

    def complete(position: str, state: str) -> None:
        if on_done is not None:
            on_done(position, state)
        conn.send(DcssHtoSMotorMoveCompleted(motor_name, position, state))

    return link.start_move(
        motor_name, target,
        on_start=lambda pos: conn.send(DcssHtoSMotorMoveStarted(motor_name, pos)),
        on_update=lambda pos: conn.send(
            DcssHtoSUpdateMotorPosition(motor_name, pos, 'moving')),
        on_complete=complete,
        duration=duration)


@register_message_handler('stoh_start_motor_move')
def start_motor_move(message: DcssStoHStartMotorMove, context: DcssContext) -> None:
    """Move one motor. A motor already moving is answered at once with `moving`."""
    motor_name = message.motor_name
    link = _link(context)
    if link.get_axis(motor_name) is None:
        _logger.warning(f'stoh_start_motor_move for unknown motor {motor_name}')
        return
    if not _move(context, motor_name, message.motor_position):
        _logger.warning(f'motor {motor_name}: move already in progress')
        _dcss(context).send(DcssHtoSMotorMoveCompleted(
            motor_name, link.position_str(motor_name), 'moving'))


@register_message_handler('stoh_set_motor_position')
def set_motor_position(message: DcssStoHSetMotorPosition, context: DcssContext) -> None:
    """Redefine where a motor is. A set is not a move: dcss expects no completion."""
    motor_name = message.motor_name
    link = _link(context)
    if link.get_axis(motor_name) is None:
        _logger.warning(f'stoh_set_motor_position for unknown motor {motor_name}')
        return
    position = link.set_position(motor_name, message.motor_position)
    _logger.info(f'set {motor_name} position to {position} (no motion)')


@register_message_handler('stoh_correct_motor_position')
def correct_motor_position(message: DcssStoHCorrectMotorPosition,
                           context: DcssContext) -> None:
    """dcss's circle correction: shift the position, send no completion."""
    motor_name = message.motor_name
    link = _link(context)
    if link.get_axis(motor_name) is None:
        _logger.warning(f'stoh_correct_motor_position for unknown motor {motor_name}')
        return
    position = link.correct_position(motor_name, message.motor_correction)
    _logger.info(f'corrected {motor_name} by {message.motor_correction} '
                 f'-> {position} (no motion)')


@register_message_handler('stoh_abort_all')
def abort_all(message: DcssStoHAbortAll, context: DcssContext) -> None:
    """Stop pressed: halt the simulator; every in-flight move completes `aborted`.

    Reads nothing from the message: an exception here kills pydhsfw's dispatcher
    for good, and a bare stoh_abort_all has no argument.
    """
    _logger.warning('stoh_abort_all - stopping every move')
    _link(context).abort_all()


# ---------------------------------------------------------------------------
# Shutter handlers
# ---------------------------------------------------------------------------
@register_message_handler('stoh_set_shutter_state')
def set_shutter_state(message: DcssStoHSetShutterState, context: DcssContext) -> None:
    """Set a shutter and report it. The video shutter also reaches the simulator."""
    name = message.shutter_name
    link = _link(context)
    if link.get_shutter(name) is None:
        _logger.warning(f'stoh_set_shutter_state for unknown shutter {name}')
        return
    state = link.set_shutter(name, message.shutter_state)
    _dcss(context).send(HtoSReportShutterState(name, state))
    _logger.info(f'shutter {name} -> {state}')


@register_message_handler('stoh_start_oscillation')
def start_oscillation(message: DcssStoHStartOscillation, context: DcssContext) -> None:
    """One shuttered turn: open, move `delta` over `time`, close, complete.

    The turn takes the time dcss asked for, not the slew-rate time, because the
    exposure is what sets it.
    """
    motor_name = message.motor_name
    shutter_name = message.shutter_name
    conn = _dcss(context)
    link = _link(context)
    axis = link.get_axis(motor_name)
    if axis is None:
        _logger.warning(f'stoh_start_oscillation for unknown motor {motor_name}')
        return
    try:
        delta = float(message.delta_deg)
        seconds = float(message.time_s)
    except (TypeError, ValueError):
        _logger.warning(f'stoh_start_oscillation {motor_name}: bad delta/time '
                        f'{message.delta_deg!r} {message.time_s!r}')
        return

    target = link.position(motor_name) + delta

    def close_shutter(position: str, state: str) -> None:
        if link.get_shutter(shutter_name) is not None:
            conn.send(HtoSReportShutterState(
                shutter_name, link.set_shutter(shutter_name, SHUTTER_CLOSED)))

    if link.get_shutter(shutter_name) is not None:
        conn.send(HtoSReportShutterState(
            shutter_name, link.set_shutter(shutter_name, SHUTTER_OPEN)))
    _logger.info(f'oscillation {motor_name} {num(delta)} deg over {num(seconds)} s '
                 f'with {shutter_name}')

    if not _move(context, motor_name, target, duration=seconds,
                 on_done=close_shutter):
        _logger.warning(f'motor {motor_name}: oscillation refused, move in progress')
        close_shutter(link.position_str(motor_name), 'moving')
        conn.send(DcssHtoSMotorMoveCompleted(
            motor_name, link.position_str(motor_name), 'moving'))


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
# Guarded under __main__ so tests can import the module without starting a live
# DHS. The @register_message_handler decorators run at import time either way.
if __name__ == '__main__':
    dhs = Dhs()
    dhs.start()
    dhs.wait({signal.SIGINT, signal.SIGTERM})
