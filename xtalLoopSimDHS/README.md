# xtalLoopSimDHS: loop-sim as a DCSS hardware server

A DCS hardware server (DHS) that puts the loop-sim camera server behind the DCS
wire protocol, so dcss and BluIce drive the simulated goniometer exactly as they
drive the real one. It owns the sample motors and the shutters in a sandbox
database and turns each DCSS motor command into a camera-server HTTP call. It
renders nothing: everything about the picture, the scene and the optics belongs
to loop-sim ([`../README.md`](../README.md), [`../docs/`](../docs/)).

It is not pmac2. It announces itself as `xtalLoopSimDHS`, and a sandbox database
re-points the devices at that name ([`sandbox/`](sandbox/)). A config naming the
production dcss host is refused at startup unless the beamline is `SIM831`.

- [Files](#files)
- [Devices](#devices)
- [Wire contract](#wire-contract)
- [What the camera server must provide](#what-the-camera-server-must-provide)
- [Config keys](#config-keys)
- [Create the env](#create-the-env)
- [Run](#run)
- [Test](#test)
- [The sandbox database](#the-sandbox-database)
- [Notes](#notes)

## Files

| file | what |
|------|------|
| `xtalLoopSimDHS.sh` | the launcher: `./xtalLoopSimDHS.sh real\|pretend [LOCAL\|SIM831] [DHS options]` |
| `xtal_loop_sim_DHS.py` | the DHS: the DCSS handlers and the three messages pydhsfw is missing |
| `sim_link.py` | the pose model, the move engine and the camera-server HTTP client |
| `config/LOCAL.config` | config for the local dcss sandbox (localhost:14242) |
| `config/SIM831.config` | config for a SIM831 dcss instance (localhost:15242) |
| `sandbox/LOCAL_loopsim.txt` | the dcss database seed that gives this DHS the devices |
| `sandbox/README.md` | how to restore that seed, run dcss, and point BluIce's video at the simulator |
| `tests/test_dhs.py` | the offline suite: handlers, HTTP client, refusal |
| `requirements.txt` | pip deps (pydhsfw is installed separately, from the local checkout) |

## Devices

Scale, speed, acceleration and the limits are the beamline's own, from
`dcsconfig/data/BL-831.dat`; the DHS reports them verbatim so dcss's picture of
the motor matches the real goniometer's. `sim_key` is the camera server's pose
key. The values live in `config/<beamline>.config`, never in `loop_sim/`.

| DCSS motor | sim key | units | scale (steps/unit) | speed (steps/s) | accel | limits | notes |
|---|---|---|---|---|---|---|---|
| `gonio_phi` | `rotx` | deg | 8385 | 9000000 | 20000 | off | dcss circle mode: reported wrapped to [0, 360) |
| `absolute_phi` | `rotx` | deg | 8385 | 3000000 | 20000 | off | the same spindle, never wrapped |
| `sample_x` | `tx` | mm | 16968 | 5000 | 100 | off | `sign` flips the direction |
| `sample_y` | `ty` | mm | 16968 | 5000 | 100 | off | `sign` flips the direction |
| `sample_z` | `tz` | mm | 16968 | 5000 | 100 | off | `sign` flips the direction |

Backlash is 0 on every axis and every limit and lock flag is off.

| DCSS shutter | what it does |
|---|---|
| `shutter` | state only |
| `detector_trigger` | state only |
| `video_trigger` | `open` also POSTs `/video-trigger?state=open`, the simulated AXIS push |

`gonio_phi` and `absolute_phi` are one spindle, so a move on either answers
`moving` on the other. `camera_zoom` is deliberately absent: it is a
dcss-internal pseudo-motor and nothing registers it to a DHS.

## Wire contract

**Accepted from dcss.** Anything else is dropped by pydhsfw's message factory;
a motor or shutter this DHS does not own is logged and ignored.

| message | what happens |
|---|---|
| `stoc_send_client_type` | reply `htos_client_is_hardware xtalLoopSimDHS` within 1 s |
| `stoh_register_real_motor <motor> <hw>` | reply `htos_configure_device` |
| `stoh_register_shutter <shutter> <state> <hw>` | adopt `<state>`, reply `htos_configure_shutter` + `htos_report_shutter_state` |
| `stoh_register_operation <op> <hw>` | logged; this DHS implements no operations |
| `stoh_configure_real_motor <motor> <pos> <upper> <lower> <scale> <speed> <accel> <backlash> <flags...>` | adopt speed, acceleration and limits |
| `stoh_start_motor_move <motor> <position>` | the move stream below |
| `stoh_set_motor_position <motor> <position>` | redefine the position, no motion, **no completion** |
| `stoh_correct_motor_position <motor> <correction>` | shift the position by `<correction>`, no motion, no completion |
| `stoh_set_shutter_state <shutter> <open\|closed>` | set it, reply `htos_report_shutter_state` |
| `stoh_start_oscillation <motor> <shutter> <delta_deg> <time_s>` | one shuttered turn, below |
| `stoh_abort_all <hard\|soft>` | stop the simulator; every move in flight completes `aborted` |

**Sent to dcss.**

```
htos_client_is_hardware xtalLoopSimDHS
htos_configure_device <motor> <pos> <upper> <lower> <scale> <speed> <accel> <backlash> <lowerOn> <upperOn> <lockOn> <backlashOn> <reverseOn>
htos_configure_shutter <shutter> open closed <state>
htos_report_shutter_state <shutter> <open|closed>
htos_motor_move_started <motor> <target>
htos_update_motor_position <motor> <pos> moving
htos_motor_move_completed <motor> <pos> normal|aborted|moving
```

`htos_configure_device` carries 13 tokens after the command name. dcss will not
accept a move for a motor it has not had one for, so it goes out on every
`stoh_register_real_motor`. A registration of `gonio_phi` from
`config/LOCAL.config` sends:

```
htos_configure_device gonio_phi 0 360 360 8385 9000000 20000 0 0 0 0 0 0
```

**A move.** Every commanded move ends in a completion, so dcss is never left
waiting, including when the camera server is unreachable.

```
htos_motor_move_started gonio_phi 40
htos_update_motor_position gonio_phi 144.152 moving
htos_update_motor_position gonio_phi 271.63 moving
htos_motor_move_completed gonio_phi 40 normal
```

Duration is `|delta| * scale / speed`, the time the real motor's step generator
would need at the database slew rate, with the acceleration term ignored and a
floor of 0.05 s. While the move runs the position is sampled every
`poll_interval_s` and reported when it has changed; a move shorter than one poll
completes with no updates, which is correct, since nothing changed that dcss had
not been told. A move commanded on a motor already moving is answered at once
with `htos_motor_move_completed <motor> <pos> moving` and nothing starts.
Positions are strings: DCS serialises with `' '.join(args)` and raises on a raw
float.

**An oscillation**, the message dcss's loop-centering and exposure scripts send.
The turn takes the time dcss asked for, not the slew-rate time, because the
exposure is what sets it:

```
htos_report_shutter_state video_trigger open
htos_motor_move_started gonio_phi 1
htos_update_motor_position gonio_phi 0.33 moving
htos_report_shutter_state video_trigger closed
htos_motor_move_completed gonio_phi 1 normal
```

**pydhsfw gaps.** Three messages are defined in `xtal_loop_sim_DHS.py` rather
than by editing the read-only framework, the way `lukepi_dhs.py` adds the string
messages it is missing: inbound `stoh_start_oscillation`, outbound
`htos_configure_shutter`, and a two-token `htos_report_shutter_state` (pydhsfw's
class always appends a third `result` token; dcss parses it as optional and the
Tcl pmac2DHS sends two).

## What the camera server must provide

The DHS is written against these four, and the camera server
(`loop_sim/server/camera_server.py`) provides all of them; the URLs are
config-driven regardless.

| call | used for |
|---|---|
| `GET /status` -> `{"positions": {...}, "target": {...}, "moving": bool}` | every position report and the end of a move |
| `POST /move?<key>=<v>&duration=<s>` | a commanded move, and `duration=0` to stop one |
| `POST /motor?<key>=<v>` | `stoh_set_motor_position` and `stoh_correct_motor_position` |
| `POST /video-trigger?state=open\|closed` | the `video_trigger` shutter |

If the camera server is unreachable, a real-mode read falls back to the last
known pose, standing still, so nothing hangs and nothing crashes -- the
position simply never changes. `pretend` mode needs none of these endpoints.

## Config keys

`config/<beamline>.config` is YAML. Every key is annotated in
`config/LOCAL.config`; the shape is:

| key | what |
|---|---|
| `dcss.host`, `dcss.port` | the dcss hardware port to connect to |
| `xtal_loop_sim.dhs_name` | the name dcss knows; must match a type-3 row in its database |
| `xtal_loop_sim.camera_server` | the loop-sim camera server's base URL (real mode only) |
| `xtal_loop_sim.http_timeout_s` | per-request connect+read timeout |
| `xtal_loop_sim.poll_interval_s` | seconds between position reports while a move runs |
| `xtal_loop_sim.min_move_s` | floor on a move's duration |
| `xtal_loop_sim.video_shutter` | which shutter pushes to the camera server |
| `xtal_loop_sim.motors.<name>` | `sim_key`, `units`, `scale`, `speed`, `accel`, `backlash`, `upper`, `lower`, the five flags, `circle`, `sign`, `position` |
| `xtal_loop_sim.shutters.<name>` | the state it starts in |

## Create the env

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-deps -e /path/to/pydhsfw   # not on PyPI
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "import pydhsfw.dcss; print('pydhsfw ok')"
```

`--no-deps` is deliberate: pydhsfw's `install_requires` names opencv, matplotlib,
scipy and requests, none of which this DHS touches. pydhsfw is the beamline's own
DHS framework (github.com/tetrahedron-technologies/pydhsfw); on the beamline a
checkout lives in `/home/classen/pydhsfw`.

## Run

```bash
./xtalLoopSimDHS.sh pretend            # no camera server; the same DCSS traffic
./xtalLoopSimDHS.sh real               # drive the camera server named in the config
./xtalLoopSimDHS.sh real SIM831 -vv    # another dcss, debug logging
```

The beamline word picks `config/<name>.config` and defaults to `LOCAL`. The
launcher exits 2 on a bad mode, 3 with no `.venv`, and 4 with no config file. The
DHS itself exits 2 when the config names the production dcss host and the
beamline is not `SIM831`.

`pretend` models the pose here, with the same timing and the same reported
trajectory, and contacts nothing. It is what the tests drive and what to use
when the camera server is not running.

## Test

```bash
.venv/bin/python -m pytest tests -q
```

No dcss, no camera server and no beamline: the handlers run against a fake
connection that records real pydhsfw serialization, and the real-mode HTTP
client runs against an in-process stub of the four endpoints above. The suite
takes about 20 s, most of it real move timing.

## The sandbox database

dcss decides which devices a DHS serves from its own database: on connect it
sends `stoh_register_<type> <device>` for every device whose `hardwareHost`
matches the connecting DHS's name. So the sandbox database needs one host row
for this DHS and each of its devices pointed at it:

```
xtalLoopSimDHS
3
localhost 2
```

and, in each device's block, `hardwareHost` changed to `xtalLoopSimDHS`.
`sandbox/LOCAL_loopsim.txt` is that seed, ready to restore, and
`sandbox/README.md` has the commands.

## Notes

- **A set moves the scene.** `stoh_set_motor_position` and
  `stoh_correct_motor_position` place the pose directly, so what dcss is told and
  what the picture shows can never drift apart. The real pmac2DHS instead shifts
  a software offset and leaves the spindle alone. Neither sends a completion.
- **A circle axis is driven by the delta** from its reported position, so the
  pose keeps counting past 360 and `absolute_phi` reads the turn count the
  spindle has actually made. dcss picks the short way round itself, with
  `stoh_correct_motor_position`.
- **dcss ignores `htos_configure_shutter`.** The el9-era `handle_hardware_message`
  has no case for it and logs `Unrecognized command from hardware client`. It is
  sent anyway, because the Tcl pmac2DHS sends it and an older or newer dcss may
  want it; the shutter state that actually lands in dcss comes from the
  `htos_report_shutter_state` beside it.
- **One instance.** dcss rejects a second connection under the same DHS name, so
  a stray instance locks out the real one.
