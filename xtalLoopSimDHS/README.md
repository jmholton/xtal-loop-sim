# xtalLoopSimDHS: loop-sim as a DCSS hardware server

A DCS hardware server (DHS) that puts the loop-sim camera server behind the DCS
wire protocol, so dcss and BluIce drive the simulated goniometer exactly as they
drive the real one. It owns the sample motors and the shutters in whichever dcss
database names it as their hardware host, and turns each DCSS motor command into
a camera-server HTTP call. It renders nothing: everything about the picture, the
scene and the optics belongs to loop-sim ([`../README.md`](../README.md),
[`../docs/`](../docs/)).

It serves the same device names as `pmac2`, the real goniometer's DHS, but it
connects under its own name, `xtalLoopSimDHS`, and needs a database edit to receive
them (the rows are under "dcss integration"). Connecting as `pmac2` would take the
real goniometer's devices from any database with no edit and leave every log line
ambiguous about which process moved a motor. A config naming the production dcss
host is refused at startup unless the beamline is `SIM831`.

Status: driven only by a local, offline dcss so far (2026-09-23), never by SIM831
or the beamline's dcss. What remains is under "dcss integration: next steps".

- [1. Files](#1-files)
- [2. Devices](#2-devices)
- [3. Wire contract](#3-wire-contract)
- [4. What the camera server must provide](#4-what-the-camera-server-must-provide)
- [5. Config keys](#5-config-keys)
- [6. Create the env](#6-create-the-env)
- [7. Run](#7-run)
- [8. Test](#8-test)
- [9. dcss integration: next steps](#9-dcss-integration-next-steps)
  - [9a. The database rows](#9a-the-database-rows)
  - [9b. Running it against a dcss](#9b-running-it-against-a-dcss)
  - [9c. What is still open](#9c-what-is-still-open)
- [10. Notes](#10-notes)

## 1. Files

| file | what |
|------|------|
| `xtalLoopSimDHS.sh` | the launcher: `./xtalLoopSimDHS.sh real\|pretend [LOCAL\|SIM831] [DHS options]` |
| `xtal_loop_sim_DHS.py` | the DHS: the DCSS handlers and the three messages pydhsfw is missing |
| `sim_link.py` | the pose model, the move engine and the camera-server HTTP client |
| `config/LOCAL.config` | config for a local, offline dcss (localhost:14242) |
| `config/SIM831.config` | config for a SIM831 dcss instance (localhost:15242) |
| `drive_dcss.py` | sends a spindle move, a stage move and an oscillation through dcss's GUI port, no BluIce needed |
| `tests/test_dhs.py` | the offline suite: handlers, HTTP client, refusal |
| `requirements.txt` | pip deps (pydhsfw is installed separately, from the local checkout) |

## 2. Devices

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
| `video_trigger` | every state change also POSTs `/video-trigger?state=open\|closed`, the simulated AXIS push |

`gonio_phi` and `absolute_phi` are one spindle, so a move on either answers
`moving` on the other. `camera_zoom` is deliberately absent: it is a
dcss-internal pseudo-motor and nothing registers it to a DHS.

## 3. Wire contract

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

**pydhsfw gaps.** Three messages pydhsfw lacks are defined in `xtal_loop_sim_DHS.py`
rather than by editing the framework checkout: inbound `stoh_start_oscillation`, outbound
`htos_configure_shutter`, and a two-token `htos_report_shutter_state` (pydhsfw's
class always appends a third `result` token; dcss parses it as optional and the
Tcl pmac2DHS sends two).

## 4. What the camera server must provide

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

## 5. Config keys

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

## 6. Create the env

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

## 7. Run

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

## 8. Test

```bash
.venv/bin/python -m pytest tests -q
```

No dcss, no camera server and no beamline: the handlers run against a fake
connection that records real pydhsfw serialization, and the real-mode HTTP
client runs against an in-process stub of the four endpoints above. The suite
takes about 20 s, most of it real move timing.

## 9. dcss integration: next steps

The DHS has been driven by one dcss: a local, offline build of `px/batchbuild`
whose database was hand-edited as below. Against it, dcss registered the five
motors and three shutters, and a spindle move, a stage move and a shuttered
oscillation each completed `normal` with the camera server's pose following and
JPEGs pushed while `video_trigger` was open. BluIce's phi buttons drove the
spindle. Nothing has been tried against SIM831 or the beamline's dcss, and no
database dump is shipped here: a dump goes stale the day the real one is edited,
so the rows are listed instead.

### 9a. The database rows

dcss decides which devices a DHS serves from its own database: on connect it
sends `stoh_register_<type> <device>` for every device whose hardware host is
the connecting DHS's name. In the dump-file format (`dcsconfig/data/BL-831.dat`
and its siblings), the DHS needs one type-3 host row:

```
xtalLoopSimDHS
3
localhost 2
```

and its eight devices pointed at it. These are the blocks as they were on the
offline dcss. The numeric line is the beamline's own for each motor (scale,
speed, acceleration; limits off; positions zero because the scene starts at its
origin); the last token on `gonio_phi` is circle mode 3, which has to be right in
the database because `htos_configure_device` cannot carry it, and it is what
makes dcss treat the spindle as a circle and send `stoh_correct_motor_position`.

```
gonio_phi
1
xtalLoopSimDHS gonio_phi
0 1 1 1 1
0 1 1 1 1
0
0.000000 360.000000 360.000000 8385.000000 9000000 20000 0 0 0 0 0 0 3

absolute_phi
1
xtalLoopSimDHS absolute_phi
0 1 1 1 1
0 1 1 1 1
0
0.000000 0.000000 0.000000 8385.000000 3000000 20000 0 0 0 0 0 0 0

sample_x
1
xtalLoopSimDHS sample_x
0 1 1 1 1
0 1 1 1 1
0
0.000000 0.000000 0.000000 16968.000000 5000 100 0 0 0 0 0 0 0
```

`sample_y` and `sample_z` are `sample_x` with the name changed. The shutters:

```
shutter
6
xtalLoopSimDHS 1 shutter
0 1 1 1 1
0 1 1 1 1
```

`detector_trigger` and `video_trigger` are `shutter` with the name changed. On a
database that already has these devices (every beamline database does), only the
host token changes: from `pmac2` (or `simDhs` on an offline dcss) to
`xtalLoopSimDHS`, on the eight devices and nowhere else.

Two things the dump does not show. The dump is the **server** database; BluIce
builds its own device objects from the client file (`dcsconfig/data/<BEAMLINE>.dat`),
so a device missing there shows as `invalid command name "::device::<x>"` in the
GUI even when the server knows it. And dcss overwrites its record of a motor from
the `htos_configure_device` this DHS sends at registration, so the numeric line
matters only until the DHS connects.

### 9b. Running it against a dcss

1. Build the database from the edited dump the way the site normally does
   (`./dcss <BEAMLINE> -r <dump>` builds and exits), then start dcss. It listens
   on three ports; on the offline dcss they were 14242 hardware, 14243 GUI, 14244.
2. Start the camera server from the repository root (`../README.md`), with
   `--jpeg-receiver <url>` if something should receive the pushed frames.
3. `./xtalLoopSimDHS.sh real` (or `real SIM831`), see "Run".
4. Without BluIce: `.venv/bin/python drive_dcss.py [--dcss host:port] [--camera url]`
   logs in on the GUI port, takes master (which steals it from any BluIce that is
   connected), and sends the three commands. A pass reads
   `stog_motor_move_completed gonio_phi 90 normal`, `... sample_x 0.2 normal`,
   `stog_report_shutter_state video_trigger open` then `closed` around the
   oscillation, and the camera pose ends at rotx 120, tx 0.2.
5. With BluIce: set the Sample-tab video source to the camera server in the
   client config and restart BluIce:

   ```
   video1.imageUrl=http://<camera host>:8081/axis-cgi/jpg/image.cgi?camera=1
   ```

### 9c. What is still open

- **SIM831.** `config/SIM831.config` is written for it (port 15242) and has never
  been used. A SIM831 run is the first thing to do; it exercises the real
  scripting engine.
- **Click-to-centre and Center Loop.** Both start scripting-engine operations
  (`moveSample`, `loopFast`) that the offline database did not define, so they
  returned `not_exist`. `loopFast` also needs loopDHS and AutoML running.
- **The beamline's dcss.** Re-pointing the eight devices there takes them away
  from `pmac2`, and the DHS refuses a config naming the production host unless
  started as `SIM831`. Nothing about this path has been tested.

## 10. Notes

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
