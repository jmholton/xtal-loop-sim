# The dcss sandbox seed

`LOCAL_loopsim.txt` is the dcss **server** database seed that gives this DHS its
devices. It is a copy of `LOCAL_full.txt` from the local dcss test rig: a dcss built
from the beamline's `px/batchbuild` source and run from a native-ext4 run-tree at
`~/dcss_local`, with a `LOCAL_TEST/` directory holding its seeds and launch scripts.
The rig is not part of this repository. `LOCAL_full.txt` is the seed `run-local.sh
--restore-full` uses and the one the BluIce GUI needs; `--restore` uses
`LOCAL_min.txt`, a two-device minimum with no motors at all, so the full one is
the base here.

Do not run any of this against a beamline host.

## What was changed

dcss decides which devices a DHS serves from this file: on connect it sends
`stoh_register_<type> <device>` for every device whose `hardwareHost` equals the
connecting DHS's name. So:

- a new type-3 hardware-host row, right after `simDhs`:

  ```
  xtalLoopSimDHS
  3
  localhost 2
  ```

- `gonio_phi`, `sample_x`, `sample_y`, `sample_z` and `shutter` re-pointed from
  `simDhs` to `xtalLoopSimDHS`;
- `absolute_phi`, `detector_trigger` and `video_trigger` added, since
  `LOCAL_full.txt` has none of them. `absolute_phi` is modelled on `gonio_phi`;
  the two shutters are modelled on `shutter`;
- the eight owned devices' numbers set to the beamline's own
  (`dcsconfig/data/BL-831.dat`): scale 8385 steps/deg and circle mode 3
  for `gonio_phi`, scale 16968 steps/mm for the sample stage, positions zeroed
  because the simulator's scene starts at its origin. The generic-sim numbers
  that were there describe a different goniometer. Circle mode is the one that
  has to be right here, because `htos_configure_device` cannot carry it: it is
  what makes dcss treat `gonio_phi` as a circle and send
  `stoh_correct_motor_position`.

Every other device in the file is untouched and still belongs to `simDhs`,
`self` or its original host.

## Restore it and run dcss

From the rig's directory (the parent of `LOCAL_TEST/`), with the rig already built:

```bash
LOCAL_TEST/make-runtree.sh                       # stage the ext4 run-tree
cp <this dir>/LOCAL_loopsim.txt ~/dcss_local/dcss/dbmapfile/
cd ~/dcss_local/dcss/linux64
./dcss LOCAL -r ~/dcss_local/dcss/dbmapfile/LOCAL_loopsim.txt   # build the DB, then exit
cd -
LOCAL_TEST/run-local.sh --bg                     # start dcss; wait ~25 s for all 3 ports
ss -ltn | grep -E ':(14242|14243|14244)'         # 14242 is the hardware port
```

Then start the DHS from its own directory:

```bash
./xtalLoopSimDHS.sh pretend       # no camera server needed
./xtalLoopSimDHS.sh real          # with the camera server running
```

Then drive it without BluIce. `drive_dcss.py` logs in on dcss's GUI port, takes
master, and sends a spindle move, a stage move and a shuttered oscillation,
printing what dcss broadcasts back and the camera server's pose after each:

```bash
../.venv/bin/python sandbox/drive_dcss.py          # from xtalLoopSimDHS/
```

A pass reads `stog_motor_move_completed gonio_phi 90 normal`, `... sample_x 0.2
normal`, `stog_report_shutter_state video_trigger open` then `closed` around the
oscillation, and the camera pose ending at rotx 120, tx 0.2. With the camera
server started as `--jpeg-receiver http://localhost:9000/` and any HTTP server
listening there, the oscillation also delivers ~30 JPEG POSTs a second.

BluIce, in a real terminal (it dies on a non-tty stdout):

```bash
LOCAL_TEST/run-bluice.sh
```

Its Sample-tab phi buttons drive `gonio_phi` through this DHS. Click-to-centre
does not: it starts the `moveSample` scripting-engine operation, which the rig's
`LOCAL_full.txt` seed does not define (`stog_operation_completed moveSample ...
not_exist`). That is a gap in the rig's database, not in the DHS.

## Point BluIce's sample video at the simulator

BluIce reads its video URLs from `dcsconfig/data/LOCAL.config`, which
`make-runtree.sh` ships with every `http://` value blanked. Set the one line:

```
video1.imageUrl=http://localhost:8080/axis-cgi/jpg/image.cgi?camera=1
```

in `~/dcss_local/dcsconfig/data/LOCAL.config` (and in the rig's
`LOCAL_TEST/LOCAL.config` to survive the next `make-runtree.sh`), then restart
BluIce. `video1` is the Sample tab.

The camera server is loop-sim's own; start it from the repository root
(`../../README.md`).

## Notes

- The seed is the **server** database. BluIce builds its `::device::` objects
  from a separate client file, `dcsconfig/data/LOCAL.dat`, so a device missing
  from that file shows as `invalid command name "::device::<x>"` in the GUI even
  when the server knows it. All eight devices here are in the stock `LOCAL.dat`,
  which is a sanitized copy of `BL-831.dat`.
- dcss updates its own record of a motor from the `htos_configure_device` this
  DHS sends at registration, so the numeric line in the seed is what the GUI
  shows only until the DHS connects.
- The rig's scripting engine logs `Error add system_idle: can't access
  ::deviceExclusiveLock::lockList` on every move and `can't read "Al"` at start.
  Both predate this DHS and do not stop a move.
