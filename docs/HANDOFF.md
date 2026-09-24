---
project: loop-sim (xtal-loop-sim), bright-field microscope + X-ray simulator for protein crystals in cryo-loops
status: active, served from pre-computed templates with no GPU at runtime; the xtalLoopSimDHS DCSS hardware server drives it from a local dcss (spindle, stage, shuttered oscillation with JPEG push all verified 2026-09-23).
last_verified: 2026-09-23
verify: for f in tests/test_*.py; do .venv/bin/python -m pytest $f -q; done (or `.venv/bin/python -m pytest tests/ -q` on a box with more than 17 GB)
---

# HANDOFF: loop-sim (xtal-loop-sim)

Orientation for whoever inherits loop-sim: what it is, where it stands, what to do
next, and where things are. RUNBOOK.md has the commands.

- [1. What this is](#1-what-this-is)
- [2. Current state](#2-current-state)
- [3. How to resume](#3-how-to-resume)
- [4. Open items](#4-open-items)
- [5. Hazards](#5-hazards)
- [6. Map of the repo](#6-map-of-the-repo)
- [7. Work log](#7-work-log)

## 1. What this is

loop-sim renders a synthetic bright-field microscope image of a protein crystal mounted
in a nylon cryo-loop (Snell refraction at every interface, Beer-Lambert absorption,
Köhler condenser illumination), and separately grid-casts the X-ray beam through the
same scene to report illuminated volume and dose per material. Output is a JPEG or an
AXIS-camera-compatible MJPEG stream that can stand in for a real beamline camera.
`xtalLoopSimDHS/` puts that camera server behind the DCS wire protocol, so dcss and
BluIce can drive the simulated goniometer the way they drive the real one. It
exists for beamline alignment-algorithm development, AI training-data generation, and
dose estimation. James Holton wrote it; Jacob Doughty made the GPU path correct and fast
enough to serve as a live camera, and added the DHS.

## 2. Current state

**Correctness.** The CUDA intersection quadratic runs in float64; the GPU trace is
byte-identical to the float64 CPU reference, and with the objective PSF on, the
delivered image agrees to +/-1 grey level. See DECISIONS §2026-06-25 float64.

**Delivery.** The camera server serves every frame from pre-computed rotation-sweep
templates and never builds one itself: `hampton_300um` and `hampton_300um_realistic`
are current; `mitegen_200um` is stale (built at supersample 1 against a default of 4)
and is served as-is. Sizes and provenance are in DATA.md. The template cache pre-warms
at boot and on scene switch, and the viewer clears the 10 fps goal on all three
candidate beamline hosts: dataserver3 33.9 fps, voltron 30.0, gateway 27.7. See
DECISIONS §2026-08-14 three beamline hosts.

**Realism.** Frames render at the real camera's 704x480 raster; BL831 pixels are 1.110
non-square, so the scene itself stays square-pixel and `field.to_sensor` resamples at
delivery. `renderer/field.py` maps transmittance through a measured camera model
(illumination field, black floor, tone response), and the pin's specular glint is
projected from the scene through its pose (`renderer/pin_projection.py`) rather than
fitted to the picture. The shipped scenes use NA 0.10; NA 0.28 matches photographs
better and switching is Open item 1. See DECISIONS §2026-08-10 camera emulation and
§2026-08-11 NA fork.

**X-ray.** `/xray` renders live on every new pose (torch when the engine is loaded,
else numpy), memoized one pose deep; `render.py --xray` does the same offline, writing
an 8-bit greyscale PNG. `/beam` reports dose and illuminated volume per material. There
is no X-ray frame library, only live rendering; DECISIONS §2026-09-22 has why a template
sweep does not work for radiographs.

**Goniometer protocol.** `xtalLoopSimDHS/` puts the camera server behind the DCS wire
protocol as a hardware DHS under its own name, `xtalLoopSimDHS` (never connected as
`pmac2`, the real goniometer's DHS, which would take its devices unasked), driving it
over localhost HTTP. It has 27 offline tests, and against a local, offline dcss it answers
a spindle move, a stage move and a shuttered oscillation correctly, with the camera
server pushing JPEGs while `video_trigger` is open. BluIce's phi buttons drive it;
click-to-centre needs the `moveSample` operation, which that database lacked. No
database dump ships; the rows dcss needs, and what remains before SIM831 or the
beamline, are in `xtalLoopSimDHS/README.md` "dcss integration: next steps". See
DECISIONS §2026-09-23.

**Environment.** Two venvs: `bash setup_venv.bash` builds the repo's `.venv/`, and
`xtalLoopSimDHS/README.md` "Create the env" builds the DHS's own. Serving from templates
never imports torch; torch is needed only to render.

**Scene generation.** `crystal_harvester` builds a scene from physical parameters and
measures every one back (`validate.py`: volume, rim-on-fiber, watertightness, crystal
placement) before it ships; a scene that fails validation ships nothing. See DECISIONS
§2026-08-07 droplet.

**Branch state.** All work is on `performance-correctness-optimizations`. The branch
exists on GitHub (`origin`), last pushed 2026-08-19, and the local checkout is ahead of
it; nothing is merged into `master`, and James owns the push and merge decisions.
Measure the gaps with `git rev-list --count origin/performance-correctness-optimizations..HEAD`
and `git rev-list --count master..HEAD` rather than trusting a number in these docs.
`master` is the working base, and the stale GitHub default `main` is a divergent
"Initial commit."

## 3. How to resume

1. Run `bash setup_venv.bash` (builds `.venv/` if missing, then runs `pytest tests/`);
   GPU parity tests skip on a CPU-only box.
2. Read `../README.md` for the user-facing pipeline/CLI/server; the module docstrings in
   `loop_sim/` carry the architecture and the server's concurrency contract.
3. For the DHS, read `xtalLoopSimDHS/README.md`, then run its tests:
   `xtalLoopSimDHS/.venv/bin/python -m pytest xtalLoopSimDHS/tests -q`.
4. Decide the push/MR for `performance-correctness-optimizations` with James.
5. Read RUNBOOK §7 "Deploy" before touching beamline deployment.
6. For the AXIS-camera use case, look at `data/frame_library/` before editing the
   renderer; delivery is templates, not live rendering.
7. Read DECISIONS.md by date for rationale; don't re-derive a measurement already there.

## 4. Open items

Priority order:

1. **Switch the scenes to NA 0.28.** Camera-space crystal/background matches photographs
   better at NA 0.28 (0.677 vs the photograph's 0.691) than the shipped NA 0.10 (0.421).
   Switching moves the supersample ceiling and rebuilds all three frame libraries,
   including the ~8 h `hampton_300um_realistic` build; not yet done, since nothing has
   spent that GPU time. See DECISIONS §2026-08-11 NA fork.
2. **The eager fallback only warns.** torch 2.6.0 is pinned in `requirements.txt`, but
   a mis-set stack falls back to eager silently rather than refusing to start; the
   fallback warns (stderr `WARNING:`, `compile_preview.error`) but nothing stops the
   server booting on the wrong stack.
3. **Run `setup_venv.bash` on voltron and dataserver3.** It works on a dev box and has
   never been run on either beamline host.
4. **Frame-library coverage.** The sweep covers rotation only; `zoom` and `tz` (the
   DHS's `sample_z`) are not free the way lateral translation is and would need their
   own sweeps or a live render.
5. **Promote the dimensional check into the test suite.** The 700.0 -> 703.0 µm pin
   measurement is architecture-independent and needs no committed image, only the
   assertion.
6. **Check whether `hampton_300um`'s loop is mislabelled or digitized at another size.**
   Its waypoints span 69x200 µm, not ~300 µm.
7. **Run the DHS against a SIM831 dcss with the real scripting engine.** The offline
   dcss proves the motor and oscillation paths; `moveSample` and `loopFast`
   (click-to-centre and Center Loop) need a database that defines those operations,
   and `loopFast` also needs loopDHS and AutoML. `xtalLoopSimDHS/config/SIM831.config`
   is written for it and untested; the database rows are in `xtalLoopSimDHS/README.md`
   "dcss integration: next steps".
8. **Calibrate the zoom-stop table.** `--camera-zoom` (default `1:1.0,2:0.5,3:0.25`) is
   a placeholder until measured against the three real BL831 sample cameras.

## 5. Hazards

- Material `colour` is an absorption spectrum, not a display tint: a coloured material
  is a strongly absorbing one, so keep water-like solvents near-white.
- Nothing rebuilds a frame library automatically. After editing a file that changes
  template pixels, run `python -m loop_sim.library --verify` (or `--force` to rebuild);
  `tests/test_render_sha_frozen.py` goes red as the reminder. See RUNBOOK "Frame
  libraries".
- Never down-cast the intersection geometry to float32 in `tube.py`, `surface_mesh.py`
  or `engine_torch.py`: it brings back the "hairy" fiber artifact (DECISIONS
  §2026-06-25 float64).
- Never use `torch.compile(mode="reduce-overhead")` in the server: its CUDA-graph
  capture is not thread-safe in `ThreadingHTTPServer` and crashes.
- Do not set `--time` in `tools/run_gpu.slurm`: voltron's GPU queue has no time limit
  and the flag cancels jobs early.
- The XYZ stage rides on the spindle, so motor axes are not image axes. A screen-space
  pan goes through `Rᵀ` (`recenter_target`, `resolve_target`, `pose_crop`) and also
  writes `tz`, so "return to origin" must zero `tx`, `ty` and `tz`.

## 6. Map of the repo

| Path | What it is | Read first |
|---|---|---|
| `render.py` | offline CLI: one optical frame, or `--xray` for a radiograph | README §2 |
| `digitize_fiber.py`, `add_stem.py`, `add_droplet.py`, `add_crystal.py`, `generate_scene.py`, `make_beam_image.py`, `template.yaml` | James's scene pipeline, run in that order; `template.yaml` is `generate_scene.py`'s default camera/material template and no shipped scene reads it | README §3 |
| `setup_venv.bash`, `requirements.txt` | build `.venv/`, the only interpreter | RUNBOOK §1 |
| `loop_sim/scene/` | YAML loader, primitives, `tube.py`, `surface_mesh.py`, `thin_shell.py`, CSG | module docstrings |
| `loop_sim/motors/goniometer.py` | pose and rotation matrices | |
| `loop_sim/renderer/` | `microscope.py` numpy reference tracer; `engine_torch.py` GPU engine; `beam.py` and `xray_torch.py` X-ray; `optics.py` PSF; `field.py` camera model; `pin_projection.py` pin glint; `torch_compat.py` | module docstrings |
| `loop_sim/server/camera_server.py` (+ `static/`) | AXIS-compatible HTTP server, control page, runtime scene switching | README §4 |
| `loop_sim/library/` | frame libraries: `frame_library.py` (`build_library`, `library_status`, `frame_for_angle`), `__main__.py` the `python -m loop_sim.library` CLI | RUNBOOK §3 |
| `crystal_harvester/` | scene generator from physical parameters; `validate.py` measures every scene back before it ships | DECISIONS §2026-08-07 |
| `data/scene_files/` | `hampton_300um.yaml` (tube-based performance baseline, never edit), `hampton_300um_realistic.yaml` (generated), `mitegen_200um.yaml` (mesh-based), `examples/hoop.yaml` (pipeline sample) | DATA.md |
| `data/frame_library/<scene>/` | tracked deliverable: a 360-frame sweep plus `manifest.json` per scene | DATA.md, RUNBOOK §3 |
| `data/real_images/` | 44 tracked BL831 sample-camera photographs, the realism reference; `MANIFEST.tsv` has provenance | `data/real_images/README.md` |
| `tools/` | benchmarks (`bench_frame.py`, `bench_serve.py`), `acceptance_voltron.py` (GO/NO-GO), profilers, James's A/B scripts (each says what it does in its header), `run_gpu.slurm`; all run from the repo root | RUNBOOK §6, §7a |
| `xtalLoopSimDHS/` | the DCSS hardware server: own README, own `.venv`, own `tests/` | `xtalLoopSimDHS/README.md` |
| `tests/` | 21 files; the verify command is in this file's front matter | RUNBOOK §6 |
| `docs/` | this file, `RUNBOOK.md`, `DECISIONS.md`, `DATA.md`, `WORK_LOG.md` | |
| `CLAUDE.md` | a short brief for AI agents | |
| `scratch/` | git-ignored, local only, never mirrored | |
| `/home/jadoughty/projects/loop_sim_MINE/investigation/` | the analysis tree: outside this repo, not in git, mirrored to the gateway only | DATA.md "Known gaps" |

## 7. Work log

See [WORK_LOG.md](WORK_LOG.md) (dated history, newest first).
