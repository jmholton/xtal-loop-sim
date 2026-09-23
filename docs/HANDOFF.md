---
project: loop-sim (xtal-loop-sim), bright-field microscope + X-ray simulator for protein crystals in cryo-loops
status: active, served from pre-computed templates with no GPU at runtime; the xtalLoopSimDHS DCSS hardware server drives it from a local dcss (spindle, stage, shuttered oscillation with JPEG push all verified 2026-09-23).
last_verified: 2026-09-23
verify: for f in tests/test_*.py; do .venv/bin/python -m pytest $f -q; done (or `.venv/bin/python -m pytest tests/ -q` on a box with more than 17 GB)
---

# HANDOFF: loop-sim (xtal-loop-sim)

## What this is

loop-sim renders a synthetic bright-field microscope image of a protein crystal mounted
in a nylon cryo-loop (Snell refraction at every interface, Beer-Lambert absorption,
Köhler condenser illumination), and separately grid-casts the X-ray beam through the
same scene to report illuminated volume and dose per material. Output is a JPEG or an
AXIS-camera-compatible MJPEG stream that can stand in for a real beamline camera.
`xtalLoopSimDHS/` puts that camera server behind the DCS wire protocol, so dcss and
BluIce can drive the simulated goniometer the way they drive the real one. It
exists for beamline alignment-algorithm development, AI training-data generation, and
dose estimation. James Holton wrote it; Jacob's contribution was making the GPU path
correct (it produced a "hairy" artifact on the loop fiber) and fast enough to drive as a
live camera.

## Current state

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
fitted to the picture. The NA question is answered: camera-space crystal/background
matches photographs better at NA 0.28 (0.677 vs 0.691) than the shipped NA 0.10 (0.421),
but nothing has switched, since a switch rebuilds all three libraries (see Open items).
See DECISIONS §2026-08-10 camera emulation and §2026-08-11 NA fork.

**X-ray.** `/xray` renders live on every new pose (torch when the engine is loaded,
else numpy), memoized one pose deep; `render.py --xray` does the same offline, writing
an 8-bit greyscale PNG. `/beam` reports dose and illuminated volume per material. The
pre-computed radiograph library is retired: the detector sits at a finite distance, so
translation changes magnification and a template sweep cannot be replayed as crops the
way the optical library is. See DECISIONS §2026-09-22.

**Goniometer protocol.** `xtalLoopSimDHS/` puts the camera server behind the DCS wire
protocol as a hardware DHS named `xtalLoopSimDHS` (never `pmac2`), driving it over
localhost HTTP from a sandbox dcss database. It has 27 offline tests, and against the
local dcss rig it answers a spindle move, a stage move and a shuttered oscillation
correctly, with the camera server pushing JPEGs while `video_trigger` is open
(`xtalLoopSimDHS/sandbox/README.md`). BluIce's phi buttons drive it; click-to-centre
needs the `moveSample` operation, which the rig's database lacks. See
`xtalLoopSimDHS/README.md` for the wire contract.

**Environment.** conda is retired; each project builds its own `.venv/`, with
`setup_venv.bash` at the repo root and its own recipe in `xtalLoopSimDHS/README.md`
"Create the env". Serving from templates never imports torch; torch is needed only to
render.

**Scene generation.** `crystal_harvester` builds a scene from physical parameters and
measures every one back (`validate.py`: volume, rim-on-fiber, watertightness, crystal
placement) before it ships; a scene that fails validation ships nothing. See DECISIONS
§2026-08-07 droplet.

**Branch state.** All work is on `performance-correctness-optimizations`, not pushed to
GitHub; James owns the push/merge decision. Measure depth with
`git rev-list --count master..HEAD` rather than trusting a number in these docs;
`master` is the working base, and the stale GitHub default `main` is a divergent
"Initial commit."

## How to resume

1. Run `bash setup_venv.bash` (builds `.venv/` if missing, then runs `pytest tests/`);
   GPU parity tests skip on a CPU-only box.
2. Read `../README.md` for the user-facing pipeline/CLI/server; the module docstrings in
   `loop_sim/` carry the architecture and the server's concurrency contract.
3. For the DHS, read `xtalLoopSimDHS/README.md`, then run its tests:
   `xtalLoopSimDHS/.venv/bin/python -m pytest xtalLoopSimDHS/tests -q`.
4. Decide the push/MR for `performance-correctness-optimizations` with James.
5. Read RUNBOOK "Deploy on the TITAN V" before touching beamline deployment.
6. For the AXIS-camera use case, look at `data/frame_library/` before editing the
   renderer; delivery is templates, not live rendering.
7. Read DECISIONS.md by date for rationale; don't re-derive a measurement already there.

## Open items

Priority order:

1. **Switch the scenes to NA 0.28.** Camera-space crystal/background matches photographs
   better at NA 0.28 (0.677 vs the photograph's 0.691) than the shipped NA 0.10 (0.421).
   Switching moves the supersample ceiling and rebuilds all three frame libraries,
   including the ~8 h `hampton_300um_realistic` build. Owner's cost call. See DECISIONS
   §2026-08-11 NA fork.
2. **The eager fallback still only warns.** torch 2.6 is now pinned in
   `requirements.txt` via `setup_venv.bash`, but a mis-set stack still falls back to
   eager silently rather than refusing to start; the fallback warns (stderr `WARNING:`,
   `compile_preview.error`) but nothing stops the server booting on the wrong stack.
3. **Run `setup_venv.bash` on voltron and dataserver3.** It replaces the old
   `setup_titan_v_env.bash` and works on a dev box; it has never been run on either
   beamline host.
4. **Frame-library coverage.** The sweep covers rotation only; `zoom` and `tz` (the
   DHS's `sample_z`) are not free the way lateral translation is and would need their
   own sweeps or a live render.
5. **Promote the dimensional check into the test suite.** The 700.0 -> 703.0 µm pin
   measurement is architecture-independent and needs no committed image, only the
   assertion.
6. **Check whether `hampton_300um`'s loop is mislabelled or digitized at another size.**
   Its waypoints span 69x200 µm, not ~300 µm.
7. **Run the DHS against a SIM831 dcss with the real scripting engine.** The local rig
   proves the motor and oscillation paths; `moveSample` and `loopFast` (click-to-centre
   and Center Loop) need a database that defines those operations, and `loopFast` also
   needs loopDHS and AutoML. `xtalLoopSimDHS/config/SIM831.config` is ready for it.
8. **Calibrate the zoom-stop table.** `--camera-zoom` (default `1:1.0,2:0.5,3:0.25`) is
   a placeholder until measured against the three real BL831 sample cameras.
9. **A renderer edit now needs a manual check.** Nothing rebuilds a frame library
   automatically: run `python -m loop_sim.library --verify` (or `--force` to rebuild)
   after editing a file that changes template pixels; see Hazards.

## Hazards

- Material `colour` is an absorption spectrum, not a display tint
  (`mu_per_ch = mu_optical + 30*(1-colour)` per mm, `microscope.py`); a colored material
  is a strongly absorbing one, so keep water-like solvents near-white.
- Nothing rebuilds a frame library automatically. After editing a file that changes
  template pixels, run `python -m loop_sim.library --verify` (or `--force` to rebuild by
  hand); `tests/test_render_sha_frozen.py` goes red as the reminder that a hashed
  tracer file changed. See RUNBOOK "Frame libraries".
- On a 17 GB WSL2 box, run the test suite one file at a time:
  `for f in tests/test_*.py; do .venv/bin/python -m pytest $f -q; done`. A single
  `pytest tests/` invocation exhausted RAM and crashed WSL2 once: the numpy mesh
  reference in `test_torch_render_parity` needed >17 GB at 96x72 on the 50,976-face
  droplet mesh; it now renders at 24x18 (2.1 GB). Expect ~80 s per file, mostly torch's
  import on the DrvFs mount.
- The XYZ stage rides on the spindle, so motor axes are not image axes: build a
  screen-space pan in lab space from the camera `fast`/`slow` axes and map it through
  `Rᵀ` (`recenter_target`, `resolve_target`, `pose_crop`). A screen pan also writes `tz`,
  so "return to origin" must zero `tx`, `ty` and `tz`.
- Never down-cast the intersection geometry to float32 in `tube.py`, `surface_mesh.py`,
  or `engine_torch.py`: it reintroduces the "hairy/spikey" fiber artifact (DECISIONS
  §2026-06-25 float64).
- Never use `torch.compile(mode="reduce-overhead")` in the server: its CUDA-graph
  capture is not thread-safe in `ThreadingHTTPServer` and crashes.
- Do not set `--time` in `tools/run_gpu.slurm`: voltron's GPU queue has no time limit
  and the flag cancels jobs prematurely.
- The MJPEG stream needs both flush mechanisms: each part closes on the boundary written
  after its payload, and new content is followed by one prompt resend a frame-interval
  later. A stream of n frames carries n+1 boundaries, so count payloads, not boundaries,
  in a test client.
- `--device cuda` only changes anything on scenes with `Tube` or `SurfaceMesh` objects; a
  scene of pure primitives/CSG renders CPU==GPU byte-identical, so `--device cuda` is a
  no-op there.
- `mitegen_200um`'s `micromount` is a `ThinShell` wrapping an internal `SurfaceMesh`, so
  it exercises the mesh GPU path (`TSurfaceMesh`) and carries the mesh path's VRAM
  behavior, not the primitives-only path some older notes claimed.
- `scene.yaml`/`loop.yaml` are gitignored, not shipped: render a `data/scene_files/*.yaml`
  or build one via the pipeline (README).
- The analysis tree lives outside this repo, out of git but mirrored to the gateway
  (`/home/jadoughty/projects/loop_sim_MINE/investigation/`). It will not arrive with
  `git clone`.
- `scratch/` at the repo root is git-ignored and local only, not mirrored to the
  gateway; safe to empty any time.
- The fiber beads at the default `n_samples=50` (19.3 µm capsules against a 20.0 µm
  fiber diameter). Raise `n_samples` for fidelity renders.
- On WSL2 there is no CUDA OOM to catch: past the card's VRAM the Windows driver spills
  into system RAM, a 10-50x slowdown that looks like a hang, not a failure. Check
  `nvidia-smi` `memory.used` near the ceiling plus degrading per-item time (RUNBOOK
  "Dev-environment caveat").

## Map of the repo

- Root scripts: `render.py` (CLI; `--xray` for a radiograph) and the pipeline
  `digitize_fiber.py -> add_stem.py -> add_droplet.py -> add_crystal.py ->
  generate_scene.py` (flags unchanged since master; see README), plus
  `make_beam_image.py`, `template.yaml`, `requirements.txt`, `setup_venv.bash`.
- `loop_sim/` is the package: `scene/` (YAML loader, `next_interface`, primitives,
  `tube.py`, `surface_mesh.py`, `thin_shell.py`, CSG), `motors/goniometer.py`,
  `renderer/` (`microscope.py` numpy reference tracer, `beam.py` X-ray, `engine_torch.py`
  GPU-resident engine, `xray_torch.py` its X-ray twin, `optics.py`
  objective PSF, `field.py` the camera model, `pin_projection.py` the pin's projected
  position, `torch_compat.py`), `server/camera_server.py` (AXIS HTTP server, control
  page, runtime scene switching), `library/` (`frame_library.py`: `build_library`,
  `library_status`, `frame_for_angle`; `__main__.py` the `python -m loop_sim.library`
  CLI). X-ray has no library: `/xray` and `render.py --xray` render live.
- `data/scene_files/` holds the bundled scenes: `hampton_300um.yaml` (tube-based, frozen
  performance baseline, never edit it), `hampton_300um_realistic.yaml` (generated
  fidelity scene), `mitegen_200um.yaml` (mesh-based), and `examples/hoop.yaml`, a sample
  digitized hoop for the pipeline. `template.yaml` (camera/material properties, repo
  root) is read by `generate_scene.py`'s `--template` default, not by any shipped scene.
- `data/frame_library/<scene>/` is a tracked deliverable, not build output: a 360-frame
  rotation sweep plus `manifest.json` per scene, for `hampton_300um`,
  `hampton_300um_realistic` and `mitegen_200um` (sizes and status in DATA.md). RUNBOOK
  "Frame libraries" has the per-scene supersample rule.
- `data/real_images/` holds 44 tracked BL831 sample-camera frames, the realism
  reference, not an input. `MANIFEST.tsv` and `README.md` carry provenance and limits.
- `crystal_harvester/` is the scene generator (loop mechanics, crystal habits, pin
  geometry, the closed-form spherical-cap droplet in `droplet.py`); `validate.py`
  measures every emitted scene back before it ships.
- `tools/` holds the benchmarks and beamline scripts: `bench_frame.py`/`bench_serve.py`
  (warm-frame and served-frame benchmarks), `acceptance_voltron.py` (TITAN V acceptance
  test: fps + VRAM + compile check, GO/NO-GO), `profile_gpu.py`/`profile_render.py` and
  their `.bash` wrappers, `test_gpu.bash`, `test_optim.bash`, `debug_optim.bash`,
  `check_diff.bash`, `run_gpu.slurm` (the voltron GPU job, no `--time`). Run from the
  repo root; the scripts `cd` there themselves.
- `xtalLoopSimDHS/` is the DCSS hardware server: own `README.md`, own `.venv`, own
  `tests/`. See "Goniometer protocol" above.
- `tests/` is the verify command: 21 files (invocation in this file's front matter).
- `README.md` (user guide) and `CLAUDE.md` (a short brief for AI agents) sit at repo
  root. `docs/` holds this file plus `RUNBOOK.md`, `DECISIONS.md`, `DATA.md`,
  `WORK_LOG.md`. `scratch/` is git-ignored and local only, not mirrored.

## Work log

See [WORK_LOG.md](WORK_LOG.md) (dated history, newest first).
