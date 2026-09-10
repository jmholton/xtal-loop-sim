---
project: loop-sim (xtal-loop-sim), bright-field microscope + X-ray simulator for protein crystals in cryo-loops
status: active, served from pre-computed templates with no GPU at runtime, usable interactively at 10+ fps on three beamline hosts.
last_verified: 2026-09-10
verify: python -m pytest tests/ -q
---

# HANDOFF: loop-sim (xtal-loop-sim)

## What this is

loop-sim renders a synthetic bright-field microscope image of a protein crystal mounted
in a nylon cryo-loop (Snell refraction at every interface, Beer-Lambert absorption,
Köhler condenser illumination), and separately grid-casts the X-ray beam through the
same scene to report illuminated volume and dose per material. Output is a JPEG or an
AXIS-camera-compatible MJPEG stream that can stand in for a real beamline camera. It
exists for beamline alignment-algorithm development, AI training-data generation, and
dose estimation. James Holton wrote it; Jacob's contribution was making the GPU path
correct (it produced a "hairy" artifact on the loop fiber) and fast enough to drive as a
live camera.

## Current state

**Correctness.** The CUDA intersection quadratic runs in float64; the GPU trace is
byte-identical to the float64 CPU reference, and with the objective PSF on, the
delivered image agrees to +/-1 grey level. See DECISIONS §2026-06-25 float64.

**Delivery.** The camera server serves every frame from pre-computed rotation-sweep
templates; no GPU runs at request time, only when a library is built. All three frame
libraries (`hampton_300um`, `hampton_300um_realistic`, `mitegen_200um`) are current,
~37 MB total (DATA.md). Both the optical and X-ray template caches pre-warm at boot and
on scene switch, and the viewer clears the 10 fps goal on all three candidate beamline
hosts: dataserver3 33.9 fps, voltron 30.0, gateway 27.7. See DECISIONS §2026-08-14
templates.

**Realism.** Frames render at the real camera's 704x480 raster; BL831 pixels are 1.110
non-square, so the scene itself stays square-pixel and `field.to_sensor` resamples at
delivery. `renderer/field.py` maps transmittance through a measured camera model
(illumination field, black floor, tone response), and the pin's specular glint is
projected from the scene through its pose (`renderer/pin_projection.py`) rather than
fitted to the picture. The NA question is answered: camera-space crystal/background
matches photographs better at NA 0.28 (0.677 vs 0.691) than the shipped NA 0.10 (0.421),
but nothing has switched, since a switch rebuilds all three libraries (see Open items).
See DECISIONS §2026-08-10 camera emulation and §2026-08-11 NA fork.

**X-ray.** `/xray` (single-shot) and `/xray-stream` (push, ~28 fps in motion) both serve
from a pre-computed 16-bit radiograph library (`xray_library/`, its own
module/root/render_sha) when one exists; `/beam` reports dose and illuminated volume per
material. See DECISIONS §2026-08-19 push stream.

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

1. Run `python -m pytest tests/ -q` (RUNBOOK "Environment" has the interpreter path);
   GPU parity tests skip on a CPU-only box.
2. Read `../README.md` for the user-facing pipeline/CLI/server; the module docstrings in
   `loop_sim/` carry the architecture and the server's concurrency contract.
3. Decide the push/MR for `performance-correctness-optimizations` with James.
4. Read RUNBOOK "Deploy on the TITAN V" before touching beamline deployment.
5. For the AXIS-camera use case, look at `frame_library/` before editing the renderer;
   delivery is templates, not live rendering.
6. Read DECISIONS.md by date for rationale; don't re-derive a measurement already there.

## Open items

Priority order:

1. **Switch the scenes to NA 0.28.** Camera-space crystal/background matches photographs
   better at NA 0.28 (0.677 vs the photograph's 0.691) than the shipped NA 0.10 (0.421).
   Switching moves the supersample ceiling and rebuilds all three frame libraries,
   including the ~8 h `hampton_300um_realistic` build. Owner's cost call. See DECISIONS
   §2026-08-11 NA fork.
2. **Launch-vs-switch grading disagreement.** `CameraServer.__init__` grades a library
   through `build_params` (fills `supersample=4` unless told otherwise); the tab strip
   grades through `_grading_params` (drops `supersample` unless the operator passed it).
   Only `mitegen_200um` (supersample 1) is out of sync now, so a bare launch on it still
   rebuilds a complete, current library. The fix is one line: route `__init__` through
   `_grading_params` too.
3. **Declare torch 2.6 as a hard requirement.** A mis-set stack still falls back to eager
   silently rather than refusing to start; the fallback now warns (stderr `WARNING:`,
   `compile_preview.error`), but nothing stops the server booting on the wrong stack.
4. **Run `setup_titan_v_env.bash` on voltron.** It scripts the torch 2.6 cu118 venv +
   devtoolset-7 recipe and ends in `acceptance_voltron.py`; written and syntax-checked
   locally, never run on real hardware.
5. **Frame-library coverage.** The sweep covers rotation only; `zoom` and `tz` are not
   free the way lateral translation is and would need their own sweeps or a live render.
6. **Promote the dimensional check into the test suite.** The 700.0 -> 703.0 µm pin
   measurement is architecture-independent and needs no committed image, only the
   assertion.
7. **Check whether `hampton_300um`'s loop is mislabelled or digitized at another size.**
   Its waypoints span 69x200 µm, not ~300 µm.

## Hazards

- Material `colour` is an absorption spectrum, not a display tint
  (`mu_per_ch = mu_optical + 30*(1-colour)` per mm, `microscope.py`); a colored material
  is a strongly absorbing one, so keep water-like solvents near-white.
- The launch path can rebuild what the serving path already serves: `mitegen_200um`
  needs `--supersample 1` on a bare launch, or `CameraServer.__init__` deletes its
  manifest and starts a ~1.9 h rebuild. `git checkout -- frame_library/<scene>/` recovers
  a deleted manifest (RUNBOOK "Frame libraries").
- The XYZ stage rides on the spindle, so motor axes are not image axes: build a
  screen-space pan in lab space from the camera `fast`/`slow` axes and map it through
  `Rᵀ` (`recenter_target`, `resolve_target`, `pose_crop`). A screen pan also writes `tz`,
  so "return to origin" must zero `tx`, `ty` and `tz`.
- Never down-cast the intersection geometry to float32 in `tube.py`, `surface_mesh.py`,
  or `engine_torch.py`: it reintroduces the "hairy/spikey" fiber artifact (DECISIONS
  §2026-06-25 float64).
- Never use `torch.compile(mode="reduce-overhead")` in the server: its CUDA-graph
  capture is not thread-safe in `ThreadingHTTPServer` and crashes.
- Do not set `--time` in `run_gpu.slurm`: voltron's GPU queue has no time limit and the
  flag cancels jobs prematurely.
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
- `scene.yaml`/`loop.yaml` are gitignored, not shipped: render a `scene_files/*.yaml` or
  build one via the pipeline (README).
- The analysis tree lives outside this repo, out of git but mirrored to the gateway
  (`/home/jadoughty/projects/loop_sim_MINE/investigation/`). It will not arrive with
  `git clone`.
- `scratch/` at the repo root is git-ignored but mirrored to the gateway as its own
  ungated pair; safe to empty any time.
- The fiber beads at the default `n_samples=50` (19.3 µm capsules against a 20.0 µm
  fiber diameter). Raise `n_samples` for fidelity renders.
- `_RENDER_SOURCES` (`library/frame_library.py`) names `renderer/` files one by one and
  globs `scene/*.py`. A new module that changes template pixels must be added to it by
  name, or its edits never grade a library stale; a serve-time module must stay out of
  the hash, which is why `field.py`, `pin_projection.py` and `torch_compat.py` live in
  `renderer/` rather than `scene/`.
- On WSL2 there is no CUDA OOM to catch: past the card's VRAM the Windows driver spills
  into system RAM, a 10-50x slowdown that looks like a hang, not a failure. Check
  `nvidia-smi` `memory.used` near the ceiling plus degrading per-item time (RUNBOOK
  "Dev-environment caveat").

## Map of the repo

- `render.py` is the CLI: load a scene, drive the goniometer, render to JPEG.
- `loop_sim/` is the package: `scene/` (YAML loader, `next_interface`, primitives,
  `tube.py`, `surface_mesh.py`, `thin_shell.py`, CSG), `motors/goniometer.py`,
  `renderer/` (`microscope.py` numpy reference tracer, `beam.py` X-ray,
  `engine_torch.py` GPU-resident engine, `xray_torch.py` its X-ray twin, `optics.py`
  objective PSF, `field.py` the camera model, `pin_projection.py` the pin's projected
  position), `server/camera_server.py` (AXIS HTTP server, control page, runtime scene
  switching), `library/` (`build_library`/`ensure_library`, `library_status`,
  `frame_for_angle`; `xray_library.py` the X-ray analogue).
- `frame_library/<scene>/` is a tracked deliverable, not build output: a 360-frame rotation
  sweep plus `manifest.json` per scene. `hampton_300um` 15 MB, `hampton_300um_realistic`
  14 MB (both `--supersample 4`), `mitegen_200um` 9 MB (`--supersample 1`); ~37 MB total,
  all `current`. RUNBOOK "Frame libraries" has the per-scene supersample rule.
- `xray_library/<scene>/` is the X-ray analogue: a 360-frame radiograph sweep, 16-bit
  greyscale PNG, its own `render_sha`/root, tracked and committed, ~39 MB
  (`hampton_300um` 12 MB, `hampton_300um_realistic` 10 MB, `mitegen_200um` 18 MB), all
  current.
- `crystal_harvester/` is the scene generator (loop mechanics, crystal habits, pin geometry,
  the closed-form spherical-cap droplet in `droplet.py`); `validate.py` measures every
  emitted scene back before it ships.
- `digitize_fiber.py -> add_stem.py -> add_droplet.py -> add_crystal.py ->
  generate_scene.py` is the pipeline that builds a scene from a real loop image (README).
- `scene_files/` holds complete example scenes: `hampton_300um.yaml` (tube-based, frozen
  performance baseline, never edit it), `hampton_300um_realistic.yaml` (generated
  fidelity scene), `mitegen_200um.yaml` (mesh-based). `template.yaml` (camera/material
  properties) is read by nothing.
- `real_images/` holds 44 tracked BL831 sample-camera frames, the realism reference, not an
  input. `MANIFEST.tsv` and `README.md` carry provenance and limits.
- `bench_frame.py` and `bench_serve.py` are warm-frame and served-frame benchmarks.
  `acceptance_voltron.py` is a self-contained TITAN V acceptance test (fps + VRAM +
  compile check, GO/NO-GO). `run_gpu.slurm` is the voltron GPU job (no `--time`), CPU vs
  GPU on `hampton_300um`. `tests/` is the verify command.
- `README.md` (user guide) and `CLAUDE.md` (a short brief for AI agents) sit at repo root.
  `docs/` holds this file plus `RUNBOOK.md`, `DECISIONS.md`, `DATA.md`, `WORK_LOG.md`.
  `scratch/` is git-ignored, mirrored local output.

## Work log

See [WORK_LOG.md](WORK_LOG.md) (dated history, newest first).
