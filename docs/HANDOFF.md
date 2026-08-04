---
project: loop-sim (xtal-loop-sim) — bright-field microscope + X-ray simulator for protein crystals in cryo-loops
status: active — camera served from pre-computed templates (no GPU at runtime); scene fidelity is the open front
last_verified: 2026-07-31        # `pytest tests/` = 78 passed in 61 s on this tree (branch performance-correctness-optimizations, RTX 4080 SUPER)
verify: python -m pytest tests/ -q        # 78 tests; "python" = the torch-enabled project interpreter (see docs/RUNBOOK.md "Environment")
---

# HANDOFF — loop-sim (xtal-loop-sim)

<!-- Single entry point. Read this first; it links out to ../README.md (user guide) and
     ../CLAUDE.md (deep architecture/engineering notes) rather than duplicating them.
     This file lives in docs/; repo paths below are relative to the repo root. -->

## What this is, and why

loop-sim renders a **synthetic bright-field microscope image** of a protein crystal
mounted in a nylon cryo-loop (Snell refraction at every interface + Beer-Lambert
absorption + Köhler condenser illumination), and separately **grid-casts the X-ray beam**
through the same scene to report illuminated volume and dose per material. Output is a
JPEG **or** an AXIS-camera-compatible MJPEG HTTP stream that can stand in for a real
beamline camera. It exists for beamline alignment-algorithm development, AI training-data
generation, and dose estimation. James Holton wrote it; Jacob's contribution was making
the GPU path **correct** (it was producing a "hairy" artifact on the loop fiber) and
**fast enough to drive as a live camera** (10 image files/s).

## Current state (2026-07-28)

- **Branch `performance-correctness-optimizations`, 24 commits ahead of `master`, NOT
  pushed to GitHub.** James owns the push/merge decision. (`master` itself is 22 commits
  ahead of the stale GitHub default `main`, which is a divergent "Initial commit" — always
  work from `master`/this branch, never `main`.)
- **Delivery has moved to pre-computed templates, and the camera server now serves from
  them.** The camera is orthographic, so the spindle is the only motor that genuinely
  changes image content; everything else is an image-space transform. `loop_sim/library/`
  renders one 360° sweep per scene into the tracked `frame_library/`, and
  `camera_server` checks for a current library at startup, builds one if it is missing or
  stale, then serves every frame by cropping/scaling/blurring a template. Measured
  with CUDA switched off entirely: **42 ms/frame (24 fps) through a spindle slew**, where
  every frame is a fresh template decode, and **13 ms (75 fps) panning at a fixed angle**.
  A GPU only accelerates *building*. Live rendering is still there behind
  `--templates off` and remains the correctness reference. See DECISIONS.md and RUNBOOK
  "Frame libraries".
- **Templates are supersampled and scene-anchored.** `--supersample` (default 4) divides
  the rendered pixel size, which is what makes `zoom` servable — 4× is where sampling
  critically matches the NA 0.10 objective, so it is a physical ceiling rather than a
  guess. The render window is measured from the scene by a coarse scout sweep rather than
  being a symmetric margin about the origin, because the pin reaches x=6.7 mm against a
  4.7 mm field and a centred window left most of it unrendered.
- **VRAM no longer limits resolution.** The tile clamp in `render_torch` was the ceiling,
  and it was never a correctness constraint — verified byte-exact down to 1000-ray tiles.
  Tiles are now sized at runtime against free VRAM, and rays are built one condenser
  sample at a time. A 14.34 Mpx template renders at **4.7 GB peak** where the old path
  needed ~14 GiB and spilled. **This retires risk A** (see Hazards).
- **Scene fidelity was audited for the first time, and it is the weak half of the
  project.** The imaging chain is dimensionally correct (a 700.0 µm pin measures 703.0 µm
  in the image), but the bundled benchmark scene contains no droplet and no crystal, and
  droplets render opaque for a non-physical reason. See "Scene fidelity" below — this is
  now the highest-value open work, ahead of any further performance tuning.
- **Correctness: DONE + committed.** The float32 "hairy/spikey fiber" GPU artifact is
  fixed — the CUDA intersection quadratic now runs in float64, and the GPU render is
  **byte-identical to the float64 CPU reference**. See DECISIONS.md §"float64 GPU
  intersection" for the root cause.
- **Performance: 10 fps interactive goal MET — and now confirmed on the beamline's TITAN V
  (11.9 fps).** A GPU-resident torch engine (`loop_sim/renderer/engine_torch.py`) runs the
  whole trace on-device; on the 4080 the live server reaches ~25 fps during animated motion
  and ~9.8 fps on the worst-case 10 Hz `/motor` stream, via a flag-gated `torch.compile`
  preview path. `acceptance_voltron.py` measured the deployed path on a real TITAN V at
  **11.9 fps median / 10.1 fps p90 (GO)** — but only with the full software stack (torch 2.6
  + a modern compiler for `torch.compile`); the beamline's default stack falls back to eager
  at 6.3 fps. Settled/offline/`/xray` frames stay bit-exact f64. See RUNBOOK "Deploy on the
  TITAN V" for the exact recipe and DECISIONS.md.
- **Verify: `pytest tests/` = 62 tests, green** on the local torch env (needs a
  torch+CUDA interpreter; GPU-gated parity tests skip on a CPU-only box).
- **Paused with clear open items** (see below) — nothing half-broken; the engine works.

## How to resume

For a stranger picking this up cold:

1. Build the environment and confirm health: follow **`RUNBOOK.md`** → run `python -m pytest
   tests/ -q` (should be 62 green). "python" is the torch-enabled interpreter — beamline:
   `/programs/pytorch/envs/pt/bin/python`; local dev: a conda env with `torch==2.6.0+cu124`.
   On a CPU-only box the GPU parity tests skip, so green there proves less.
2. Understand the design before editing: **`../CLAUDE.md`** is the deep engineering doc
   (architecture map, the float64 precision rationale, the server concurrency model, the
   `next_interface` API contract, the comparison workflow). `../README.md` is the user guide
   (pipeline, CLI, live server, endpoints).
3. Decide the **push/MR** for branch `performance-correctness-optimizations` (James) — the
   work is durable only on this local tree + the gateway mirror until then.
4. **Beamline (TITAN V) readiness is now measured** — the 10 fps result reproduces on the
   real card (11.9 fps) *when the software stack is right* (RUNBOOK "Deploy on the TITAN V").
   The remaining work is packaging that stack, not proving the hardware.
5. **Know which half of the project you are in.** The *renderer* is well verified — 62 tests,
   GPU byte-identical to the CPU reference, and a dimensional check against physics. The
   *scenes* are not: they were never validated until 2026-07-28 and two known-wrong ones
   ship. Speed work is essentially done; fidelity work has barely started. If you are
   deciding where to spend a week, spend it on "Scene fidelity" under Hazards.
6. **For the AXIS-camera use case, look at `frame_library/` before touching the renderer.**
   Delivery has shifted to pre-computed rotation sweeps replayed at request time, which
   sidesteps the frame-rate problem rather than fighting it (RUNBOOK "Frame libraries").

The highest-value open engineering items, in rough priority:
- **Scene fidelity** — the physics is validated but the scenes are not (see "Scene
  fidelity" under Hazards). Settle the camera calibration, decide what the bundled
  `hampton_300um` loop is meant to be, and work out why droplets render opaque. This
  outranks further performance work: the frame rate is already met, the pictures are not
  yet known to be right.
- **Give `TSurfaceMesh` the AABB cull that `TTube` has** — now blocking, not cosmetic. It
  is what stands between the project and rendering *any* scene with a solvent droplet at
  full resolution (risk A below).
- **Package the TITAN V deployment** (the recipe is measured; see RUNBOOK "Deploy on the
  TITAN V"): a torch-2.6 env + a modern compiler for `torch.compile`, plus making the
  silent-fallback-to-eager failure loud so a mis-set stack can't quietly miss 10 fps.
- **Wire `render.py --device cuda` to the resident engine** — it still uses the legacy
  per-object CUDA path; only `camera_server` uses `engine_torch`. Unifying them removes a
  confusing second GPU path.
- **Click-to-recentre bug** (paused) — lands ~100–200 px off, non-deterministically;
  leading hypothesis is a frame/pose lag race. Full resume plan in `../CLAUDE.md` §"Click-to-
  recentre" and DECISIONS.md.

## Hazards & gotchas

**Scene fidelity — the renderer is validated, the scenes are not.** Until 2026-07-28 all
verification was self-consistency (GPU render vs CPU render of the same scene), which
cannot detect a wrong *scene*. What is now measured:

- **The physics and imaging chain are sound.** The pin's ground-truth diameter is 700.0 µm
  and it measures **703.0 µm** in the rendered image at four independent columns — 0.4%,
  the half-pixel edge threshold. Camera model, pixel size, projection and the goniometer
  transform are all correct. This check is architecture-independent (it compares against
  physics, not against another render on the same box) and is the cheapest guard worth
  promoting into the test suite.
- **`scene_files/hampton_300um.yaml` is not a realistic sample.** Its `solvent` is a sphere
  of `radius: 0.0` — there is no droplet; there is no crystal; and its `loop_fiber`
  waypoints span **69 × 200 µm** despite the `300um` name. Every parity test, every
  benchmark and the 11.9 fps acceptance number are measured on this scene. It is a fine
  *performance* baseline and a misleading *fidelity* one. **Two scenes, two jobs:** keep
  this one for speed, and generate a second for picture quality —
  `python -m crystal_harvester.cli --loop-type hampton --loop-size 300 --crystal hexagonal
  -o scene_files/hampton_300um_realistic.yaml`. Do not simply fill the droplet in: that
  makes the scene unrenderable at full resolution (risk A) and invalidates every fps
  number taken on it.
- **Solvent droplets render essentially opaque, and it is not absorption.** Solvent is
  defined with `mu_optical: 0.00`, yet the drop core renders at a mean brightness of
  **0.0405**. Sweeping the objective NA: 0.10 → 0.0405, 0.25 → 0.2005, 0.50 → 0.2613,
  0.90 → 0.2633. A five-fold brightening that saturates by NA 0.5 means the rays are being
  discarded by the **NA collection gate** after refracting through the drop's curvature. A
  real bright-field drop is near background brightness with a dark rim. Two candidates:
  the scenes' low NA values, and `MAX_DEPTH` bounce exhaustion in `microscope.py`.
- **Three different cameras are in circulation.** `template.yaml` — which DATA.md calls the
  authoritative calibration — specifies 0.82 µm pixels and NA 0.28/0.17, but **no shipped
  scene uses it**: the Hampton scenes use 7.4 µm and NA 0.10/0.07, `mitegen_200um` uses
  1.0 µm. Since NA is the knob driving the opaque-droplet result, settling which of these
  matches the real beamline camera is a prerequisite for judging fidelity.
- **The fiber is beaded at the default sampling.** Tubes become `n_samples - 1` capsules;
  at the default `n_samples=50` a 300 µm loop yields 19.3 µm segments against a 20.0 µm
  fiber — capsules as long as they are wide. Raise `n_samples` for fidelity renders.
- **`crystal_harvester` is the trustworthy source of scenes.** Its 300 µm circular loop
  measures 300.4 × 300.0 µm, its droplet mesh spans 300 × 300 × 150 µm, its pin is exactly
  700 µm. The hand-built bundled scenes are the outlier.
- Reusable harnesses for all of the above live in `investigation/scene_survey.py` (renders
  a set of scenes across spindle angles into a labelled contact sheet) and
  `investigation/scene_dimcheck.py` (dimensional + NA-sensitivity checks). Note
  `investigation/` is **not shipped** — see Open questions.

**Deployment reality — the 10 fps target reproduces on the TITAN V (11.9 fps), but only with
the full software stack; the hardware was never the bottleneck, the beamline's default
software stack is.** Measured on a real voltron TITAN V (2026-07-17, `acceptance_voltron.py`,
torch 2.6, devtoolset-7): compiled preview **11.9 fps median / 10.1 fps p90 (GO)**, eager
fallback **6.3 fps**. The three things that decide whether you get 11.9 or 6.3:

- **A — RESOLVED 2026-07-31. VRAM no longer scales with resolution.** The binding
  constraint was never the mesh geometry as such, it was `tile_size = max(tile_size, WH)`
  at `engine_torch.py`, which forced every trace tile to be at least a whole frame. The
  comment above that line already said per-ray results are tile-independent and
  byte-exact, and that is now verified (tiles of 307200 / 100000 / 37649 / 8192 / 1000 rays
  all byte-identical, at φ = 0, 37, 90) and test-guarded. With the clamp relaxed, rays
  built per condenser sample, and the tile sized at runtime against
  `torch.cuda.mem_get_info()`, peak memory is set by the tile rather than the image: a
  14.34 Mpx template renders at 4.7 GB. The same build runs on an 8 GB card and on the
  12 GB TITAN V, in more passes. `TSurfaceMesh` still lacks the AABB cull `TTube` has, so
  mesh scenes remain *slow* — but they are no longer unrenderable, and the cull is back to
  being an optimisation rather than a blocker. The superseded analysis follows.

  *(historical)* **the mesh scene's VRAM is a knife's-edge fit, not a hard OOM.** `TSurfaceMesh` has no
  AABB cull, so `mitegen_200um` (n_cond=1) brute-forces Möller-Trumbore to **~11.1 GB
  reserved**. On the TITAN V's 12 GB that *fits on torch 2.6* (its allocator packs it in),
  but with only ~tens of MB free once the CUDA context is counted — and it **OOMs on torch
  2.0.1** (worse fragmentation). So: fine on a free card with a modern torch, but one
  concurrent allocation or a busier card from the edge. Tube scenes like hampton are
  unaffected (~250 MB, measured 0.33 GB peak at n_cond=7).

  **CORRECTED 2026-07-28 — the tiling escape hatch does not work as written, and the risk
  is wider than mesh-only-on-12 GB.** Measured law: mesh peak ≈ `tile_rays × faces ×
  24 bytes`, times ~6 for the Möller-Trumbore temporaries — driven by tile size and face
  count, **not** by image resolution. `render_torch` clamps `tile_size = max(tile_size,
  W*H)` (`engine_torch.py:1003`), so at 640×480 a tile can never be smaller than 307,200
  rays: 307200 × 2880 faces × 24 B ≈ **19.8 GB predicted, and a 19.78 GiB OOM observed on a
  16 GB card**. Passing `tile_size=32768` therefore does nothing at full resolution — the
  clamp raises it straight back. Dropping resolution does not rescue it either (320×240
  still OOMs; only 160×120 renders). So: any scene with a solvent droplet — including a
  routine `crystal_harvester` Hampton loop, not just `mitegen_200um` — is unrenderable at
  full resolution on current hardware. The durable fix is giving `TSurfaceMesh` the
  `_aabb_survivors` cull `TTube` already has (`engine_torch.py:59`), which attacks `faces`;
  relaxing the clamp is the stopgap.
- **B — `torch.compile` silently falls back to eager, and on the beamline it WILL fail
  without help.** compiled ≈ 11.9 fps vs eager ≈ 6.3 fps. `camera_server.py`
  `_warmup_compiled_preview` (≈line 714) and the runtime path (≈line 478) catch **any**
  exception and degrade to eager while the server keeps booting and streaming — so a stack
  that can't compile silently misses 10 fps. On voltron, compile fails out of the box two
  ways: the pt env's **torch 2.0.1** has an Inductor `pkg_resources` bug, and even on torch
  2.6 the **system gcc 4.8.5** is too old for Inductor's codegen (`stdatomic.h`). Both are
  cleared by the RUNBOOK recipe (torch 2.6 + devtoolset-7). **Make the fallback loud** (fail
  hard, or a persistent warning) so a mis-set stack can't hide, and **pin torch** — it is not
  listed in `requirements.txt` at all (the CUDA build is site-specific: cu124 on the dev box,
  cu118 on voltron).
- **C — voltron's CPU is a 2× Xeon E5-2650 v4 (2016), and it does NOT tank the compiled
  path.** The eager path is dispatch-bound, so the slow CPU shows there (6.3 fps); but the
  compiled path fuses ~8k kernel launches into ~8 graphs, becoming GPU-bound — so the weak
  CPU stops mattering and voltron's compiled 11.9 fps matches the dev box. (This was an open
  worry that the old CPU would drag the frame to ~5–6 fps; measurement refuted it — but
  *only* because compile works, which loops back to risk B.)

**Other traps:**
- **`--device cuda` only does something on scenes with `Tube` or `SurfaceMesh` objects.**
  Those are the only shapes with a CUDA path, so a scene of pure primitives/CSG
  (half-spaces, cylinders, spheres) renders CPU==GPU byte-identical and `--device cuda` is
  a no-op. Use `hampton_300um` (tube-based) to exercise the tube GPU path.
  **`mitegen_200um` is a trap in both directions** (an earlier HANDOFF called it
  "mesh-based", the RUNBOOK called it "primitives/CSG only" — each was half right): its
  `micromount` is a `ThinShell`, which *wraps an internal `SurfaceMesh`*. On the torch
  engine `build_torch_shape` maps it to `TSurfaceMesh`, so it does exercise the mesh path
  and does carry the risk-A memory behaviour. But `scene.py::_build_shape` never passes
  `device=` to `ThinShell`, so on `render.py`'s legacy per-object path the shell's mesh
  stays on the CPU and `--device cuda` genuinely is a no-op. Same scene, different answer
  depending on which engine you are in.
- **Never down-cast the intersection geometry to float32** — it reintroduces the hairy-fiber
  artifact (DECISIONS.md).
- **Never use `torch.compile(mode="reduce-overhead")` in the server** — its CUDA-graph
  capture is not thread-safe in this threaded server and crashes.
- **SLURM: do not set `--time`** in `run_gpu.slurm` — voltron's queue has no time limit and
  the flag cancels jobs prematurely (`../CLAUDE.md`).
- **`scene.yaml`/`loop.yaml` are gitignored, NOT shipped** — render a `scene_files/*.yaml`
  (e.g. `hampton_300um.yaml`, `mitegen_200um.yaml` are complete) or build one via the
  pipeline (README).
- **The `investigation/` dir is NOT shipped** (excluded from the gateway push) — see Open
  questions; the soak-test and profiling harnesses live there.

## Open questions

- **Which camera calibration is real?** `template.yaml` (0.82 µm px, NA 0.28/0.17), the
  Hampton scenes (7.4 µm, NA 0.10/0.07), and `mitegen_200um` (1.0 µm) disagree, and DATA.md
  names `template.yaml` authoritative although nothing uses it. NA drives the opaque-droplet
  result, so this gates fidelity work. Needs a measurement against the real beamline camera.
- **Is the bundled `hampton_300um` loop mislabelled, or digitized at another size?** Its
  waypoints span 69 × 200 µm, not ~300 µm. Worth comparing against the physical part before
  assuming the geometry is wrong rather than the name.
- **Why do droplets go opaque?** The NA sweep shows rays being culled by the collection
  gate rather than absorbed, but even at NA 0.90 the drop only reaches 0.26 brightness —
  so there is likely a second loss mechanism. `MAX_DEPTH` bounce exhaustion in
  `microscope.py` is the first candidate to rule out.
- **Should the dimensional check become a test?** The 700 µm → 703.0 µm pin measurement is
  architecture-independent and would close the "no golden reference / gates are
  architecture-blind" gap DATA.md records. It needs no committed image, only the assertion.
- **Frame-library coverage.** The sweep covers rotation; `zoom` and `tz` are not free the
  way lateral translation is and would need their own sweeps or a live render. Decide
  whether the AXIS consumer needs them before treating the library as complete.
- **Push/merge decision for `performance-correctness-optimizations`** — owner: James. Until
  pushed, the branch lives only on this tree + the gateway mirror.
- **Package the TITAN V software stack.** The 11.9 fps result needs torch 2.6 + a modern
  compiler at runtime (RUNBOOK "Deploy on the TITAN V"), assembled by hand in a venv. A
  reproducible env (a pinned recipe, or a launch wrapper that sets `CC`/`CXX`) would make
  deployment turnkey instead of a five-step manual setup.
- **Make the silent compile-fallback loud** (risk B) and **declare torch 2.6** as required —
  a mis-set stack currently misses 10 fps with no signal.
- **The perf-validation tooling in `investigation/` is push-excluded** (`soak_server.py`,
  profiling experiments); `bench_frame.py` and `acceptance_voltron.py` (repo root) ARE
  shipped. `acceptance_voltron.py` is the one-command GO/NO-GO check — run it on the target
  GPU and read the printed verdict + `acceptance_report.json`.

## Decisions

See **[DECISIONS.md](DECISIONS.md)** — it carries the float64 correctness fix, the
GPU-resident-engine and 10 fps architecture, the TITAN V deployment analysis, and an
**"Already tried"** record holding the measurements behind fp32 preview, CUDA-graph /
reduce-overhead in the threaded server, the Triton megakernel, and dense-mask tracing, so
those numbers don't have to be re-derived.

## Map of the repo

<!-- Paths relative to the repo root. One line per top-level thing;
     ../CLAUDE.md §"Architecture overview" has the full tree. -->

- `render.py` — CLI: load scene, drive goniometer, render to JPEG.
- `loop_sim/` — the package: `scene/` (YAML loader, `next_interface`, primitives, `tube.py`,
  `surface_mesh.py`, `thin_shell.py`, CSG), `motors/goniometer.py`, `renderer/`
  (`microscope.py` numpy reference tracer, `beam.py` X-ray, **`engine_torch.py`**
  GPU-resident engine), `server/camera_server.py` (AXIS HTTP server + control page),
  **`library/`** (pre-computed rotation sweeps — `build_library` / `ensure_library`,
  `frame_for_angle`, `pose_crop`, `zoom_limits`; CLI `python -m loop_sim.library`).
- `frame_library/<scene>/` — **tracked deliverable**, not build output: a rendered 360°
  sweep plus a `manifest.json` per scene. The repo ignores `*.jpg` globally, so
  `.gitignore` carries an explicit re-include for this tree. **Currently shipped:
  `hampton_300um`** (360 frames, 1° steps, `--supersample 4`, 5578×2570 each, 84.9 MB) and
  **`mitegen_200um`** (360 frames, `--supersample 1`, 1840×2296, 26.8 MB) — both verified
  against live renders at 0.00 px. The supersample differs because the two cameras sample
  the same NA 0.10 optics very differently; RUNBOOK "Frame libraries" has the rule. Note
  library size in git (see DATA.md "Known gaps") — `--supersample 2` is 4× cheaper than 4
  if that matters for a future scene.
- `crystal_harvester/` — scene *generator* (James's original geometry code: elastica loop
  mechanics, Bashforth-Adams droplets, crystal habits, pin geometry). Builds a complete
  scene from physical parameters — 8 Hampton loop sizes × 3 shapes, 9 MiTeGen models. This
  is the dimensionally-correct source of scenes; prefer it to hand-editing YAML.
- `digitize_fiber.py → add_stem.py → add_droplet.py → add_crystal.py → generate_scene.py`
  — the pipeline that builds a scene from a real loop image (README).
- `scene_files/` — complete example scenes (`hampton_300um.yaml` tube-based;
  `mitegen_200um.yaml` mesh-based). `template.yaml` — camera/material properties.
- `bench_frame.py` — warm-frame benchmark (`--compiled`, `--fp32`). `acceptance_voltron.py`
  — self-contained TITAN V acceptance test (fps + VRAM + compile check → GO/NO-GO +
  `acceptance_report.json`; auto-picks a free GPU). `run_gpu.slurm` — voltron GPU job (no
  `--time`!). `tests/` — 62 tests (the verify command).
- `README.md` — user guide (repo root). `CLAUDE.md` — deep engineering notes (repo root:
  architecture, precision, concurrency, the recentre bug). `docs/` — the handoff docs
  (this file + `RUNBOOK.md`, `DECISIONS.md`, `DATA.md`). `investigation/` — **not
  shipped**; experiment scratch + perf harnesses.

## Work log (append-only)

- **2026-08-03** — Second scene shipped and two defects fixed. **`mitegen_200um` library
  built**: 360 frames at 1840×2296, 26.8 MB, 103 min at 17.3 s/frame; verified against
  live renders at **0.00 px** sub-pixel offset across φ = 0/90/213 and at zoom 0.5, and
  served with CUDA disabled at 19.8 ms/frame (50 fps). Both scenes now serve without a
  GPU. **Supersample is per-scene and follows the optics**, not preference: the Nyquist
  limit for NA 0.10 is 1.68 µm, so hampton's 7.4 µm pixel under-samples 4.4× (→ 4) while
  mitegen's 1.0 µm pixel already over-samples 1.7× (→ 1); see RUNBOOK "Frame libraries".
  Fixed: (1) **the server rebuilt the library on every default launch** — `--jpeg-quality`
  (what the server *sends*) was being forwarded as the library's `quality` (what gets
  *stored*), different defaults, and quality is a staleness key. Mapping extracted to
  `library_kwargs_from_args()` with a separate `--template-quality`, and a test now asserts
  a no-flag launch cannot invalidate a no-flag build. (2) **`plan_tile_size` rewritten**:
  extrapolating a slope from two 64 k probes to a ~10 M-ray tile under-predicted the mesh
  path by ~2.7 GB and spilled; it now probes *reserved* memory by a measured doubling ramp
  that stops before a rung exceeding budget. (3) `manifest.json` is now written atomically
  and fsynced — a host crash after a long build previously lost it and orphaned the
  library. Suite: **80 passed**.

- **2026-07-31** — The template pipeline was wired end to end, and the VRAM ceiling that
  blocked it turned out not to be real. Audit first: `loop_sim/library/` existed but had
  **zero callers** — the server never imported it, so every frame was still a live
  raytrace despite the `.gitignore` claiming otherwise. Templates were also not
  high-resolution (the builder scaled width/height by the margin but left `pixel_size`
  alone, so 960×720 was pure extra field of view), `crop_window` ignored the
  translation/rotation coupling and had an inverted sign, and `is_current` ignored every
  build parameter. Fixed all of it: `--supersample` (default 4, the zoom ceiling, set by
  the NA 0.10 Rayleigh limit), a scene-anchored render window measured by a scout sweep
  (the pin reaches 6.7 mm against a 4.7 mm field, so a centred window left most of it
  unrendered), `pose_crop` taking a full pose with the φ coupling and a Gaussian defocus,
  and a `TemplateSource` in `camera_server` with `ensure_library` at startup. The web UI
  gained X/Y/Z/φ target boxes and a **GO** button reusing the existing animator.
  **Serving costs 42 ms/frame through a slew and 13 ms panning, CUDA switched off
  entirely** (24 and 75 fps against an original goal of 10). The enabling fix was
  removing the `max(tile_size, WH)` clamp and building rays per condenser sample: peak
  memory dropped from ~14 GiB (which spilled on a 16 GB card) to 4.7 GB at 14.34 Mpx, and
  the tile is now sized at runtime from free VRAM with 20% headroom. Risk A is retired.
  The settle-parity tests were expected to be casualties but were not — the live path is
  preserved, so all 62 stayed green and new template tests were added on top.
  An independent review pass then found a further crop of defects, all of the same
  character — wrong output that looks entirely plausible on screen: the clamp squeezed the
  two axes independently (so every zoom below ~0.75 was served stretched, across the whole
  advertised range), the crop box was rounded to integers (a 0.375 px registration offset
  at 4× plus ~0.1% magnification flicker during a pan), `--axis roty` silently built a
  geometrically wrong library, an interrupted build left a manifest describing
  half-overwritten frames, the VRAM-spill warning was baselined against frame 0 (the
  slowest frame, so it could never fire), the tile budget double-counted the resident set,
  and `plan_tile_size` re-probed on every one of the 360 frames. All fixed and
  test-guarded. The shipped `hampton_300um` library was rebuilt at `--supersample 4`:
  **360 frames of 5578×2570, 84.9 MB, 43 min** at 7.2 s/frame and 4.7 GB peak VRAM.
  Verified against live renders at supersample 4 with **0.00 px sub-pixel offset** at
  φ = 0/45/90/180 and at zoom 2, and the servable zoom range is 0.80–4×.
  Suite: **78 passed in 61 s**. **Next:** build libraries for the remaining scenes;
  `TSurfaceMesh` AABB cull is now an optimisation, not a blocker; scene fidelity (the
  hazard block above) is untouched and remains the highest-value open work.
- **2026-07-28** — First scene-fidelity audit, and a change of delivery architecture.
  Performance work stopped; the question became whether the pictures are *right*. Built
  survey and measurement harnesses (`investigation/scene_survey.py`,
  `investigation/scene_dimcheck.py`) and rendered a spread of Hampton and MiTeGen mounts
  across spindle angles. **The imaging chain validated** — a 700.0 µm pin measures 703.0 µm
  at four independent columns. **The scenes did not:** the benchmark scene has a
  zero-radius droplet, no crystal and a 69 × 200 µm loop despite its `300um` name; droplets
  render opaque via the NA collection gate rather than absorption (drop core 0.0405 at
  NA 0.10, 0.2005 at 0.25, saturating ~0.26); three mutually inconsistent camera
  calibrations are in circulation; the fiber is beaded because capsule length ≈ fiber
  diameter at the default `n_samples`. Also corrected risk A — the documented
  `tile_size=32768` escape hatch is defeated by a `max(tile_size, W*H)` clamp, and *any*
  droplet-bearing scene (not just `mitegen_200um`) is unrenderable at full resolution,
  so the `TSurfaceMesh` AABB cull is now blocking rather than cosmetic. Separately,
  measured that lateral translation is an **exact** image shift (max pixel difference
  0.000000) — panning is a crop — which made a **pre-computed rotation sweep** the better
  architecture than chasing live frame rate. Added `loop_sim/library/` plus the tracked
  `frame_library/` output and a `.gitignore` re-include. Recorded the team's
  "Lagrange polynomial waypoints" idea under Already Tried: it is the current design, and
  the global form is deliberately capped at degree 3 to avoid Runge oscillations.
  **Next:** settle the camera calibration, land the AABB cull, decide what the bundled
  `hampton_300um` loop is meant to be; James's call on pushing the branch.
- **2026-07-23** — `contacts:` roster dropped from the HANDOFF front-matter (`96356be`),
  following a knowledge-transfer protocol change: a standing roster ages into
  mis-attribution, so people are named inline where they own a specific artifact instead.
- **2026-07-17** — TITAN V acceptance measured. Added `acceptance_voltron.py` (`df33635`), a
  self-contained one-command GO/NO-GO harness (fps + VRAM + compile check), and ran it on a
  real voltron TITAN V. Result: compiled preview **11.9 fps median / 10.1 fps p90 (GO)**,
  eager 6.3 fps; mesh scene fits torch 2.6 at ~11.1 GB (knife's edge), OOMs torch 2.0.1. The
  10 fps goal reproduces on the target **with** the stack in RUNBOOK "Deploy on the TITAN V"
  (torch 2.6 + devtoolset-7); the beamline's default stack (torch 2.0.1, gcc 4.8.5) can't
  compile and silently falls back to 6.3 fps. Resolved the "voltron CPU unknown" question
  (2× Xeon E5-2650 v4) and corrected two earlier predictions (mesh isn't a hard OOM; the old
  CPU doesn't tank the compiled path). **Next:** package the stack + make the silent fallback
  loud; James's call on pushing the branch.
- **2026-07-16** — RUNBOOK.md written (env from nothing → run → verify → voltron/SLURM →
  rollback), completing the doc set. Verify command re-run on this tree: **62 passed in
  67 s** (RTX 4080 SUPER). **Next:** de-risk the TITAN V deployment (risks A/B/C above);
  James's call on pushing the branch.
- **2026-07-15** — Scaffolded HANDOFF.md + DECISIONS.md as the first knowledge-transfer
  flush of the loop-sim memories (bug-hunt root cause, 10 fps architecture, rejected
  approaches, TITAN V risks — none of which were written down in the repo before; they
  lived only in the contractor's agent memory + the push-excluded `investigation/` dir).
  Verified: git HEAD `edefca0`, tree clean, `pytest tests/` green (62 tests) on the local
  torch env.
- **2026-07-06** — 10 fps interactive goal met (RTX 4080 SUPER): compiled preview path,
  single-flight render owner, sync-starved trace loop; fp32 preview tried and rejected.
  62 tests green. (James owns push.)
- **2026-06-25** — float64 GPU correctness fix (hairy fiber gone, GPU byte-identical to CPU)
  + GPU-resident torch engine + threaded camera server. Branch created off `master`.
