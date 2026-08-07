---
project: loop-sim (xtal-loop-sim) — bright-field microscope + X-ray simulator for protein crystals in cryo-loops
status: active — camera served from pre-computed templates (no GPU at runtime) and usable interactively; scene geometry/fidelity is the open front
last_verified: 2026-08-07        # `pytest tests/` = 149 passed in 114 s on this tree (branch performance-correctness-optimizations, RTX 4080 SUPER)
verify: python -m pytest tests/ -q        # 149 tests; "python" = the torch-enabled project interpreter (see docs/RUNBOOK.md "Environment")
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

## Current state (2026-08-07)

- **The interactive path now works, and it had never been driven by a person before.**
  Templates made frames cheap in 2026-07-31, but the control page was still unusable: the
  stream was capped at 5 fps by a stale default, the last frame of every move arrived a
  second late, the translate pad moved the sample along motor axes rather than image axes
  at any φ≠0, out-of-range entries were accepted and silently ignored, and a 15° jog
  arrived as a single-frame jump. All five are fixed and measured — see DECISIONS.md
  §2026-08-06. **The renderer was not touched**; every one of these lived in delivery or
  in the control geometry, and none was visible from server-side timings or the test suite.
- **Branch `performance-correctness-optimizations`, 30 commits ahead of `master`, NOT
  pushed to GitHub.** James owns the push/merge decision. (`master` itself is 22 commits
  ahead of the stale GitHub default `main`, which is a divergent "Initial commit" — always
  work from `master`/this branch, never `main`.)
- **Delivery has moved to pre-computed templates, and the camera server now serves from
  them.** The camera is orthographic, so the spindle is the only motor that genuinely
  changes image content; everything else is an image-space transform. `loop_sim/library/`
  renders one 360° sweep per scene into the tracked `frame_library/`, and
  `camera_server` checks for a current library at startup, builds one if it is missing or
  stale, then serves every frame by cropping/scaling/blurring a template. Measured
  with CUDA switched off entirely on the lossless-PNG library: **68 ms/frame (14.7 fps)
  through a spindle slew**, where every frame is a fresh template decode, and panning at a
  fixed angle reuses the decoded template and is far cheaper. (The JPEG library it replaced
  slewed at ~46 ms / 21.6 fps — PNG trades some decode time for losslessness; both are well
  above the 10 fps goal.)
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
- **Scene fidelity is the weak half of the project, and as of 2026-08-07 there is no
  trustworthy droplet at all.** The imaging chain is dimensionally correct (a 700.0 µm pin
  measures 703.0 µm in the image), but the bundled benchmark scene contains no droplet and
  no crystal; droplets render opaque for a non-physical reason; and the *generator* that
  was supposed to be the way out silently substitutes a hemisphere for the solved droplet,
  ignoring the volume and contact angle it was given, and places it outside the loop. So
  both routes to a realistic sample are currently blocked. See "Scene fidelity" below —
  this is the highest-value open work, ahead of any further performance tuning.
- **Correctness: DONE + committed.** The float32 "hairy/spikey fiber" GPU artifact is
  fixed — the CUDA intersection quadratic now runs in float64, and the GPU **geometric
  trace is byte-identical to the float64 CPU reference**. With the objective PSF enabled
  (the default since 2026-08-06) the delivered image agrees to **±1 grey level**: the two
  traces always differed by ~3e-8 on ~0.7% of values, and the PSF makes that visible at
  the quantisation boundary. Both properties are tested separately. See DECISIONS.md
  §"float64 GPU intersection" and §2026-08-06 "realism pass".
- **Performance: 10 fps interactive goal MET — and now confirmed on the beamline's TITAN V
  (11.9 fps).** A GPU-resident torch engine (`loop_sim/renderer/engine_torch.py`) runs the
  whole trace on-device; on the 4080 the live server reaches ~25 fps during animated motion
  and ~9.8 fps on the worst-case 10 Hz `/motor` stream, via a flag-gated `torch.compile`
  preview path. `acceptance_voltron.py` measured the deployed path on a real TITAN V at
  **11.9 fps median / 10.1 fps p90 (GO)** — but only with the full software stack (torch 2.6
  + a modern compiler for `torch.compile`); the beamline's default stack falls back to eager
  at 6.3 fps. Settled/offline/`/xray` frames stay bit-exact f64. See RUNBOOK "Deploy on the
  TITAN V" for the exact recipe and DECISIONS.md.
- **Scenes can now be switched at runtime** — `GET /scenes`, `POST /scene`, and a
  tab strip on the control page swap the served sample without a restart or a
  stream drop. A library that is complete but built with older settings is
  **served as-is with a warning, never rebuilt implicitly** — `mitegen_200um`
  was that case until it was rebuilt on 2026-08-07, and an implicit rebuild
  would have cost ~1.9 h on the first tab click. Both shipped libraries now read
  `current`. See DECISIONS.md §2026-08-06 for the lock order and the deadlock
  this work uncovered in the existing `_servable` path.
- **Verify: `pytest tests/` = 149 tests, green** on the local torch env (needs a
  torch+CUDA interpreter; GPU-gated parity tests skip on a CPU-only box).
- **Paused with clear open items** (see below) — nothing half-broken; the engine works.

## How to resume

For a stranger picking this up cold:

1. Build the environment and confirm health: follow **`RUNBOOK.md`** → run `python -m pytest
   tests/ -q` (should be 149 green). "python" is the torch-enabled interpreter — beamline:
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
5. **Know which half of the project you are in.** The *renderer* is well verified — 110 tests,
   GPU matching the CPU reference (byte-identical on the geometric trace, ±1 grey level
   once the objective PSF is applied), and a dimensional check against physics. The
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
  recentre" and DECISIONS.md. **New evidence 2026-08-06:** on the template path, which
  serves a frame in ~35 ms instead of the ~1 s an n_cond=7 live render took, a recentre
  driven over HTTP landed **2.5 px** from target (centroid measured from the server's own
  served frame, target 320,240 → 317.5,240.0; part of that residual is the centroid
  including an asymmetric stem stub). That is consistent with the lag hypothesis but does
  not confirm it: the test drove the endpoint directly, so the browser-side MJPEG
  buffering the hypothesis blames was absent. Reproducing it by clicking in a real browser
  on the template path is the check that would settle it.

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
- **`crystal_harvester` gets the hardware right and the droplet wrong — CORRECTED
  2026-08-07.** This entry previously read "`crystal_harvester` is the trustworthy source
  of scenes", citing a droplet mesh spanning 300 × 300 × 150 µm. That measurement was of a
  **fallback hemisphere, not a solved droplet**, and the distinction was invisible until
  someone generated a scene and rendered it. What still holds: its loop, fiber, stem and
  pin are dimensionally correct (300 µm loop, 20 µm fiber, pin exactly 700 µm), and the
  hand-built bundled scenes remain the outlier for *that* geometry. What does not hold is
  the droplet — see the next entry. Treat `crystal_harvester` as trustworthy for the
  mount and untrustworthy for the solvent until the solver is fixed.
- **The droplet solver silently fails and substitutes a hemisphere.**
  `crystal_harvester/droplet.py::bashforth_adams` integrates the meniscus profile outward
  and looks for where it crosses the loop radius. For a 300 µm loop it never gets there at
  **any** pressure in its own bracket: the profile's widest point peaks at ~78 µm against
  the 150 µm needed, then the surface turns vertical (ψ = π/2) and stops. Measured, on the
  CLI's own defaults:

  | dP | max radius reached | reaches R_loop = 150 µm? |
  |---|---|---|
  | 3.3 (bracket low) | 0.0 µm | no |
  | 33 | 78 µm | no |
  | 267 (bracket high) | 9.6 µm | no |
  | 800 (past the bracket) | diverges (5.9 × 10⁵ mm) | — |

  The first guard therefore fires and it returns `_hemisphere_mesh(R_loop)` with no
  warning. Three consequences, all silent: **`--solvent-volume` is ignored** (0.002 mm³
  requested, 0.00707 delivered — the hemisphere's own volume, 3.5× larger);
  **`--contact-angle` is ignored** (nothing downstream of the failed solve reads it); and
  the drop is a **flat-bottomed dome spanning z = [0, +150] µm**, sitting *on* the loop
  plane instead of straddling it as a lens pinned at the rim. The radius shortfall is not
  a bracket you can widen — the maximum peaks mid-bracket and falls off on both sides,
  which points at a scaling or non-dimensionalisation error in the ODE rather than a
  search-range problem. **Deliberately not fixed** (owner's call, 2026-08-07): it is
  physics work on the generator with a real risk of producing something plausible and
  still wrong, and it wants a validation check — drop volume and rim radius measured back
  off the mesh — as part of it.
- **The generated droplet and crystal are not in the loop.** In
  `scene_files/hampton_300um_realistic.yaml` the loop aperture is centred at
  **x = −248 µm** (the loop path spans x ∈ [−504, 0] µm, the stem x ∈ [0, +700] µm) while
  the droplet and crystal are centred at **x = 0** — the loop/stem junction. So the
  aperture renders empty and the drop hangs off the stem. Every dimension is correct;
  only the position is wrong, and there is no CLI option for drop position, so it is
  generator behaviour rather than a mis-set flag. Separate from the solver failure above,
  though a single missing "place the drop in the loop's frame" step would explain both.
- Reusable harnesses for all of the above live **outside this repo** (they are analysis
  scratch, not a deliverable) at
  `/home/jadoughty/projects/loop_sim_MINE/investigation/2026-07_scene_and_perf_harnesses/`:
  `scene_survey.py` renders a set of scenes across spindle angles into a labelled contact
  sheet, and `scene_dimcheck.py` does the dimensional + NA-sensitivity checks. They import
  `loop_sim`, so run them with the repo on the path — see that directory's `README.md`.

**Deployment reality — the 10 fps target reproduces on the TITAN V (11.9 fps), but only with
the full software stack; the hardware was never the bottleneck, the beamline's default
software stack is.** Measured on a real voltron TITAN V (2026-07-17, `acceptance_voltron.py`,
torch 2.6, devtoolset-7): compiled preview **11.9 fps median / 10.1 fps p90 (GO)**, eager
fallback **6.3 fps**. The three things that decide whether you get 11.9 or 6.3:

- **A — RESOLVED 2026-07-31, but only for callers that asked; the DEFAULT path
  still OOM'd until 2026-08-07.** The VRAM-aware sizing below ran only when a
  caller passed `tile_size=None`, and almost none did: `camera_server`,
  `bench_frame.py`, `acceptance_voltron.py`, `investigation/` and
  `frame_library`'s own scout sweep all took the flat 1,000,000-ray default, so
  a 640×480 frame went through in one pass — 19.8 GB on a 2880-face droplet
  scene. Every scene with a solvent droplet was therefore unrenderable at
  default settings, and a library build for one would have died in the scout
  before reaching the auto-sizing its main loop does use. The default now
  *calculates* the tile from mesh size and free VRAM (160 B per ray per face,
  measured); meshless scenes are unchanged. See DECISIONS.md §2026-08-07 and
  `tests/test_tile_sizing.py`. The original entry follows.

  **VRAM no longer scales with resolution.** The binding
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
- **RESOLVED 2026-08-07 — `mitegen_200um` was rebuilt; both shipped libraries are now
  current.** It is PNG with the objective PSF baked in, at `--supersample 1` (its own
  optically-correct value, see the sampling table below): 360 frames, 1840×2296,
  **16 MB**, 102 min at 17.0 s/frame. Like the hampton rebuild, PNG came out *smaller*
  than the JPEG it replaced (16 vs 28 MB). Nothing ships stale any more.

  **The asymmetry it used to illustrate is still there, though, and still unresolved.**
  Switching to a stale-but-complete library at runtime serves it as-is; launching with
  `--scene <that scene>` still calls `ensure_library` in `__init__` and would rebuild
  before the socket binds. No shipped scene triggers it today, so it is latent rather
  than live — but the next scene built with older settings will hit it, and a fresh
  clone can still be wedged for hours by a launch flag where a tab click would not.
- **CPU/GPU parity is now "±1 grey level", not "byte-identical", once the PSF is on.**
  The two float64 traces always differed by ~3e-8 on ~0.7% of values; that was invisible
  while the image was near-binary and the PSF makes it visible at the quantisation
  boundary. The exact `== 0` tests still run against the geometric trace (`psf=False`);
  a second family bounds the PSF path at ≤1. Exceeding 1 means something structural
  broke, not more rounding. See DECISIONS.md §2026-08-06 "realism pass".
- **The XYZ stage rides on the spindle, so motor axes are not image axes.** Anything that
  moves the sample *the way it looks on screen* must build the displacement in lab space
  from the camera `fast`/`slow` axes and map it into motor space with `Rᵀ` —
  `recenter_target` and `resolve_target`'s pan both do, and `pose_crop` documents the same
  coupling for the crop. Adding `panx` to `tx` is correct only at φ=0; at φ=90 it is a
  no-op or pure defocus. **Corollary:** a screen-space pan writes `tz`, so any "return to
  origin" must zero `tx`, `ty` **and** `tz`. The recenter button zeroed only tx/ty and was
  therefore a no-op at φ=90.
- **The MJPEG stream needs both of its flush mechanisms.** Each part is closed by the
  boundary written after its payload, *and* new content is followed by one prompt resend
  one frame-interval later. They cover different consumers (see DECISIONS.md
  §2026-08-06); dropping either leaves some clients holding the previous frame for a full
  keepalive, which looks exactly like the stage stalling short of target and teleporting.
  Guarded by `test_new_content_is_followed_promptly`. Note that a stream of n frames
  therefore carries n+1 boundaries — count payloads, not boundaries, when writing a test
  client.
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
- **The analysis tree lives outside this repo — out of git, but ON the mirror.** Those are
  two different questions and these docs used to conflate them, asserting in three places
  that `investigation/` was "NOT shipped (excluded from the gateway push)". The intent was
  only ever the first half: keep experiment scratch out of the deliverable's history,
  while still letting the team see what is being worked on. It now lives at
  `/home/jadoughty/projects/loop_sim_MINE/investigation/`, split into
  `2026-06_float64_gpu_parity/` (the original bug-hunt workings) and
  `2026-07_scene_and_perf_harnesses/` (`scene_survey.py`, `scene_dimcheck.py`,
  `soak_server.py`). Not a git repo; mirrored to the gateway, so it will not arrive with a
  `git clone` but will be there beside the repo.
- **The mirror ships `loop_sim_MINE` as three pairs, and that is load-bearing.** The
  analysis tree and the repo's `scratch/` both carry workspace-local paths (133 files in
  the June bug-hunt, one build log in `scratch/`), and the push's path-leak gate is
  **fail-closed and per pair** — as a single pair those tokens would block the deliverable
  itself from shipping. Split, each tree carries its own gate decision: `xtal-loop-sim`
  stays **gated**, the two scratch trees are `nogate`. Keep it that way. Ungating the
  deliverable to make a scratch tree travel would trade the one mechanical check the
  knowledge-transfer protocol has for something nobody clones.
- **An rsync exclude cannot hide a tracked file.** While the harnesses were inside the
  repo they were tracked in git, and the mirror ships `.git/` wholesale — so they reached
  the gateway inside the pack files no matter what the exclude said (and the exclude was
  anchored a level too high to match them anyway). Whenever "this must not travel" is the
  requirement, the content has to be out of the repo; a pattern is not a mechanism.
- **A `scratch/` directory at the repo root is git-ignored but IS mirrored** — renders,
  screenshots, one-off outputs. Nothing in it is a deliverable and it is safe to empty at
  any time; it reaches the gateway as its own ungated pair so the team can see working
  output without it entering the repo's history.

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
- **Why can the Bashforth-Adams profile never reach the loop radius?** Its widest point
  peaks at ~78 µm against the 150 µm a 300 µm loop needs, and falls off on both sides of
  the bracket, so widening the search will not help — the numbers point at a scaling or
  non-dimensionalisation error in the ODE. Until it is answered, every generated droplet
  is the fallback hemisphere and `--solvent-volume`/`--contact-angle` do nothing. The
  first thing to check is the units of the gravity/capillary term in `_ba_rhs` against
  the `dP` the bracket supplies. **Any fix needs a validation check that measures drop
  volume and rim radius back off the generated mesh** — the current failure is invisible
  precisely because nothing does that.
- **Why does `crystal_harvester` place the droplet and crystal at the loop/stem junction
  rather than in the loop aperture?** Loop aperture centred at x = −248 µm, drop centred
  at x = 0, so the aperture renders empty. Every dimension is right, only the position is
  wrong, and there is no CLI flag for it. Possibly the same root cause as the solver
  failure — a missing "place the drop in the loop's frame" step would explain the wrong
  position *and* the drop sitting on the loop plane rather than straddling it. Together
  these two block using generated scenes for fidelity work, which is the whole point of
  generating them. To see it: `python render.py
  scene_files/hampton_300um_realistic.yaml --device cuda` and look at the loop.
- **Should launching on a stale-library scene behave like switching to one?** Runtime
  switching serves a stale-but-complete library as-is; `CameraServer.__init__` still
  rebuilds it. Both behaviours are defensible on their own and they now disagree with
  each other — see the `mitegen_200um` trap above.
- **Push/merge decision for `performance-correctness-optimizations`** — owner: James. Until
  pushed, the branch lives only on this tree + the gateway mirror.
- **Package the TITAN V software stack.** The 11.9 fps result needs torch 2.6 + a modern
  compiler at runtime (RUNBOOK "Deploy on the TITAN V"), assembled by hand in a venv. A
  reproducible env (a pinned recipe, or a launch wrapper that sets `CC`/`CXX`) would make
  deployment turnkey instead of a five-step manual setup.
- **Make the silent compile-fallback loud** (risk B) and **declare torch 2.6** as required —
  a mis-set stack currently misses 10 fps with no signal.
- **The perf-validation tooling lives outside the repo** (`soak_server.py`, profiling
  experiments — see Hazards); `bench_frame.py` and `acceptance_voltron.py` (repo root) ARE
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
  GPU-resident engine), `server/camera_server.py` (AXIS HTTP server + control page +
  runtime scene switching),
  **`library/`** (pre-computed rotation sweeps — `build_library` / `ensure_library`,
  `library_status` / `library_diff` (current/stale/missing, and what differs),
  `frame_for_angle`, `pose_crop`, `zoom_limits`; CLI `python -m loop_sim.library`).
- `frame_library/<scene>/` — **tracked deliverable**, not build output: a rendered 360°
  sweep plus a `manifest.json` per scene. The repo ignores `*.png` and `*.jpg` globally, so
  `.gitignore` carries explicit re-includes for both under this tree. **Currently shipped:
  `hampton_300um`** (360 frames, 1° steps, `--supersample 4`, 5578×2570 each, 28.7 MB PNG) and
  **`mitegen_200um`** (360 frames, `--supersample 1`, 1840×2296, 16 MB PNG, rebuilt
  2026-08-07) — both verified against live renders at 0.00 px, and both now current. The supersample differs because the two cameras sample
  the same NA 0.10 optics very differently; RUNBOOK "Frame libraries" has the rule. Note
  library size in git (see DATA.md "Known gaps") — `--supersample 2` is 4× cheaper than 4
  if that matters for a future scene.
- `crystal_harvester/` — scene *generator* (James's original geometry code: elastica loop
  mechanics, Bashforth-Adams droplets, crystal habits, pin geometry). Builds a complete
  scene from physical parameters — 8 Hampton loop sizes × 3 shapes, 9 MiTeGen models.
  Dimensionally correct for the **mount**; its droplet solver silently falls back to a
  hemisphere and its droplet placement is wrong (see "Scene fidelity"). Still preferable
  to hand-editing YAML for the loop/stem/pin.
- `digitize_fiber.py → add_stem.py → add_droplet.py → add_crystal.py → generate_scene.py`
  — the pipeline that builds a scene from a real loop image (README).
- `scene_files/` — complete example scenes (`hampton_300um.yaml` tube-based;
  `mitegen_200um.yaml` mesh-based). `template.yaml` — camera/material properties.
- `bench_frame.py` — warm-frame benchmark (`--compiled`, `--fp32`). `acceptance_voltron.py`
  — self-contained TITAN V acceptance test (fps + VRAM + compile check → GO/NO-GO +
  `acceptance_report.json`; auto-picks a free GPU). `run_gpu.slurm` — voltron GPU job (no
  `--time`!). `tests/` — 149 tests (the verify command).
- `README.md` — user guide (repo root). `CLAUDE.md` — deep engineering notes (repo root:
  architecture, precision, concurrency, the recentre bug). `docs/` — the handoff docs
  (this file + `RUNBOOK.md`, `DECISIONS.md`, `DATA.md`). `scratch/` — git-ignored,
  git-ignored local scratch (mirrored, not versioned). The experiment/perf harnesses are
  **outside this repo**, at
  `/home/jadoughty/projects/loop_sim_MINE/investigation/`.

## Work log (append-only)

- **2026-08-07 (sync)** — Documentation catch-up on the day's three commits (`13d42b7`,
  `b0b89bf`, `3cceb47`), plus two corrections that matter more than the additions.
  **(1) The droplet diagnosis was written down.** It existed nowhere in the repo:
  `crystal_harvester`'s Bashforth-Adams solver silently returns a hemisphere for ordinary
  inputs, so `--solvent-volume` and `--contact-angle` do nothing and every generated drop
  is the wrong shape in the wrong place. Recorded with its measurements under "Scene
  fidelity", with the reason it was left unrepaired in DECISIONS. This also **corrects**
  the standing claim that `crystal_harvester` is the dimensionally-trustworthy source of
  scenes — that assessment, and the 2026-07-28 audit's approving citation of a
  300 × 300 × 150 µm droplet mesh, were both measuring the fallback. The claim is now
  split: trustworthy for the mount, not for the solvent.
  **(2) The analysis tree moved out of the repo, and the claim that it was excluded was
  false until it did.** `investigation/` was tracked in git and the mirror ships `.git/`
  wholesale, so an rsync exclude could never have hidden it — and the exclude pattern was
  anchored a level too high to match it anyway. It now lives outside the repo, split into
  `2026-06_float64_gpu_parity/` and `2026-07_scene_and_perf_harnesses/`; ~14 doc
  references were repointed at the new absolute location. A git-ignored `scratch/` was
  added at the repo root for renders and one-off outputs. Both it and the analysis tree
  are mirrored to the gateway as their own ungated pairs, which is what keeps the
  deliverable's own leak gate intact.
  Verify re-run on the post-move tree: **149 passed**. **Next:** unchanged — scene
  geometry correctness is still the highest-value open work, and the droplet solver is now
  its first concrete blocker rather than a vague one.

- **2026-08-07 (later)** — `mitegen_200um` rebuilt; nothing ships stale any more.
  Suite **149**. 360 PNG frames with the objective PSF, `--supersample 1` (its own
  optically-correct value), 1840×2296, **16 MB**, 102 min at 17.0 s/frame — smaller
  than the 28 MB JPEG library it replaced, the same way the hampton rebuild went.
  **A grading flaw had to be fixed first, or no rebuild could have cleared the
  warning:** the server graded every scene's `supersample` against one global
  default, but supersample is per-scene by design (hampton 4, mitegen 1 — it
  follows each camera's sampling against the objective's Nyquist limit). Whichever
  scene did not match the default was permanently `stale`, and unclearably so,
  since the value being called stale is the correct one for that scene; satisfying
  it by rebuilding mitegen at 4 would have been optically wrong, ~29 h and ~430 MB.
  It is now graded only when explicitly passed. Everything else stays graded.
  Both shipped libraries now read `current` and switching to mitegen raises no
  warning. **Left alone deliberately:** the `crystal_harvester` droplet defects
  found earlier the same day (see the previous entry) — not diagnosed further, not
  patched.

- **2026-08-07** — First `crystal_harvester` scene generated, and it exposed that
  droplet scenes could not be rendered at all with default settings. Suite **147**.
  Generated `scene_files/hampton_300um_realistic.yaml` (the scene the fidelity
  block has recommended since 2026-07-28): 300 µm teardrop loop, 20 µm fiber, a
  real 300 × 300 × 150 µm droplet mesh and a hexagonal crystal — everything the
  bundled `hampton_300um` lacks. **Two findings, both from actually rendering it.**
  (1) **`render_torch`'s default tile OOM'd on any mesh scene**: the flat
  1,000,000-ray default put a 640×480 frame through in one pass, which at 2880
  faces is 19.8 GB. The 2026-07-31 VRAM work only ran for callers passing
  `tile_size=None`, and essentially nobody does — including `frame_library`'s own
  scout sweep, so a library build for this scene would have died before reaching
  the auto-sizing its main loop uses. The default now calculates the tile from
  mesh size and free VRAM (160 B/ray/face, measured across two scenes and four
  tile sizes); meshless scenes are bit-for-bit unchanged, so no benchmark moves.
  Probing was rejected as the default for three separate reasons — see
  DECISIONS.md §2026-08-07. Renders in **3.2 s at 3.6 GB** where it used to OOM.
  (2) **The generated scene's droplet and crystal are in the wrong place**: the
  loop aperture is centred at x = −248 µm and the drop at x = 0, the loop/stem
  junction, so the aperture renders empty and the drop hangs off the stem. Sizes
  are all correct; only the placement is wrong. Not yet diagnosed — there is no
  CLI option for drop position, so it is generator behaviour, not a mis-set flag.
  The render also reproduces the known opaque-droplet bug. **Next:** that
  placement question, and it sits directly on the critical path for the whole
  scene-fidelity effort.

- **2026-08-06 (latest)** — Runtime scene switching, and a live deadlock found on
  the way. Suite **139** (was 110). The server held one scene for the life of the
  process; it now has `GET /scenes`, `GET /scene`, `POST /scene?path=&build=` and
  a **tab strip** on the control page, and swaps scenes without restarting or
  dropping the stream. Design: `_build_bundle` does everything fallible off-lock
  and writes nothing to `self`; `_install_bundle` writes `self` and cannot raise,
  so a failed switch is structurally a no-op with no rollback path.
  **Three defects fixed in passing, all pre-existing.** (1) A genuine two-thread
  **deadlock**: `_servable` reads `_templates` and is called from inside
  `_gonio_lock`, so scene-guarding `_templates` naively would invert against
  `_render_now` and hang the whole server on one `/motor` during one render.
  `_servable` now acquires nothing and `_set_pose_instant` hoists `_scene_lock`
  outside `_gonio_lock`; `tests/test_server_lock_order.py` checks the order
  statically (verified by reintroducing the bug into a copy — it names it).
  (2) `_snapshot_gonio` read the pose under `_gonio_lock` and `_scene.geometry`
  *after* releasing it, and `_command_recenter` had the same split — both would
  pair one scene's pose with another's axes. (3) `--engine torch --templates on`
  built a `TorchScene` nothing ever calls (the guard tested `engine == "auto"`).
  **`mitegen_200um` is served stale, deliberately and permanently.** Its manifest
  predates the `format` and `psf` build keys, so `is_current` is false — but its
  360 frames are fine and `ensure_library` would have started a ~1.9 h rebuild on
  the first tab click. New `library_status` splits the answer into
  `current`/`stale`/`missing`; the switch path never calls `ensure_library`, and
  `library_diff` names what differs ("stored as jpeg, not png; built without the
  objective PSF; 1× supersample, so zoom is capped at 1×"). Verified that nothing
  in the serving path reads `format`/`psf`, so a legacy manifest cannot KeyError.
  Builds are explicit only, go to a separate untracked `frame_library_preview/`
  root (building in place would overwrite frames the live `TemplateSource` caches
  by filename), and are **refused without CUDA** — `--allow-cpu` on the library
  CLI is the sole escape hatch. Driven end to end against the real server: the
  picture changes, a switch mid-slew cancels the animation without writing
  (pose lands at home, not the 33° it had reached), the stream keeps flowing, and
  the stage still responds afterwards. **Next:** scene geometry correctness —
  the fidelity block under Hazards is still the highest-value open work.

- **2026-08-06 (later still)** — Motion realism, and a race it uncovered. **COMMITTED**
  (`ada9e41`, `471ce20`); suite **110**. Two commits: (1) a **pre-existing bug** —
  `_run_animation` checked the animation generation and wrote the goniometer in separate
  critical sections, and the settle block never checked at all, so a preempted move could
  stamp its pose after a newer command had landed; both are now single critical sections,
  and the payoff is that a cancelled animation provably touches nothing, which is what
  will let scene switching skip quiescing the animator. (2) **Trapezoidal motion** —
  `velocity_step` ramps to full speed over a fixed 0.15 s, holds, and brakes to arrive at
  rest, with speed carried across a preempt so a burst of jog clicks is one continuous
  motion rather than N accelerate-brake cycles. Measured: 149 → 170 → 143 °/s across a
  180° move; pose never still longer than 38 ms through a 20-click burst. Stage rates
  **halved** (4 s screen crossing, 180 °/s, zoom 2/s) — the old speeds were about twice
  life-size, so what needed the dial at 0.5× is now 1.0×. The 0.25 s duration floor was
  deleted as redundant once a real ramp exists. **Next: runtime scene switching** — a
  plan exists (bundle/install split, `_scene_lock`, `/scenes` + `/scene` endpoints, tab
  UI, preview libraries in an untracked root, no-GPU refusal); see DECISIONS.md
  §2026-08-06 for the design constraints already established.

- **2026-08-06 (later)** — Realism pass: the renderer gained the optics it was missing,
  and templates became lossless. **UNCOMMITTED, Jacob commits.** Driven by a simple
  observation — the image looked blocky at zoom 4 — which turned out not to be aliasing:
  at zoom 4 the 4× supersample budget is exactly exhausted (1.00 template px per output
  px), and underneath, **97.7% of a frame was pure black or white** with edges resolving
  in ~1 template pixel. NA 0.10 at 550 nm cannot form an edge sharper than ~1.8. The
  renderer was about twice as sharp as the optics it claims to model.
  **(1) Objective PSF** (`loop_sim/renderer/optics.py`): a Gaussian approximation to the
  Airy PSF, σ = 0.21 λ/NA, applied by both renderers through one shared numpy
  implementation — two implementations could not have stayed byte-equal. σ is derived
  from `eff_px`, so it is fixed in object space and only becomes visible under
  magnification (0.156 px on hampton's camera pixel, 0.624 px in its template). An edge
  went from a pure 255→0 step to a 255→254→209→46→1 ramp.
  **(2) A pre-existing float divergence surfaced and the parity claim was restated
  honestly:** the numpy and torch traces were never bit-identical (~3e-8 on ~0.7% of
  values); near-binary images hid it, and the PSF exposes it at the rounding boundary.
  CPU/GPU now agree to **±1 grey level** with the PSF on, and the exact `== 0` tests were
  *kept* — they now run against the geometric trace (`psf=False`), with a second family
  bounding the PSF path.
  **(3) Templates are lossless PNG.** A real AXIS camera compresses once; storing JPEG
  templates and re-encoding on the wire compressed twice. PNG is also *smaller* here
  (28.7 vs 84.9 MB for the hampton sweep) because the frames are overwhelmingly
  flat. `format` and `psf` became build parameters, `build_library` now deletes frames
  whose extension no longer matches (they would otherwise strand in git), and
  `.gitignore` gained the `*.png` re-include without which the new deliverable would
  never have been committed at all.
  Auto-rebuild on stale was deliberately left as-is — the team must never have to run a
  build step. README gained a table of contents and a physics section (what is modelled,
  and explicitly what is not); RUNBOOK gained an "Every lever" table of every CLI flag,
  scene key and environment requirement, marking which ones force a library rebuild.
  Suite **102 passed**, up from 86. **hampton_300um rebuilt; `mitegen_200um` deliberately
  not** — it will silently auto-rebuild (~1.9 h) on first use until someone does it.
  **Next:** scene geometry correctness.

- **2026-08-06** — The camera was driven interactively for the first time, and the
  delivery path turned out to be the weak part, not the renderer. **UNCOMMITTED — 8 files,
  Jacob commits.** Nothing in `loop_sim/renderer/` was touched. Fixed, each with a
  measurement (DECISIONS.md §2026-08-06): **(1)** `--fps-limit` default 5.0 → 30.0 — the
  MJPEG clamp, not rendering, was setting the frame rate, and every inter-frame gap was
  exactly 200 ms until the flag changed (5.12 → 28.1 fps, same server, same scene); the
  24 fps in the 2026-07-31 entry was a render-cost figure the shipped default could not
  deliver. **(2)** The ~1 s freeze at the end of every move: MJPEG parts are now closed as
  they are written *and* new content is followed by one prompt resend — both are needed,
  for different consumers, and this regressed twice before the test existed
  (1002 ms → 35 ms worst gap for the strictest consumer). **(3)** The translate pad was
  moving the sample along motor axes: it now resolves through `Rᵀ` like `recenter_target`
  always did — it had been sending the same `ty = −0.888 mm` at every φ, a no-op at 90°
  and backwards at 180°; verified by phase correlation, 0 wrong out of 14 across seven
  angles. **(4)** Out-of-range zoom/x/y/z was accepted and silently ignored; the commanded
  pose is now clamped to what the library can serve via a new `servable_pose()` (which
  inverts `pose_crop`'s own box rather than reimplementing the clamp), so the readout and
  the target boxes match the picture — re-requesting the reported pose reproduces the
  image byte-for-byte. **(5)** A 15° jog was a one-frame jump; non-zero moves now have a
  0.25 s duration floor so bursts chain into continuous rotation (18 stalls → 1 over a
  20-click burst). Control page also gained a typable φ box beside relabelled
  `[−15°] [φ] [+15°]` jog buttons, and the goniometer-target boxes now follow the stage
  (a box you type in holds its value until GO). Suite **86 passed**, up from 80: new
  guards for pan-under-rotation, the pan/recenter sign convention, the duration floor,
  `servable_pose` exactness, and the stream flush. **Next:** scene geometry correctness —
  the fidelity block under Hazards is still untouched and is the highest-value work.

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
