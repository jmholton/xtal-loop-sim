---
project: loop-sim (xtal-loop-sim) — bright-field microscope + X-ray simulator for protein crystals in cryo-loops
status: active — camera served from pre-computed templates (no GPU at runtime) and usable interactively; renders go out through a measured camera model on the real 704x480 raster; all three frame libraries current; the NA half of the camera-calibration fork is the one open front
last_verified: 2026-08-11        # `pytest tests/` = 213 passed in 119 s on this tree (branch performance-correctness-optimizations, RTX 4080 SUPER)
verify: python -m pytest tests/ -q        # 213 tests; "python" = the torch-enabled project interpreter (see docs/RUNBOOK.md "Environment")
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

## Current state (2026-08-11)

- **The output is now judged against PHOTOGRAPHS, and that changed what the top
  problem was.** `real_images/` (44 tracked frames, `MANIFEST.tsv`) is the first
  reference set this project has had. The gap nobody had named turned out to dominate:
  the render was **84.6% pure white / 14.4% pure black / 1.1% anything else**, where real
  frames carry **19–27% intermediate tone**. A binary silhouette against a continuous-tone
  photograph. `loop_sim/renderer/field.py` now maps transmittance through a measured
  camera model — illumination field, black floor, tone response — and the served frame
  went to **100% intermediate tone, range 32–181, mean 144.5** against C07's ~150. It
  runs at **serve time only**, so no template and no library changed. See DECISIONS.md
  §2026-08-10.
- **Frames go out on the real camera's 704×480 raster, and 640 was never wrong.** The
  worry was a silent 10% horizontal scale error. The BL831 pixels are **1.110
  non-square** and 704/640 = 1.100 cancels it: 640 × 7.4 µm and the real 704 × 6.7324 µm
  cover the same field to **0.08% / 0.98%**. Rendering 704 wide at 7.4 µm would
  over-cover by +9.9%. The scene stays square-pixel; `field.to_sensor` resamples at
  delivery, because dcss stores a µm-per-pixel constant and a stand-in emitting 640
  columns reads 10% wide. `--sensor-pitch off` restores square pixels.
- **The pin carries its specular glint.** The tracer models the pin as purely opaque, so
  it rendered as a flat silhouette; real ones show a bright broken streak along the
  shank, and it was the largest remaining structural difference. Measured on A01 and E02
  first — which corrected the plan's spec twice (its "1.25× background peak" was a
  three-pixel frame maximum, and its grain figure was the pin body, not the ridge, which
  is ~10× grainier). 3.4 ms/frame, camera space, no rebuild. `--pin-streak off`.
- **The glint survives being DRIVEN, which is how its remaining defects were
  found.** Four came out of an operator turning the spindle and the zoom rather
  than out of any test or still frame: it sloped +/-6.4 degrees with phi (the
  chisel tip dragging the fitted axis), its grain read as parallax in motion
  (anchored to a centroid that drifts as the pin leaves frame), it vanished
  above ~1.5x zoom (the in-frame piece becomes wider than long, so the moments
  called the shank vertical), and it stayed razor-sharp on a defocused pin.
  All four are fixed and measured; DECISIONS.md §2026-08-11 has the numbers and
  the two dead ends that did NOT separate the cases.
- **Rotating frame rate is decode-bound, and the levers are known.** A rotating
  frame is 97.2 ms: **template decode 73.2 ms (75%)**, camera stage 15.8 ms,
  the rest 8.2 ms. The socket delivers 10-12 fps and a browser shows about half
  that. Two ways out, neither taken: a prefetch decode pool (~30 fps, no
  rebuild — PIL releases the GIL during PNG decode and a slew's direction is
  predictable) or rebuilding at `--supersample 2` (~24 fps, 47 min, zoom
  ceiling 4x -> 2x). float32 in the camera stage was measured and buys nothing.
- **The pin's glint is inferred from the SILHOUETTE, and that is an accepted
  limitation, not an oversight.** Because the geometry comes from what is in
  frame, the glint disappears when the pin's side leaves the frame, and on a
  tip-only view the ridge follows the tip's curve instead of the shank. Judged
  a small incorrectness and accepted (owner, 2026-08-10). The fix is not a
  better inference: hand the stage the pin's axis and radius from the SCENE,
  which the server already knows, and ~150 lines of heuristic in
  `renderer/field.py` delete. See `_pin_axis`'s docstring.
- **`render_sha` closes the last silent-staleness hole.** A renderer edit used to leave
  every manifest reading `current` while the frames on disk had been traced by code that
  no longer existed. Now hashed into the manifest and `_BUILD_KEYS`. `field.py` is
  deliberately excluded — it is serve-time and never enters a template.
- **All three frame libraries are current again.** `hampton_300um_realistic` was
  rebuilt overnight 2026-08-11 for slice 3 -- 360 frames, 8.06 h at 80.6 s/frame,
  verified against a live f64 render at **0,0 px, mean |diff| 0.00056, 99.1%
  identical**. It was held pending the NA question and built anyway at the
  owner's call, so **a switch to NA 0.28 would still invalidate it** (RUNBOOK
  "Frame libraries" has the command and the two WSL2-critical flags).
  **Launching a server on that scene still requires `--supersample 1`** --
  verified 2026-08-11 with `build_library` patched to raise: a bare launch
  grades the S=1 library stale against `build_params`' S=4 default and would
  rebuild it. The tab strip and `--templates off` are safe.
- **The NA fork now has its strongest evidence, and it came from fixing the drop.** See
  "Open questions" below — the drop is fine at every NA; it is the immersed *crystal*
  that goes 3× too dark at NA 0.10 and recovers halfway at 0.28.

## Earlier state (2026-08-07)

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
- **The black-droplet mystery is SOLVED and the generator now produces a trustworthy,
  validated droplet (2026-08-07, later).** Two scene-side mechanisms, no renderer bug:
  the solvent's `color` was secretly a strong absorber (color IS an absorption spectrum
  in this renderer — the ~0.26 brightness ceiling in the old NA sweep), and the silently
  substituted hemisphere was the worst possible lens shape for the NA gate. The solver
  was replaced with the closed-form spherical-cap lens `add_droplet.py` already had, the
  droplet now sits **in the loop aperture** straddling the loop plane, the crystal is
  centred in it, and every generated scene is measured back by
  `crystal_harvester/validate.py` (volume, rim-on-fiber, watertightness — run by the CLI,
  guarded by `tests/test_scene_geometry.py`). The regenerated
  `hampton_300um_realistic.yaml` renders a near-background-bright drop with a dark rim
  (core 0.67 vs background 0.93 at NA 0.10, was 0.04). Full detail in DECISIONS.md
  §2026-08-07 (later). What remains open is the **camera calibration fork** (below).
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
- **Verify: `pytest tests/` = 213 tests, green** on the local torch env (needs a
  torch+CUDA interpreter; GPU-gated parity tests skip on a CPU-only box).
- **Paused with clear open items** (see below) — nothing half-broken; the engine works.

## How to resume

For a stranger picking this up cold:

1. Build the environment and confirm health: follow **`RUNBOOK.md`** → run `python -m pytest
   tests/ -q` (should be 213 green). "python" is the torch-enabled interpreter — beamline:
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
5. **Know which half of the project you are in.** The *renderer* is well verified — GPU
   matching the CPU reference (byte-identical on the geometric trace, ±1 grey level once
   the objective PSF is applied), and a dimensional check against physics. *Generated
   scenes* are now validated mesh-back too (2026-08-07 later). The bundled
   `hampton_300um.yaml` remains deliberately bare (perf baseline);
   `hampton_300um_realistic.yaml` is the fidelity scene. The open fidelity question is
   the camera calibration, not the geometry.
6. **For the AXIS-camera use case, look at `frame_library/` before touching the renderer.**
   Delivery has shifted to pre-computed rotation sweeps replayed at request time, which
   sidesteps the frame-rate problem rather than fighting it (RUNBOOK "Frame libraries").

The highest-value open engineering items, in rough priority:
- **Settle the objective NA** — the last question gating fidelity, and the thing the
  held rebuild is waiting on. The pixel half of the old "three cameras" puzzle is
  settled (2026-08-10): there is one camera with 1.110 non-square pixels at two zoom
  stops, and the Hampton scenes already render its mid stop correctly on square pixels.
  NA is what remains, and it now has a measurement rather than an argument: at NA 0.10 a
  correct half-maximum drop leaves the immersed crystal at **0.218 × background against
  0.696 in the reference photograph**, recovering to 0.455 at NA 0.28. Two independent
  hints point at 0.28 — the reference frame is hi mag, whose own calibration is 0.28,
  and the drop's rim deflection lands at sin 0.273. The cheapest decider is still **one
  photograph of a real 300 µm loop carrying a drop at a known zoom stop** (Jacob
  captures). Cost a switch first: the supersample ceiling moves and every library
  rebuilds.
- **Decide how to spend the decode budget** — 75% of a rotating frame is PNG
  decode. A prefetch pool would roughly triple the rate for no rebuild and no
  loss of zoom range; `--supersample 2` is simpler but costs both. Neither is
  urgent: 10-12 fps at the socket already clears the 10 fps goal.
- **Give `TSurfaceMesh` the AABB cull that `TTube` has** — a speed optimisation for mesh
  scenes (they render, but slowly; the fidelity scene is ~5.5k faces now).
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

**Scene fidelity — the renderer is validated, and generated scenes now are too.** Until
2026-07-28 all verification was self-consistency (GPU render vs CPU render of the same
scene), which cannot detect a wrong *scene*; since 2026-08-07 (later) every generated
scene is measured back against what was requested (`crystal_harvester/validate.py`).
What is measured:

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
- **RESOLVED 2026-08-07 (later) — the opaque droplet was two scene-side mechanisms**, and
  the trap that caused half of it is still live for scene authors: **material `color` is
  an absorption spectrum, not a display tint** — `mu_per_ch = mu_optical + 30·(1−color)`
  per mm (`microscope.py`), so the old solvent color `[0.2,0.4,0.8]` absorbed at
  (24,18,6)/mm despite `mu_optical: 0.00`. That was the ~0.26 ceiling the NA sweep
  saturated at (and why it could not reach ~1.0 at NA 0.90); the rest was the fallback
  hemisphere's flat-bottomed shape refracting ~92% of the drop's area past the NA gate.
  Any colored material is a strongly absorbing material — keep water-like solvents
  near-white. `MAX_DEPTH` bounce exhaustion was **exonerated**: an exhausted ray keeps
  its partial product and skips the NA test entirely, so it errs *bright* and cannot
  blacken anything. TIR-as-absorption is what draws the dark rim and is correct. See
  DECISIONS.md §2026-08-07 (later) for all measurements.
- **CORRECTED 2026-08-10 — there are not three cameras, and the PIXEL half is settled.**
  `template.yaml` (0.82 µm, NA 0.28/0.17), the Hampton scenes (7.4 µm, NA 0.10/0.07) and
  `mitegen_200um` (1.0 µm) look like three calibrations, but the BL831 sample camera has
  **1.110 non-square pixels** at both usable zoom stops. The Hampton scenes' square
  640 × 7.4 µm IS that camera's mid stop on square pixels — same field of view to 0.08%
  horizontally and 0.98% vertically. `template.yaml` is the hi stop's HORIZONTAL pitch
  used as a square pixel, hence 9.91% short vertically; nothing uses the file. What
  remains open is **NA alone**, and it now has measured evidence (see Open questions).
  The original entry follows.

  *(superseded)* Three different cameras are in circulation. `template.yaml` — which
  DATA.md calls the authoritative calibration — specifies 0.82 µm pixels and NA
  0.28/0.17, but **no shipped scene uses it**: the Hampton scenes use 7.4 µm and NA
  0.10/0.07, `mitegen_200um` uses 1.0 µm. Since NA is the knob driving the
  opaque-droplet result, settling which of these matches the real beamline camera is a
  prerequisite for judging fidelity.
- **The fiber is beaded at the default sampling.** Tubes become `n_samples - 1` capsules;
  at the default `n_samples=50` a 300 µm loop yields 19.3 µm segments against a 20.0 µm
  fiber — capsules as long as they are wide. Raise `n_samples` for fidelity renders.
- **`crystal_harvester` is now trustworthy for the mount AND the solvent — and it
  proves it on every run.** The mount was always dimensionally correct (300 µm loop,
  20 µm fiber, pin exactly 700 µm). The solvent is now a closed-form spherical-cap lens
  pinned in the loop aperture, and the CLI measures every emitted scene back
  (`crystal_harvester/validate.py`): divergence-theorem volume within 2% of
  `--solvent-volume`, rim on the fiber, watertight, plane-straddling, crystal centred
  and priority-ordered. A scene that fails ships nothing — there is no fallback shape
  in the generator any more. Note `--contact-angle` is deprecated and ignored: with the
  rim pinned at the loop, contact angle is an output of volume + rim radius.
- **RESOLVED 2026-08-07 (later) — the Bashforth-Adams solver was deleted, not fixed.**
  Its defect was found (the azimuthal-curvature term used the wrong angle convention,
  `sin(ψ)/r` where its ψ-from-vertical frame needs `cos(ψ)/r`, which is exactly why the
  profile peaked mid-bracket and could never reach the loop radius) — but the deeper
  answer is that the ODE solves a problem with a closed form: at Bo ≈ 0.003 the pinned
  zero-gravity surface is *exactly* a spherical cap, and `add_droplet.py` already
  implemented the two-cap biconvex lens correctly. One shared implementation now lives in
  `crystal_harvester/droplet.py`, used by both the generator and `add_droplet.py`. The
  validation check the owner required (volume + rim measured back off the mesh) shipped
  with it and runs on every generated scene. DECISIONS.md §2026-08-07 (later) has the
  full reasoning; the superseded diagnosis with its measurements is preserved below it.
- **RESOLVED 2026-08-07 (later) — droplet and crystal are now in the loop.** The
  generator was writing solver vertices into the YAML verbatim (origin-centred, at the
  loop/stem junction). The droplet is now built in the loop's canonical frame, pinned on
  the inner fiber edge all the way around the aperture (measured 6.8–10 µm from the
  10 µm-radius fiber axis), straddles the loop plane, and rotates with the loop for any
  `loop_axis`. The crystal sits at the droplet's volume centroid. Note the crystal
  (±50 µm) is thicker than a 2 nL drop (±21 µm) and pokes out — a real mount does this
  too, but the renderer shows hard crystal/air interfaces with no wetting film; the
  validator reports it as a warning.
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
  **Superseded 2026-08-10:** three libraries ship now, and
  `hampton_300um_realistic` is `missing` rather than stale — slice 3 changed its
  scene and its rebuild is deliberately held (see Current state).

  **The asymmetry it used to illustrate is still there — and as of 2026-08-08 it is
  LIVE, not latent.** Switching to a stale-but-complete library at runtime serves it
  as-is; launching with `--scene <that scene>` still calls `ensure_library` in
  `__init__` and rebuilds before the socket binds. `hampton_300um_realistic` is the
  scene that triggers it: its per-scene-correct `--supersample 1` library grades stale
  against the launch path's defaults, so a bare
  `camera_server --scene scene_files/hampton_300um_realistic.yaml` **deletes the
  manifest and starts a days-long supersample-4 mesh rebuild before serving anything**
  (it did exactly this on 2026-08-08; the manifest had to be reconstructed — see the
  work log). **Workaround until fixed: always pass `--supersample 1` when launching
  the server on this scene.** The tab-switch path is unaffected and serves it fine.
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

- **Which camera calibration is real? — the PIXEL half is settled, the NA half is now
  the single open front, and it has hard evidence.**

  *Settled 2026-08-10 (pixel scale).* The three "cameras" are not three cameras. The
  BL831 sample camera has **1.110 non-square pixels** at both usable zoom stops
  (6.7324 × 7.4729 µm mid, 0.8233 × 0.9139 hi — `sample_camera_constant` in
  `wash_pin/claude/BL-831.dat`, halved for 704×480). The Hampton scenes' square
  640 × 7.4 µm is that camera's **mid stop rendered on square pixels**, matching its
  field of view to 0.08% horizontally and 0.98% vertically. `template.yaml` is the **hi
  stop's horizontal pitch used as a square pixel**, which makes it 9.91% short
  vertically — the one real 10% error in the set, in a file nothing uses. Nothing here
  needs changing; see DECISIONS §2026-08-10.

  *Open (NA).* NA sets how much of a refracting body's light clears the objective, and
  the drop-volume fix made this measurable for the first time. **The drop is not the
  problem at any NA** (0.955–0.989 × background throughout). The **crystal** is: once a
  correct half-maximum drop immerses it, it reads 0.218 × background at NA 0.10 against
  **0.696 in the reference photograph**, recovering only to 0.455 at NA 0.28. Two things
  point the same way — the reference frame `D01` is **hi mag**, whose own calibration is
  NA 0.28; and the drop's rim-ray deflection lands at **sin 0.273**, which is NA 0.28
  almost exactly. Neither number is proof, and the crystal's absorption was deliberately
  **not** tuned to hide the gap.

  The cheapest remaining ground truth is unchanged: **one photograph of a real 300 µm
  loop carrying a drop, at a known zoom stop**, on the beamline camera (Jacob captures).
  Cost any switch to NA 0.28 first — the supersample ceiling moves and every frame
  library rebuilds. **This is why `hampton_300um_realistic`'s rebuild is being held**:
  9.5 h that the switch would invalidate.
- **Is the bundled `hampton_300um` loop mislabelled, or digitized at another size?** Its
  waypoints span 69 × 200 µm, not ~300 µm. Worth comparing against the physical part before
  assuming the geometry is wrong rather than the name.
- ~~Why do droplets go opaque?~~ **ANSWERED 2026-08-07 (later)**: colour-as-absorption
  was the 0.26 ceiling, the fallback hemisphere's shape was the NA-gate floor, and
  `MAX_DEPTH` was exonerated (it errs bright). See DECISIONS.md.
- **Should the dimensional check become a test?** The 700 µm → 703.0 µm pin measurement is
  architecture-independent and would close the "no golden reference / gates are
  architecture-blind" gap DATA.md records. It needs no committed image, only the assertion.
- **Frame-library coverage.** The sweep covers rotation; `zoom` and `tz` are not free the
  way lateral translation is and would need their own sweeps or a live render. Decide
  whether the AXIS consumer needs them before treating the library as complete.
- ~~Why can the Bashforth-Adams profile never reach the loop radius?~~ **ANSWERED
  2026-08-07 (later)**: a curvature-term convention error (`sin(ψ)/r` for `cos(ψ)/r`).
  The solver was deleted for the closed-form spherical-cap lens, with the mesh-back
  validation the fix was required to carry. See DECISIONS.md.
- ~~Why does `crystal_harvester` place the droplet at the loop/stem junction?~~
  **ANSWERED 2026-08-07 (later)**: it wrote solver vertices into the YAML verbatim with
  no translation. The droplet is now pinned in the aperture and the crystal centred in
  the droplet; `python render.py scene_files/hampton_300um_realistic.yaml --device cuda`
  now shows exactly that.
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
  GPU-resident engine, `optics.py` objective PSF, **`field.py`** the camera model —
  sensor raster, illumination field, black floor, tone, the pin's specular streak;
  numpy-only, applied at SERVE time and inside neither tracer, which is what keeps it
  off the templates), `server/camera_server.py` (AXIS HTTP server + control page +
  runtime scene switching; `encode_frame` is the one place a served frame becomes
  bytes),
  **`library/`** (pre-computed rotation sweeps — `build_library` / `ensure_library`,
  `library_status` / `library_diff` (current/stale/missing, and what differs),
  `frame_for_angle`, `pose_crop`, `zoom_limits`; CLI `python -m loop_sim.library`).
- `frame_library/<scene>/` — **tracked deliverable**, not build output: a rendered 360°
  sweep plus a `manifest.json` per scene. The repo ignores `*.png` and `*.jpg` globally, so
  `.gitignore` carries explicit re-includes for both under this tree. **Currently shipped:
  `hampton_300um`** (360 frames, 1° steps, `--supersample 4`, 5578×2570 each, 28.7 MB PNG) and
  **`mitegen_200um`** (360 frames, `--supersample 1`, 1840×2296, 16 MB PNG, rebuilt
  2026-08-07) and **`hampton_300um_realistic`** (360 frames, `--supersample 1`,
  1396×644, 2.9 MB PNG) — all verified against live renders at 0.00 px. The first two
  are current; **the third is `missing` since 2026-08-10 and awaits a rebuild** (see
  Current state, and RUNBOOK "Frame libraries" for the command and its two load-bearing
  flags). The supersample differs because the two cameras sample
  the same NA 0.10 optics very differently; RUNBOOK "Frame libraries" has the rule. Note
  library size in git (see DATA.md "Known gaps") — `--supersample 2` is 4× cheaper than 4
  if that matters for a future scene.
- `crystal_harvester/` — scene *generator* (James's original geometry code: elastica loop
  mechanics, crystal habits, pin geometry; the droplet is a closed-form spherical-cap
  lens in `droplet.py`, shared with `add_droplet.py`). Builds a complete scene from
  physical parameters — 8 Hampton loop sizes × 3 shapes, 9 MiTeGen models — and
  **`validate.py` measures every emitted scene back** (volume, rim-on-fiber,
  watertightness, crystal placement; the CLI runs it on every build). Dimensionally
  correct for the mount *and* the solvent since 2026-08-07 (later).
- `digitize_fiber.py → add_stem.py → add_droplet.py → add_crystal.py → generate_scene.py`
  — the pipeline that builds a scene from a real loop image (README).
- `scene_files/` — complete example scenes (`hampton_300um.yaml` tube-based and the
  frozen performance baseline — never edit it; `hampton_300um_realistic.yaml` the
  fidelity scene, generated; `mitegen_200um.yaml` mesh-based). `template.yaml` — camera
  and material properties, and the hi zoom stop's horizontal pitch used as a square
  pixel, so 9.91% short vertically; nothing reads it.
- `real_images/` — **44 tracked BL831 sample-camera frames, the realism reference.** Not
  inputs: this is the ground truth the output is judged against. `MANIFEST.tsv` gives
  magnification, subject, why each was kept and its source path; `README.md` carries the
  three limits that make it a FALSIFICATION set rather than a fitting set (no single
  real background, two mutually exclusive calibrations inside the directory, and the
  A/C sets' scale is inferred and not self-consistent).
- `bench_frame.py` — warm-frame benchmark (`--compiled`, `--fp32`). `acceptance_voltron.py`
  — self-contained TITAN V acceptance test (fps + VRAM + compile check → GO/NO-GO +
  `acceptance_report.json`; auto-picks a free GPU). `run_gpu.slurm` — voltron GPU job (no
  `--time`!). `tests/` — 213 tests (the verify command).
- `README.md` — user guide (repo root). `CLAUDE.md` — deep engineering notes (repo root:
  architecture, precision, concurrency, the recentre bug). `docs/` — the handoff docs
  (this file + `RUNBOOK.md`, `DECISIONS.md`, `DATA.md`). `scratch/` — git-ignored,
  git-ignored local scratch (mirrored, not versioned). The experiment/perf harnesses are
  **outside this repo**, at
  `/home/jadoughty/projects/loop_sim_MINE/investigation/`.

## Work log (append-only)

- **2026-08-11 — the camera model met an operator, and the droplet library was
  rebuilt.** Suite **213** (was 206). Four commits.
  `1e35cf9` fixed four things found by DRIVING the viewer rather than by any
  test: the glint sloped ±6.4° with φ (the 45° chisel dragging the fitted
  axis — now fitted from the shank's two long sides, worst tilt 0.42°), its
  grain read as parallax in motion (now scintillates per pose), it vanished
  above ~1.5× zoom (the in-frame piece is wider than long, so the moments
  called the shank vertical — both orientations are now fitted and the
  consistent one wins), and the background was invisible (the residual was
  re-measured at 3.4–3.9% with a proper mask, not 2.6–2.9%, and its energy is
  multi-scale — now six octaves of fBm).
  `bb0d36d` rebuilt `hampton_300um_realistic` for slice 3: 8.06 h at
  80.6 s/frame, verified against a live f64 render at **0,0 px / mean |diff|
  0.00056 / 99.1% identical**. All three libraries are current again.
  `f19f5d9` corrected a doc claim that had become wrong in the dangerous
  direction, and `09aaafb` made the glint defocus with the sample.
  **Two negatives worth not re-deriving:** float32 in the camera stage is not
  faster (index-bound, not bandwidth-bound), and neither aspect nor
  bar-likeness can separate a zoomed-in pin from a mount whose sides are off
  frame. **Next:** the NA fork, unchanged and now the only open front.

- **2026-08-10 — the renders were compared against photographs for the first time,
  and four of the six named gaps closed.** Suite **206** (was 167). Seven commits;
  full reasoning and every measurement in DECISIONS.md §2026-08-10.
  `4b2a2a4` camera emulation (tone: 1.5% → 100% intermediate, range 0–255 → 32–181,
  mean 239.6 → 144.5 against C07's ~150), `8c29003` generator hygiene (six falsy-zero
  `or` bugs; `--pin-bevel 0` was a documented lie), `a7053a1` `real_images/` tracked,
  `ab0c90c` the 704×480 sensor raster, `468665c` the pin's specular streak,
  `557c418` `render_sha`, `f19fcbf` slice 3.
  **Three things the plan got wrong, all caught by measuring before building.**
  (1) 640 vs 704 was not a 10% error — the pixel aspect cancels it exactly, and the
  error would have been introduced by "fixing" it. (2) The streak spec's "peak 1.25×
  background" was a **three-pixel** frame maximum, and its grain figure was the pin
  body, not the ridge. (3) The drop-volume dissent was void on both sides: the "core
  0.72×" it turned on came from sampling about the origin, and the drop has not been
  at the origin since 2026-08-07.
  **Three defects found by DRIVING the streak, not writing it** — a glint that blinked
  six times a revolution on `mitegen_200um`, a fake taper on the last 8 px of every
  hampton frame, and an arbitrary ridge angle on compact bodies. Each became a rule
  (a side the frame cuts / an end the frame cuts / `min_aspect`), not a patch.
  **Next:** the NA fork (Open questions), then the held rebuild. Nothing is
  half-finished; the only outstanding artifact is `hampton_300um_realistic`'s library.

- **2026-08-08 — the droplet scene's frame library shipped, after two traps fired.**
  `frame_library/hampton_300um_realistic/`: 360 frames, 1396×644 (supersample 1,
  n_cond 7, PSF, PNG), 2.9 MB, zoom 0.80–1×. Verified against a live f64 render at
  φ=30: **0,0 px registration, mean |diff| 0.00055, 99% of pixels identical**.
  **Trap 1 — the library CLI's tile-size "auto" is the probing ramp, and on WSL2 it
  spills.** An overnight build at auto crawled at **~45 min/frame** (11 frames in 9 h):
  the ramp sized the tile near the VRAM ceiling and WSL2 silently spilled to host RAM
  — the exact failure DECISIONS §2026-08-07 gives as reason 3 for rejecting probing as
  the render default. The builder's own spill warning cannot catch it (it baselines
  against early frames, and a run that is slow from frame 0 sets a slow baseline).
  Relaunched with an explicit `--tile-size 6800` (≈6 GB peak by the 160 B/ray/face
  law): **95.2 s/frame, rock-steady, 9.5 h total**. On WSL2, always pass an explicit
  tile for mesh-scene builds; the auto ramp is fine on native Linux where OOM raises.
  **Trap 2 — the server launch path destroyed the fresh library's manifest** (see the
  updated "asymmetry" entry under Other traps): a bare `--scene` launch graded the
  S=1 library stale, deleted `manifest.json`, and began a days-long S=4 rebuild. It
  was killed before any frame was overwritten; the manifest was **reconstructed
  deterministically** using `frame_library`'s own functions (the scout window
  recomputes byte-identically: 10.330 × 4.766 mm → 1396×644) and then verified by the
  live-render comparison above. Launch this scene's server with `--supersample 1`.
  Suite unchanged (**169**); no source edits this session — library + docs only.

- **2026-08-07 (pin joint, bundled scene)** — The same stem-pin gap existed in
  the hand-built `hampton_300um.yaml`: stems ended at x = 0.700, pin metal
  begins on the axis at x = 1.000 (same bevel geometry as the generator
  emits). Both stem fiber paths were extended in place, continuing their
  exact helix (R = 0.015 mm, 3000°/mm, same waypoint spacing) 19 waypoints to
  x = 1.041, inside the pin CSG. Tube render cost is set by `n_samples`, not
  waypoint count, so the perf-baseline character of the scene is unchanged;
  suite **169** green on the edited scene. This is a deliberate, measured
  exception to "never modify hampton_300um.yaml" (owner-requested; the rule's
  target was droplet-filling, which would change the render-cost class). The
  frame library was rebuilt (360 × 5578×2570 PNG, 29.3 MB, 47.6 min at
  7.93 s/frame, status `current`) and a served template verified to show the
  attached joint. The `hampton_300um_realistic` library build was started at
  `--supersample 1` (~8 s/frame projected → not measured; the mesh has no
  AABB cull so supersample 4 would be days) and **killed at the owner's
  request to re-run overnight** — the builder restarts from frame 0 by
  design, so nothing is lost; relaunch with
  `python -u -m loop_sim.library --scene scene_files/hampton_300um_realistic.yaml --supersample 1`.

- **2026-08-07 (pin joint)** — The stem now reaches the pin. `make_pin` places
  the scored break face `bevel_offset_mm` (0.3) PAST `tip_pos`, but the stem
  fibers were built to end AT `tip_pos` — a 0.3 mm air gap between nylon and
  metal, visible as a floating stem on rotation. The fibers now run past the
  tip and into the metal, so they emerge from the break face at every angle
  (the overrun is swallowed by the opaque pin), and the validator gained a
  mount-attachment check (stem endpoints must lie inside the pin CSG; the
  truncated-stem corruption is tested to fail). The chisel tip itself is
  deliberate — `pin_geometry.py` models a scored-and-snapped tube; `--pin-bevel
  0` gives a flat cut if the real BL831 pins warrant it. `mitegen_mounts`
  passes `make_pin` the same default score offset and was NOT touched — worth
  checking its polymer mount against x = 0.3 the same way. Scene regenerated;
  suite **169**.

- **2026-08-07 (scene fidelity)** — The black-droplet bug was diagnosed by a
  four-member council session and fixed the same day; generated scenes are now
  validated mesh-back. Suite **167** (was 149). Root causes (both scene-side,
  renderer untouched): the solvent's `color` was secretly a (24,18,6)/mm absorber
  (`color` IS an absorption spectrum — the ~0.26 ceiling the old NA sweep
  saturated at; falsified by changing only that field: NA-0.90 core
  0.2633 → 0.6436), and the silently-substituted hemisphere was a flat-bottomed
  plano-convex lens refracting ~92% of the drop's area past the NA gate.
  `MAX_DEPTH` was exonerated (exhaustion errs bright). Fixes: the Bashforth-Adams
  ODE (curvature-term convention error, `sin(ψ)/r` for `cos(ψ)/r`) was **deleted**
  for the closed-form biconvex spherical-cap lens `add_droplet.py` already had —
  one shared implementation in `crystal_harvester/droplet.py`; the droplet is
  pinned in the loop aperture (rim measured 6.8–10 µm from the fiber axis),
  straddles the loop plane, rotates with `loop_axis`, and hits `--solvent-volume`
  to 0.002%; the crystal sits at the droplet's volume centroid;
  `--contact-angle` is deprecated (pinned rim ⇒ contact angle is an output); all
  silent fallbacks are deleted and `crystal_harvester/validate.py` measures every
  generated scene back (18 new tests, corruption variants proven to fail).
  Also fixed: all five root scripts imported **James's old tree** ahead of the
  repo wherever `/home/jamesh/...` resolves (this workspace does) — the repo now
  wins. `hampton_300um_realistic.yaml` regenerated: drop core **0.67 at NA 0.10**
  against a 0.93 background (was 0.04), dark rim, crystal centred.
  `hampton_300um.yaml`, both frame libraries, and `loop_sim/` untouched.
  **Next:** the camera-calibration fork (see Open questions) — the zoom-lens
  hypothesis and a reference photograph of a real loop with a drop.

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
