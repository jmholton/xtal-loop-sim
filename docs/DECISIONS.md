# DECISIONS — loop-sim (xtal-loop-sim)

<!-- Append-only, newest first. Never delete an entry; if superseded, add a new one that
     says so. This log exists so the next person understands WHY the code is the way it is
     and doesn't "fix" a deliberate choice, or spend a week re-deriving a measurement that
     already exists. ../CLAUDE.md has the code-level detail; this file is the reasoning and
     the evidence behind it. -->

## Decisions

### 2026-08-06 — motion is a velocity profile, and the stage speeds were halved

**A real stage accelerates.** `_run_animation` interpolated linearly: instant full
speed, instant stop. Replaced with `velocity_step`, a trapezoidal profile — ramp up at a
fixed acceleration, hold, brake so the stage arrives at rest. `DEFAULT_RAMP_S = 0.15 s`
to full speed, **distance-independent** (fixed acceleration is what makes it a motor
rather than an eased tween); moves too short to reach full speed come out triangular.
Measured on a live 180° move, angular velocity by third: **149 → 170 → 143 °/s**.

**Speed is state, not a function of elapsed time — and that is the whole design.** A
preempted move hands its current speed *and heading* to its replacement. Deriving
position from a clock instead would restart every profile at v = 0, so a burst of jog
clicks would brake to a stop and re-accelerate at each one — the same per-click stutter
the duration floor was added to hide, returning in a subtler form. Speed is inherited
only when the new move continues the old heading; a reversal starts from rest, since it
needs the braking anyway. The heading test is a dot product over mixed units (mm and
degrees) — meaningless as a magnitude, correct as a *sign* for the same-axis case that
matters. Measured across a 20-click burst: the pose never stands still longer than
**38 ms**, and never once past 60 ms.

**The 0.25 s `min_time` floor was removed.** It was a stand-in for the missing ramp; with
a real profile a 15° jog is ~0.22 s on its own merits. Deleting it is a simplification,
not a regression.

**Rates halved** — `cross_time` 2→4 s, `rot_rate` 360→180 °/s, `zoom_rate` 4→2 /s. The
old speeds were roughly twice what the real goniometer looks like, so what needed the
speed dial at 0.5× is now 1.0×. Zoom was rescaled with the rest so the dial means one
thing on every axis, even though zoom is the microscope rather than the goniometer.
Note `move_duration` now returns the **constant-speed** time — the input to the stepper,
not the wall-clock duration of the move.

**Measurement note for whoever tunes this next.** The old jog harness counted MJPEG frame
gaps over 80 ms, a threshold calibrated when the JPEG library rendered at ~35 ms/frame.
The PNG library costs ~68 ms/frame through a slew, so that threshold now sits barely
above the render cadence and the metric measures decode cost rather than motion. Sample
the **pose** (`/motor`) instead — it isolates the animation from frame delivery and is
the measurement that actually answers the question.

### 2026-08-06 — a preempted animation could write its pose after losing the race

`_run_animation` checked `_anim_gen` and wrote the goniometer in **two separate critical
sections**, and the settle block had **no generation check at all**. A preempt landing
between check and write stamped the loser's pose on top of the winner's. Reachable today
by a `/motor` or a second `/move` at the wrong instant; it would become a guaranteed
corruption under runtime scene switching, where the cancelled animation would write the
old scene's pose onto the new scene's goniometer.

Both are now single `_anim_cv` acquisitions (existing `_anim_cv > _gonio_lock` order
preserved). The consequence is worth knowing because it simplifies everything built on
top: **a cancelled animation provably touches nothing**, so a scene swap never has to
join or quiesce the animator thread.

Related, same class: the `camera_cfg` reads in `_animator_loop` and `_command_move` moved
inside `_anim_cv`, and `_handle_recenter` now passes the click as a *fraction* for the
server to scale under its own lock. A target can no longer be resolved against one
scene's pixel size and another's axes.

### 2026-08-06 — realism pass: the objective PSF, and lossless templates

The bar moved from *fast and self-consistent* to *realistic*. The original model is a
useful starting point, not ground truth, and may be deviated from where physics says so.

**The renderer was sharper than the optics it models.** Ray tracing is geometric optics
with a binary NA collection gate — a ray is either collected or it is not — so it produces
edges no objective can form. Measured on hampton: **97.7% of a frame was pure 0 or pure
255**, and a silhouette edge resolved in ~1 template pixel where NA 0.10 at 550 nm has a
Rayleigh resolution of 3.35 µm and cannot beat ~1.8 template pixels. At zoom 4 the 4×
supersample budget is exactly exhausted (1.00 template px per output px), so nothing hides
it and the picture reads as blocky.

**Decision:** convolve the traced image with a Gaussian approximating the objective's Airy
PSF, `σ = 0.21 λ / NA`, λ = 550 nm hard-coded (`loop_sim/renderer/optics.py`). σ is
computed from `eff_px = pixel_size / zoom`, so it is a fixed size in *object* space and
scales correctly with both zoom and supersampling:

| scene | camera px | σ at camera res | template px | σ stored |
|---|---|---|---|---|
| hampton_300um | 7.40 µm | 0.156 px | 1.85 µm | 0.624 px |
| mitegen_200um | 1.00 µm | 1.155 px | 1.00 µm | 1.155 px |

Which is also why the softening only appears when you magnify — exactly where the
geometric sharpness became visible. The X-ray path is untouched: no objective, no PSF.

**One implementation, called by both renderers.** `microscope.render` and `render_torch`
are asserted equal after quantisation; two blur implementations (scipy here, a torch conv
there) would differ in kernel truncation, normalisation, border handling and summation
order. So the PSF is numpy-only and the torch engine round-trips through it. The caller
transfers the result to the host immediately anyway, so the extra sync is cheap. This also
preserves the documented property that the CPU reference needs no PyTorch.

**A pre-existing float divergence surfaced, and the parity claim needed restating.** The
two float64 traces were **never bit-identical**: measured, they differ by up to **3e-8 on
~0.7% of values** (numpy vs torch summation order and library differences). That was
invisible while the image was essentially binary — 0.0 and 1.0 quantise the same either
way. The PSF redistributes those values into intermediate greys, where a 3e-8 difference
can land either side of a rounding boundary. Result: with the PSF on, CPU and GPU agree to
**±1 grey level**, never more, measured at 96×72 and full res on both CPU and CUDA.

The tests keep both properties rather than trading one away: the exact `== 0` assertions
still run with `psf=False` and still guard the trace, and a second family asserts `<= 1`
with the PSF on. If that bound is ever exceeded, something structural has broken. The
claim to make in future is therefore precise: *the geometric trace is byte-identical; the
delivered image agrees to one grey level.*

**Templates are stored losslessly (PNG).** A real AXIS camera applies exactly **one** JPEG
compression; storing JPEG templates and re-encoding on the wire applied **two**, a
signature no real camera has. The wire stays JPEG — MJPEG requires it — so this collapses
the pipeline to a single generation of loss. PNG is also *smaller* here, which makes it a
free choice rather than a trade: measured on the rebuilt hampton sweep, **28.7 MB against
the 84.9 MB it replaced — very nearly 3x smaller**, because the frame is overwhelmingly flat black and white, which deflate
handles far better than JPEG — which spends its bits ringing around exactly the hard edges
that matter. Decode is dearer (55 vs 36 ms), costing only on a spindle slew where every
frame is a fresh decode: measured **68 ms/frame, 14.7 fps** through a sustained spin
against ~46 ms / 21.6 fps before, still comfortably above the 10 fps goal. Panning at a
fixed angle reuses the decoded template and is unaffected. `compress_level` is deliberately **not** a build key: it changes file size,
never a pixel.

`format` and `psf` are build parameters, so a pre-PSF or JPEG library correctly reads as
stale. `build_library` now also deletes frames whose extension no longer matches — frames
are overwritten in place, so a format change would otherwise strand the old ones on disk
and in git, silently doubling the shipped library.

**Auto-rebuild was deliberately left alone.** It was tempting to make a stale library
refuse to rebuild and demand an explicit command, since the rebuild is 45 min. Rejected:
the delivery goal is that the team never runs a build step. The project ships with
pre-rendered scenes, and if a library is ever stale the server should quietly regenerate
it rather than block someone who just wants a camera. Note the consequence — a library
left un-rebuilt costs the *next* person the wall-clock, so rebuild deliberately before
handing over.

**Caveat to carry forward.** `na_condenser / na_objective = 0.70` < 1, so this is
partially coherent imaging. Intensity is only a straight convolution with the PSF in the
fully incoherent limit; at 0.70 real edges overshoot and ring in a way a Gaussian will not
reproduce. This is a large step toward realism, not the end of it. Do not read the
softened edges as exact — measuring an edge position or a droplet boundary off a rendered
frame inherits this approximation.

### 2026-08-06 — the interactive path: what the operator sees must be what the server means

Serving from templates made frames cheap (13–46 ms), but the *interactive* path had never
been driven by a person. Doing so surfaced five defects, all of the same character as the
2026-07-31 crop of bugs — the picture looks entirely plausible while being wrong, or the
numbers on screen disagree with the picture. Each is recorded with its measurement.

- **`--fps-limit` default 5.0 → 30.0.** The MJPEG handler clamps the wire rate to
  `1/fps_limit`. 5 fps was a sensible ceiling when every frame was a ~1 s live raytrace;
  once templates cut a frame to ~35 ms it became the binding constraint, and nobody
  re-tuned it. Measured on the same server, same scene, changing only the flag: at
  `--fps-limit 5` every inter-frame gap was 200 ms (min 200, max 204) for **5.12 fps**; at
  30 the median gap is 34 ms for **28.1 fps**. The headline 24 fps in the template work
  was a *render-cost* number the shipped default could not deliver — anyone following the
  RUNBOOK and watching a browser would have concluded the template work did nothing.
  A static pose still publishes nothing and rides the 1 s keepalive, so an idle stream is
  ~1 fps by design; that is not the cap.

- **MJPEG parts are closed as they are written, and new content is followed by one prompt
  resend.** The obvious framing writes each part's boundary *before* its payload, so the
  last frame of a motion stays unterminated until the next send — a full keepalive away.
  Consumers differ in when they consider a part finished, and the cost falls entirely on
  the stricter ones. Measured on one move, three client behaviours on the same stream:

  | client finalises a part when… | worst gap | final frame visible |
  |---|---|---|
  | `Content-Length` is satisfied | 35 ms | +0.86 s |
  | the closing boundary arrives | 1002 ms → 35 ms | +1.86 s → +0.86 s |
  | the NEXT part's headers arrive | 1002 ms → 35 ms | +1.86 s → +0.90 s |

  Closing each part immediately fixes the second row. It does **not** fix the third, and
  cannot: `Content-Length` is unknown until the next frame exists, so the next part's
  headers cannot be sent early. That row needs one duplicate frame sent promptly (one
  frame interval) after any new content. Both mechanisms are required; either alone
  leaves some consumer holding the previous frame for a second, which reads as the stage
  stalling short of target and then teleporting. **This regressed twice during one
  session** — once by removing the prompt resend after adding the reframing, on the
  reasoning that it was now redundant. It is guarded by
  `tests/test_server_singleflight.py::test_new_content_is_followed_promptly`.

- **Screen-space pan is resolved through Rᵀ, like `recenter_target` always was.**
  `resolve_target` added `panx`/`pany` straight into `tx`/`ty`, which are *motor* axes.
  The XYZ stage rides on the spindle, so those coincide with image axes only at φ=0. The
  same request produced an identical `ty = −0.888 mm` at every angle: at φ=90 that is pure
  defocus (the image does not move at all) and at φ=180 it moves the image backwards. The
  pan is now built as a lab-space displacement from the camera `fast`/`slow` axes and
  mapped into motor space by `Rᵀ`, reducing algebraically to the old expression at zero
  rotation. Verified against the live server by FFT phase correlation — commanded vs
  actual image shift, 0 wrong out of 14 across φ = 0/30/45/90/135/180/270.
  **Consequence worth knowing:** a screen pan now writes `tz`, so anything that "returns
  to the origin" must zero `tz` too. The recenter button did not, and was a no-op at φ=90.

- **The commanded pose is clamped to what the library can serve, not just the crop.**
  `pose_crop(clamp=True)` slid the crop box and left the pose where the operator put it,
  so the readout and the target boxes advertised a pose that was not on screen.
  `servable_pose()` reports the nearest servable pose and the server now clamps the
  command itself. It is not a second implementation of the clamp: it calls `pose_crop`
  and inverts its box back into motor coordinates, so the two cannot disagree. Depth is
  carried through unclamped — it only defocuses. Invariant, checked live: re-requesting
  the pose the server reports reproduces the image byte-for-byte.

- **Non-zero moves get a 0.25 s duration floor.** A 15° φ jog is 42 ms at 360 °/s — about
  one frame — so it arrived as a jump, and a burst of clicks read as N separate jumps
  rather than one rotation. With a floor, a single click glides and a click landing while
  the previous is still running preempts and extends it. A/B on the same 20-click burst
  at 120 ms intervals: **18 stalls >70 ms → 1**, median gap 68 ms → 39 ms, 39 → 67 distinct
  frames. The floor is a floor, not a fixed cost: a 360° spin still takes 1.0 s and the
  speed dial still scales it. Zero-distance moves stay 0 so a no-op does not animate.

**The lesson that generalises:** every one of these was invisible from the server side.
Render-path timings, unit tests and byte-comparisons all looked healthy while the thing a
person actually experienced was broken. Measure the delivered stream and the on-screen
numbers, not just the renderer.

### 2026-07-31 — templates serve every frame; VRAM stops limiting resolution
- **Decision:** the camera server serves *every* frame from the pre-computed sweep
  (`--templates on`, default): pick the nearest spindle angle, crop, scale, blur, encode.
  Live rendering remains available (`--templates off`) and is still the correctness
  reference. Startup calls `ensure_library`, so the tool checks for templates and builds
  them if missing before serving.
- **Measured:** serving costs **1–16 ms/frame with no GPU at all** (decode 1.4/4.6/21.4 ms
  at 1×/2×/4× supersample, plus crop+scale+encode). The old goal was 10 fps of live
  raytracing; this is faster than that by a wide margin and the frame rate no longer
  depends on the hardware.
- **`supersample` is the zoom ceiling, and 4 is a physical number, not a guess.** NA 0.10
  gives a Rayleigh resolution of 0.61λ/NA = **3.35 µm**; Nyquist wants 1.68 µm/px against a
  7.4 µm native pixel, so **4.41×** is where sampling critically matches the optics. Below
  it real fiber detail aliases; above it you magnify resolution the objective cannot
  deliver (the renderer is geometric and models no diffraction, so it will happily keep
  producing sharper edges that no real instrument would show). If the camera-calibration
  question ever resolves toward `template.yaml` (0.82 µm px, NA 0.28) the answer becomes
  **1.37×** — hence the flag.
- **The render window is measured from the scene, not centred on the goniometer origin.**
  The hampton pin runs to x=6.7 mm against a 4.736 mm field, so a symmetric margin left
  more than half the pin unrendered and panning right scrolled in blank background.
  `content_window()` renders a coarse wide-field scout sweep and measures where the image
  differs from background — this works for CSG and half-spaces, which have no finite
  bounding box. The sweep is then rendered at a fixed `tx` offset that centres that window;
  `tx` is parallel to the spindle axis, so a constant offset is rotation-invariant and
  exactly equivalent to moving the camera. Costs nothing: the scene-anchored window is
  10.32 × 4.76 mm where a symmetric 1.5× margin was 7.10 × 5.33 mm.
- **Depth is a Gaussian blur, not a focus stack.** `σ_px = 0.5 · NA_cond · |Δz| / eff_px`
  keeps build time and disk independent of focus, at the cost of not reproducing the
  discrete 7-replica ghosting a real `n_cond=7` render shows.
- **What breaks if you change it:** the crop math depends on the goniometer composing
  `T = Rz·Ry·Rx·T_trans` — the stage rides on the spindle, so the lab displacement is
  `R·vec` and *which motor is lateral rotates with φ*. At φ=90 a `ty` move produces zero
  image shift and `tz` produces all of it. Reversing that leaves the picture looking
  entirely plausible while showing the wrong part of the sample;
  `tests/test_frame_library.py::test_template_matches_live_render` is the guard.
- **Three sub-pixel constraints, all found in review, all invisible to an
  integer-shift comparison:**
  1. `pose_crop` returns a **float** source box consumed by
     `Image.resize(size, box=…)`, not an integer crop box. PIL maps output pixel *i* to
     source *edge* `left + (i+0.5)·scale`, i.e. index `left + i·scale + 0.5(scale−1)`,
     so the box origin must sit half a source pixel early. Dropping that offsets every
     served frame by 0.375 camera px at 4×.
  2. Rounding the box to integers also makes the span vary by ±1 template px at
     non-integer `scale`, and the result is always resized to exactly *W* — so
     magnification flickers ~0.1% as a pan crosses pixel boundaries.
  3. The template must have the **same parity** as the camera, or `(RW−W)/2` is a half
     integer and every crop inherits a half-pixel offset. This one was caught by
     measurement: it showed up as a systematic ±1 px wobble and a 6× worse residual.
- **The zoom floor is not `camera / template`.** The window is anchored on the sample and
  the sample is long and thin, so it is deliberately off-centre; at the home pose the
  camera runs out of room on the near side before the span stops fitting.
  `zoom_limits()` measures from the nearer edge. Clamping **slides** the crop and never
  squeezes it — squeezing the axes independently changes magnification per axis, hence
  the aspect ratio, and looks entirely plausible on screen.

### 2026-07-31 — the tile clamp was the VRAM ceiling, and it was never a correctness one
- **Finding:** `render_torch` already traced in batches, but `tile_size = max(tile_size, WH)`
  forced every tile to be at least a whole frame, so peak memory scaled with resolution.
  **This supersedes the 2026-07-28 entry below**, which concluded the clamp made the tiling
  fix "unreachable" and that droplet scenes were unrenderable at full resolution. The clamp
  was a performance floor. Removing it is safe.
- **Verified before relying on it:** tracing the same pose in tiles of 307200, 100000,
  37649 (deliberately straddling condenser-sample boundaries), 8192 and 1000 rays is
  **byte-identical** to a single-pass trace, at φ = 0, 37 and 90. The comment above the
  clamp had said as much; it is now a test.
- **Second ceiling, also removed:** `o_all`/`d_all`/`o_t`/`d_t`/`out` were allocated at the
  full `M = n_cond × W × H` before any tracing — 5.4 GB on device at 10.7 Mpx and n_cond=7.
  Tracing one condenser sample at a time and accumulating drops that 7×, making peak memory
  independent of `n_cond` as well as resolution. The accumulation is a sequential
  `sum / n_cond`, which matches the numpy reference's own reduction order; all 62
  pre-existing tests stayed green through the change.
- **Measured law:** peak ≈ `0.12 + 1.13 × tile_Mrays` GiB on this tube scene. The slope is
  scene-dependent, so it must be measured per scene, not assumed.
- **Size the tile by a measured doubling ramp, not by extrapolating a slope.** The first
  implementation fitted `k` from two 64 k-ray probes and solved for a ~10 M-ray tile — a
  150× extrapolation. On the mesh path it under-predicted by ~2.7 GB and pushed a
  `mitegen_200um` render to 15.7/16 GB, i.e. straight into the WSL2 spill. Two corrections:
  probe against **reserved**, not allocated (the mesh Möller-Trumbore temporaries fragment
  the pool, so reserved is what actually fills the card and it exceeds allocated), and
  **measure each rung** rather than extrapolating, stopping before a rung that would exceed
  the budget. No probe can then trigger the failure it is sizing to avoid.
  - Trap found while writing the ramp: sampling the probe rays through **one fixed stride**
    caps the sample's length, so every rung past it re-measures the same rays, reports a
    flat cost, and the ramp doubles to the top — which then attempted an **87 GiB**
    allocation. The stride has to be recomputed per rung.
- **Mesh scenes are memory-bound on `TSurfaceMesh`, which still has no AABB cull.** It
  brute-forces `(B, F, 3)` Möller-Trumbore intermediates, so at 2880 faces a single ray
  costs ~69 kB and the tile that fits a 12 GB budget is only ~262 k rays. That is why
  `mitegen_200um` renders at 18 s/frame against `hampton_300um`'s 7.2 s despite being
  3.4× *smaller*. The cull `TTube` already has is the fix, and it is now purely a speed
  optimisation — the scene renders correctly either way.
- **Sizing is predictive, never try-and-retry.** Under WSL2 there is no OOM to back off
  from: past capacity the driver silently spills to host RAM and the render crawls 10–50×,
  so a retry loop would hang rather than recover. A sustained per-frame slowdown during a
  build is the only available spill signal and is now warned about.
- **Result:** a 14.34 Mpx template renders at **4.7 GB peak** where the unclamped path
  needed ~14 GiB and spilled. The same templates build on an 8 GB card, and on the
  beamline's 12 GB TITAN V, in more passes.

### 2026-07-28 — deliver a pre-computed rotation sweep, not a faster live renderer
- **Decision:** for the AXIS-camera use case, ship a **pre-rendered frame library**
  (`loop_sim/library/`, output in `frame_library/`) instead of pushing live frame rate
  further. A full 360° sweep about the spindle is rendered once; the consumer replays it.
- **Why it collapses the problem:** the camera is **orthographic**, so translating the
  sample sideways shifts the image by an exact whole number of pixels and changes nothing
  else — measured, `tx` of 5 px and 20 px worth reproduced the un-translated frame rolled
  by exactly +5 and +20 px, **max pixel difference 0.000000**. Panning is therefore a
  *crop*, not a render. Rotation is the only motor that genuinely changes image content,
  so the interactive envelope needs one sweep, not a grid over all seven motors. At 1°
  steps that is 360 frames; a frame is ~12–19 kB JPEG (640×480, scene with content), so a
  library is single-digit MB and seekable instantly.
- **Design constraints that are load-bearing:** frames are rendered `margin`× larger than
  the camera (default 1.5) so there is material to pan into — `pan_px` in the manifest is
  the crop limit, and `crop_window()` raises rather than silently clamping past it. The
  manifest carries a SHA-256 of the scene YAML, so an edited or new scene rebuilds on
  first use (`ensure_library()`). Build failures raise; they never degrade quality
  silently, because a library rendered at a reduced setting is indistinguishable from a
  good one afterwards.
- **What this does NOT cover:** `zoom` and `tz` (focus) are not free the way lateral
  translation is — they need their own sweeps or a live render. The library is invalidated
  by any scene change, so it suits a fixed sample being explored, not scene authoring.
- **`.gitignore`:** the repo ignores `*.jpg` globally; `frame_library/**/*.jpg` and its
  manifests are explicitly re-included. The library is a deliverable, not build output.

### 2026-07-28 — scene-fidelity audit: the physics is sound, the bundled scene is not
- **Finding (positive):** the geometry-to-image chain is **dimensionally correct**. The
  pin's ground-truth diameter is 700.0 µm; measured in the rendered image at four
  independent columns it is **703.0 µm every time** (0.4%, i.e. the half-pixel edge
  threshold). Camera model, pixel size, projection, and the goniometer transform are all
  right. This is an architecture-independent check — unlike the GPU↔CPU parity gates, it
  compares against physics rather than another render on the same machine, and it is the
  cheapest available answer to the "is any of this correct?" question.
- **Finding (negative):** `scene_files/hampton_300um.yaml` — the scene every test,
  benchmark and the TITAN V acceptance number runs on — is **not a realistic sample**. Its
  `solvent` object is a sphere of `radius: 0.0` (no droplet at all), it has **no crystal**,
  and its `loop_fiber` waypoints span **69 × 200 µm** despite the `300um` name. The
  radius-0 sphere was optimised around rather than questioned (commit `77fa545`,
  "radius-0 Sphere -> TNull"). The performance work is real; it was tuned on a bare fiber
  and a pin.
- **Why not just fill the drop in:** a real droplet is a `SurfaceMesh`, and the mesh path
  is exactly what exhausts VRAM (below). Populating the benchmark scene would make it
  unrenderable at full resolution *and* invalidate every fps number measured on it.
  Prefer a second, generated scene for fidelity work (see HANDOFF "Two scenes, two jobs").
- **Scene generation is trustworthy:** `crystal_harvester` produces dimensionally correct
  geometry — its 300 µm circular loop measures 300.4 × 300.0 µm, the droplet mesh spans
  300 × 300 × 150 µm, and the pin is exactly 700 µm. The hand-built bundled scenes are the
  outlier, not the generator.

### 2026-07-28 — the mesh VRAM law, and why the documented tiling fix is unreachable
- **Measured law:** mesh peak memory ≈ `tile_rays × faces × 24 bytes`, multiplied by ~6
  for the Möller-Trumbore temporaries. It is driven by the **tile size and the face count,
  not by image resolution**.
- **The clamp is the binding constraint.** `render_torch` does
  `tile_size = max(tile_size, W*H)`, so at 640×480 the tile can never be smaller than
  307,200 rays: 307200 × 2880 faces × 24 B ≈ **19.8 GB predicted, 19.78 GiB observed as an
  OOM**. This **corrects the 2026-07-17 entry's safety note** — calling
  `render_torch(..., tile_size=32768)` does nothing at full resolution, because the clamp
  raises it straight back to `W*H`. The tiling fix is only reachable by relaxing that clamp.
- **Reducing resolution does not rescue it:** 320×240 still OOMs (predicted 4.9 GB × ~6
  temporaries). Only 160×120 renders, at ~8 GB peak. Half-resolution rendering is a real
  *speed* lever (640×480 → 320×240 measured **3.5× faster**, 1.29 s → 0.37 s) but a weak
  *memory* one (only **1.7×** less VRAM, 0.33 → 0.19 GB, because resident scene geometry
  is a fixed floor) — and it is a poor trade here regardless, since the loop fiber is only
  ~3 px wide at the scene's own pixel size and sub-sampling destroys the feature under
  inspection.
- **The durable fix remains the `TSurfaceMesh` AABB cull** (`TTube` already has it,
  `engine_torch.py:59`): it attacks `faces`, which is the term that actually needs to fall.

### 2026-07-17 — TITAN V measured: 10 fps confirmed (11.9 fps), gated on the software stack
- **Finding:** the 10 fps target reproduces on a real voltron TITAN V — **compiled preview
  11.9 fps median / 10.1 fps p90 (GO)**, eager fallback 6.3 fps — measured by
  `acceptance_voltron.py`. The GPU and its VRAM were never the bottleneck; the beamline's
  default *software* stack is. This **supersedes the predictions** in the 2026-07-14 analysis
  entry below on two points: the mesh scene does **not** hard-OOM (it fits torch 2.6 at
  ~11.1 GB — a knife's-edge fit, ~tens of MB free after the CUDA context; it OOMs torch
  2.0.1), and the 2016 Xeon does **not** drag the compiled path (compile fuses ~8k launches
  into ~8 graphs → GPU-bound → the CPU stops mattering; the eager path stays CPU-bound at
  6.3 fps).
- **Why the stack is load-bearing:** `torch.compile` (the whole 10 fps lever) needs torch ≥
  2.x AND a modern C compiler at runtime. Voltron's defaults block both — the pt env ships
  torch 2.0.1 (an Inductor `pkg_resources` failure) and the system gcc is 4.8.5 (too old for
  Inductor's `stdatomic.h` codegen). torch 2.6 + devtoolset-7 clears both. The exact,
  reproducible recipe is in RUNBOOK "Deploy on the TITAN V".
- **What breaks if you ignore it:** `camera_server` catches any compile failure and silently
  runs eager (6.3 fps) — the server looks healthy and just misses 10 fps. Treat a working
  compile as a deployment precondition, not a given: make the fallback loud, and verify with
  `acceptance_voltron.py` (its verdict is GO only when compile actually engaged and beat
  eager).
- **VRAM note:** mesh scenes are a knife's-edge fit on 12 GB even on torch 2.6. The
  byte-exact sub-frame-tiling fix (or the `TSurfaceMesh` AABB cull) is the safety margin —
  see HANDOFF risk A.

### 2026-07-06 — 10 fps interactive via a flag-gated `torch.compile` preview path
- **Decision:** motion/preview frames route `next_interface` (and the tube `_kernel`,
  compiled *separately* with a dynamic survivor batch) through `torch.compile`; settled
  frames stay eager. Flags: `--compile-preview` (default on), `--preview-mode` (default on).
- **Why:** the frame is CPU-dispatch-bound (~8k kernel launches/frame); the pure-PyTorch
  dispatch ceiling is ~11 fps. `torch.compile` fuses the launch explosion (~1.7–1.86×),
  which is what clears the 10 fps bar.
- **Design constraints that are load-bearing:** CUDA-only (Inductor's CPU backend
  *miscompiles* the mesh/CSG path — mitegen came out 1734/6912 px wrong); `mode="default"`
  with `dynamic=True`; warm the compiled fn **single-threaded in `start()` before any
  worker thread** (concurrent first-compile crashes dynamo); the compiled/eager choice is
  threaded **explicitly** through the call chain, never shared mutable state (a shared
  `_active_compiled` flag caused a measured `/xray`-vs-bg-render data race); sticky
  eager fallback on any runtime failure.
- **What breaks if you change it:** Inductor reorders float arithmetic at tangent
  boundaries → shifts `best_t` ~1 ULP → flips ≤~16 genuinely-tangent CUDA pixels, so the
  strict `max==0` parity gate is relaxed to a **≤16 px CUDA flip budget** (CPU stays
  exact — compile is off there). Do not tighten that gate back to 0 on the compiled path.

### 2026-07-06 — converge-on-idle preview policy (approximate in motion, exact at rest)
- **Decision:** during motion the server renders `n_cond=1` (+ optionally compiled)
  previews; a full-`n_cond` **bit-exact f64** frame renders automatically once the pose
  settles. Offline renders and `/xray` are always exact. This applies to instant `/motor`
  sets too (the AXIS-consumer path): a stream of `/motor` updates gets fast previews, one
  exact frame on quiet (`--settle-delay`, default 0.5 s).
- **Why:** loop-sim's real purpose is a drop-in AXIS camera producing ~10 image/s; motion
  smoothness matters, per-frame exactness during motion does not. Settled output is never
  degraded, so nothing permanent is sacrificed (test-gated: `test_server_settle_parity`).
- **What breaks if you change it:** making motion frames "exact" collapses fps; the
  invariant is *exact-when-settled*, not exact-always.

### 2026-06-25 — GPU-resident torch engine, byte-identical to the numpy reference (f64)
- **Decision:** `loop_sim/renderer/engine_torch.py` runs the *whole* trace on-device
  (`TorchScene`, `render_torch`); the numpy `microscope.py` path is **FROZEN as ground
  truth** and kept as the CPU fallback. The active ray set uses **compaction**, not masking.
- **Why:** profiling showed ~84% of the render was numpy on the CPU — only the
  intersection touched CUDA, so the GPU sat ~84% idle. Keeping all ray state resident (1
  upload / 1 download per frame, condenser batched) was the real speedup (~6–8×), not
  faster kernels.
- **What breaks if you change it:** do not let the torch engine diverge from the numpy
  reference — it is both the correctness anchor and the no-GPU fallback. Parity is
  enforced by `tests/test_torch_render_parity.py` (incl. full-res CUDA).

### 2026-06-25 — float64 in the GPU intersection quadratic (THE correctness fix)
- **Decision:** `TTube`/`TSurfaceMesh` compute the intersection quadratic in float64
  internally, always — even on CUDA. (`tube.py`, `surface_mesh.py`, `engine_torch.py`.)
- **Why:** float32 **catastrophically cancelled** in `c_ = baba*oaoa - baoa² - r²*baba`,
  amplified by the microscope's 50 mm ray-march origin offset. For a ~7.5 µm fiber 50 mm
  upstream, `oaoa ≈ 2500 mm²` swamps the r² signal (~6e-9, below the float32 noise floor
  ~1.6e-8) → discriminant sign flips → binary hit/miss + a **median ~9 µm t-error** → the
  hit point lands off-surface → poisoned normals (~19–32°) → Snell sends the ray ~20° off →
  it trips the hard binary NA cutoff / TIR (NaN→0) gates → full 0↔255 pixel flips = the
  speckled **"hairy/spikey" fiber**. Decisive isolation: forcing *only* the tube
  intersection to f64 (rest stays CUDA) dropped the hampton roty45 n1 diff **203 px → 0**.
- **What breaks if you change it:** down-casting the intersection geometry or inputs to
  float32 brings the artifact straight back. This is also *why* fp32 preview is rejected
  (below). (`tests/test_gpu_cpu_parity.py` guards it.)

### 2026-06-25 — threaded camera server
- **Decision:** `CameraServer` inherits `ThreadingHTTPServer` (one thread per connection).
- **Why:** a plain single-threaded `HTTPServer` was starved by a browser's idle preconnect
  socket — the MJPEG stream "loaded forever" and `/motor` couldn't be served while
  streaming.

### (inherited from James's model — recorded so they aren't "cleaned up")
- **The twisted-pair stem must be TWO separate tube objects** (`stem_1`, `stem_2`). A
  single helical tube models only one fiber; the crossover shadows disappear and the stem
  looks wrong.
- **Crystal must be listed before droplet** in scene priority. A crystal voxel is also
  inside the droplet; if droplet is first, every crystal voxel is assigned `solvent` and
  the crystal renders invisible.

### 2026-07-14 — deploy target is a TITAN V; GPU speed is NOT the bottleneck (analysis)
- **Decision/finding:** the beamline runs loop-sim on voltron's **TITAN V** (Volta,
  12 GB), but the 10 fps result was only ever measured on an RTX 4080 SUPER. Profiling
  shows the frame is **CPU-dispatch-bound** (73–85% self-CPU, GPU ~29% busy, ~52–65 W of
  320 W), so the TITAN V's 8.5× FP64 advantage is a **red herring** — the dominant ops are
  low-intensity elementwise, not FP64-ALU-bound. The real deployment risks are A (12 GB
  mesh OOM), B (silent `torch.compile` fallback), C (unknown voltron CPU) — see HANDOFF
  "Hazards".
- **Verified measurements (don't re-derive):** hampton n_cond=1 f64, worst-case
  centred-loop `/motor` pose — eager **147.5 ms (6.8 fps)**, compiled **88.7 ms (11.3
  fps)**; per-launch ~**8.25 µs** on WSL2 GPU-PV (native Linux is typically 3–5 µs, so
  voltron *may be faster* CPU-side). **L2 inversion:** the `(N,3)` f64 chain is L2-served
  on the 4080 (64 MB L2) but a 7 MB working set won't fit the TITAN V's **4.5 MB** L2 → it
  falls to HBM2 → real GPU regression ≈**1.5×** (still hidden unless the GPU must be ~2.6×
  slower to bottleneck — margin is thin; this is the #1 thing `ncu` should settle).
- **What breaks / traps:** **byte-exact gates are architecture-blind** — they compare
  against a reference computed on the *same* machine, so a Volta box could pass every gate
  while producing wrong images. Add a numpy-anchored golden-image gate before trusting the
  target. Tooling notes: **CUPTI works under WSL2** (an older note claiming otherwise is
  wrong); `torch.profiler` reports zero GPU time due to a CLOCK_REALTIME vs
  CLOCK_MONOTONIC mismatch (fix: env `CuptiUseRawGpuTimestamps=false`). `bench_results/`
  is gitignored → commit baselines if you want them to travel to voltron.
- **Status:** the profiling/acceptance suite is **PLANNED, zero code written.** Intended
  deliverable: a self-contained `acceptance_voltron.py` the beamline staff run themselves
  (no Nsight needed) → one JSON back, making the perf prediction falsifiable. Cheapest
  first experiment: the **OOM canary** — `torch.cuda.set_per_process_memory_fraction(12/16)`
  caps the 4080 to a Titan-V-sized 12 GB *today*.

## Already Tried

<!-- Evidence, not fences. Each entry is here so nobody spends a week re-deriving a
     number that already exists. -->

### global Lagrange / Neville polynomial waypoints for the fiber path
- **Status: already implemented, and deliberately bounded.** Every fiber path — the loop,
  both stem strands, the droplet rim, the Kapton outline — is built by `neville_sample()`
  in `loop_sim/scene/tube.py`, and Neville's algorithm *is* Lagrange interpolation
  (same polynomial, stabler evaluation). So "use Lagrange waypoints" is the current design,
  not a change.
- **What was measured:** the global form is used only for **≤ 4 waypoints** (degree ≤ 3).
  Above that the code switches to `scipy` `CubicSpline`, because global Neville at degree
  5+ produces Runge-phenomenon knots on curved paths. Real loop paths carry 40–59
  waypoints, so a global fit would be degree ~58 and would put visible oscillations in the
  fiber. The interpolant also runs **once at scene load**, so it is not on the render hot
  path and offers no speed lever.
- **Where fiber quality is actually lost:** the curve is sampled into `n_samples - 1`
  capsules and every ray tests all of them. At the default `n_samples=50` a 300 µm loop
  gives **19.3 µm segments against a 20.0 µm fiber diameter** — capsules as long as they
  are wide, which renders as a visibly beaded ring rather than a smooth fiber. Raising
  `n_samples` fixes the appearance and costs render time roughly linearly.
- **Would be worth another look if:** the goal is *performance* rather than smoothness —
  in which case the target is intersecting an analytic swept surface, or hierarchical
  culling over the capsule chain, rather than changing the interpolant.

### fp32 preview mode
- **What was measured:** on the sync-starved compiled engine, fp32-compiled is **~2×
  SLOWER** than f64-compiled (≈188 ms vs 120–126 ms on worst-case poses), and fp32-eager
  also loses on the worst case. The f32↔f64 cast traffic at the **deliberately-float64
  tube-kernel boundary** outweighs the halved memory bandwidth, and fp32 perturbs the
  material-probe interval enough to change ray-path work profiles. The often-cited June
  "~2× fp32 win" microbench **predates this pipeline** and no longer holds. Reproduce with
  `bench_frame.py --fp32` (the negative result is recorded in commit `fb38fdb`).
- **Would be worth another look if:** the whole engine went fp32-native. Note the
  constraint that shapes this: the intersection quadratic has to stay f64 (see the
  correctness decision above), so any fp32 scheme has to keep that boundary — which is
  exactly where the cast cost showed up.

### `torch.compile(mode="reduce-overhead")` / CUDA graphs in the threaded server
- **What was measured:** `reduce-overhead` silently enables CUDA-graph capture, which is **not
  thread-safe** in the `ThreadingHTTPServer` ("already recording to mempool_id" /
  `CUBLAS_NOT_INITIALIZED`). Whole-frame graph capture is *also* blocked by the AABB cull's
  data-dependent `nonzero` (a graph-break), and dropping the cull to enable capture is a
  **catastrophic 1.95 fps** (a 50× tube blow-up). The fast algorithm (AABB cull) and
  whole-frame capture are mutually exclusive.
- **Would be worth another look if:** a capture strategy that tolerates dynamic shapes.

### custom Triton megakernel (one thread per ray, whole bounce loop in registers)
- **What was measured:** proven **byte-identical** but only **~1.1×** (54 ms vs torch 60 ms) on
  this fp64 thin-fiber workload — it's GPU-compute-bound on fp64 and trades dispatch
  overhead for dense-depth compute, a wash on this geometry. Not worth the complexity; it
  was a throwaway prototype and no code remains.
- **Would be worth another look if:** fp32 (with its cancellation risk) + a single-pass
  tube + 2D ray tiling — uncertain.

### dense alive-mask `trace_rays` (instead of compaction)
- **What was measured:** byte-identical but **regresses** (5.6→6 fps — it runs 12 full-N depths
  vs ~3 with compaction), and CUDA graphs still don't engage (the cull's `nonzero`
  graph-breaks). Compaction is the correct design.

### "fix the hairy fiber" via normals, gate-softening, or more condenser rays
- **What was measured (all four refuted):** the root cause is the float32 intersection **t**, not the
  normals or the gates. `recompute_normals_f64` is already optimal given a correct t (0°
  error); forcing CUDA's `best_k` makes normals *worse*; softening the NA/TIR gates catches
  ~0% of the flips; and **raising n_cond makes the fiber hairier** (the flip count roughly
  doubles), not smoother. Float64 at the quadratic is the only fix.
