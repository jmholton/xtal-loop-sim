# DECISIONS — loop-sim (xtal-loop-sim)

<!-- Append-only, newest first. Never delete an entry; if superseded, add a new one that
     says so. This log exists so the next person understands WHY the code is the way it is
     and doesn't "fix" a deliberate choice, or spend a week re-deriving a measurement that
     already exists. ../CLAUDE.md has the code-level detail; this file is the reasoning and
     the evidence behind it. -->

## Decisions

### 2026-08-14 (later still) — the viewer pre-warms, and every "GiB of cache" figure in this repo was 30% low

**Pre-warm.** `--template-cache auto` sized the cache to hold the sweep but filled
it LAZILY, so the first revolution after a restart paid a decode on every frame and only
the second ran at the promised rate. `TemplateSource.prewarm()` now decodes the library at
boot, blocking, before the socket is bound — measured on the dev box, the FIRST revolution
goes **28.01 ms (35.7 fps) → 13.24 ms (75.5 fps)** for 5.3 s of startup. `--prewarm off`
restores lazy filling. It refuses when the cache cannot hold a whole revolution, because a
partial warm is evicted before it is used.

Deliberately NOT in `TemplateSource.__init__`: `bench_serve` builds one to measure COLD
costs, and a constructor that quietly decoded 360 frames would destroy that measurement and
add 10-30 s to every run. It is also blocking rather than threaded — the server's threading
rules are strict, and a few seconds at boot is the cheaper trade. `_build_bundle` warms a
switched-to scene too, which is safe because that runs off-lock and writes nothing to
`self`, so the old scene keeps serving at full rate throughout.

**The accounting bug found while verifying it.** Reported footprint said 1.64 GiB; actual
RSS was 2.36 GiB. **PIL stores an RGB image as 4-byte-aligned RGBX**, so a decoded pixel
costs 4 bytes plus object overhead — measured **4.22 B/px** holding 40 real 3940x414
templates (275.2 MB against the 195.7 MB that `w*h*3` predicts). Every `w*h*3` in the cache
path was therefore a third low.

That is worse than a cosmetic mis-report, because `plan_template_cache` **sized the cache**
from it. Its docstring guarantees it under-promises ("under-promise, which is the right
direction to be wrong in"); with `w*h*3` it over-promised by 41%, planning a cache needing
more RAM than it had budgeted. Now `_DECODED_BYTES_PER_PX = 4.25` (measured 4.22, rounded
up so the error stays safe). Reported 2.32 GiB against a measured 2.30 GiB RSS.

**Retroactive correction to figures quoted throughout these docs:** the pre-crop cache was
**~20.4 GiB, not 14.4**, and a full-window template was **58 MiB, not 41**. The cropped
sweep is **2.32 GiB, not 1.76**. The *ratio* is unchanged at 8.8x, since it is the same
pixel ratio — only the absolute numbers were wrong, and they were wrong before this change
as well. Historical entries below still quote the old figures; they are not being rewritten,
but they are low by a third wherever they name decoded bytes.
`tests/test_frame_library.py::test_decoded_bytes_per_px_matches_what_pil_actually_allocates`
measures it against the allocator rather than asserting it, because believing an arithmetic
model over the allocator is exactly what went wrong.

### 2026-08-14 (later) — voltron measured: the crop lands, and what is left is memory, not CPU

The projection in the entry below said ~57 ms / 17.6 fps cold on voltron. **Measured: 71.4
ms / 14.01 fps cold, 37.7 ms / 26.53 fps warm, on 2.32 GiB instead of 20.4 (both
figures corrected in the entry above; the 8.8x ratio is unchanged).** voltron now
clears the 10 fps goal on a COLD slew, which it missed by 3x before; the condition
"usable there only with `--template-cache auto`" is retired.

Two things the run taught that the projection could not.

**Run it twice after a push.** The first run read 94.3 ms with p10 71.7 / p90 130.0 — a
1.8x spread. The second read 71.4 ms with p10 67.7 / p90 77.3, a 1.14x spread, p10 barely
moved. That is first-touch I/O off the shared ZFS pool on files rsynced minutes earlier:
~23 ms a frame, paid exactly once. It is also the first direct evidence of what the E4
experiment flagged as unmeasurable from here — cold I/O on that pool is real and large,
which is another reason the raw-uncompressed-on-disk idea stayed on the shelf.

**The remaining gap is DRAM, and the stage split cannot see it.** The four stages sum to
52.8 ms against a 71.4 ms slew. The split re-renders a few angles back to back so each
stage reads what the last left in cache; a real slew streams a different 4.89 MB template
from DRAM every frame. Measured on the dev box: holding ONE template resident costs 10.3
ms/frame, TWO costs 10.4, FOUR costs 14.1, and it is then flat out to 1762 MB. That step
is the L3 boundary (30 MB on the E5-2650 v4, ~33 on the dev box). It is the same effect
that makes `pan` (25.6 ms) beat `slew_warm` (37.7 ms) on voltron when neither decodes —
and that 12 ms gap is only 2.9 ms on the newer box, i.e. it scales with the memory
subsystem (DDR4-2400 era) rather than with the CPU ratio of ~2.4x.

**Consequence for what to reach for next.** The serve path is now memory-bound on that
host, not decode-bound. That reframes two shelved items: an LoD tier would cut the
streamed template from 4.89 MB to 1.22 MB at zoom <= 1, which is now a bandwidth argument
rather than only a decode one; and the 2026-08-11 refutation of float32 in the camera
stage ("index-bound, not bandwidth-bound") was measured on the DEV box and may not hold on
voltron, where the same stage is 2.4x slower against a 2.4x CPU ratio but sits behind a
much weaker memory system. Neither is worth doing on this evidence alone — but neither is
closed by the old measurement either.

`bench_serve` now prints the stage-sum-vs-slew gap and names it, rather than leaving a 26%
discrepancy to read as noise.

### 2026-08-14 — templates store their content, not the window; and the viewer delivers colour

**The problem was footprint, not framerate.** voltron cleared the 10 fps goal
only via `--template-cache auto`, which held the whole decoded sweep in
**14.4 GiB**. That is a great deal of memory for a program that replays PNGs,
and it fails ungracefully: LRU against a cyclic sweep evicts each frame just
before it comes round again, so a cache short of one revolution is worth
**zero** rather than a proportional share — the 16 GB dev box got nothing at all.

**The measurement that decided it.** A template's content occupies **10.4%** of
its frame, and for both hampton libraries it is the *same rectangle at all 360
angles* (the spindle axis is the pin axis; `mitegen_200um` varies, 231 distinct
boxes). Templates are that big only because `plan_window()` unions the measured
content with the **centred field of view** plus `pan_mm` — the FOV term, not the
scene. Storing only the content and filling the rest at read time is **exact,
not approximate**: templates hold raw transmittance, every ray is born at 1.0,
and the photographic look is applied downstream at serve time.

| | before | after |
|---|---|---|
| `hampton_300um_realistic` on disk | 28.9 MB | **12.9 MB** |
| decoded, whole library | 15.48 GB | **1.76 GB** |
| dev-box slew | 87.3 ms / 11.5 fps | **23.0 ms / 43.6 fps** |
| voltron, projected, **no cache** | 279 ms / 3.6 fps | **58.7 ms / 17.0 fps** |
| voltron, projected, cached | 67.3 / 14.9 fps @ 14.4 GiB | **27.7 / 36.1 fps @ 1.76 GiB** |

So the crop alone beats what 14.4 GiB used to buy, with no cache at all — and
the cache is now cheap enough to be the default, including on the 16 GB box
where it previously thrashed to nothing.

**Four things this deliberately does NOT do, each because it was measured.**

1. **The render window is unchanged, and no library was rebuilt.** Rendering the
   tight window instead of the full one buys **1.08x on 8.79x fewer pixels**
   (79.58 → 73.54 s/frame, four poses each, RTX 4080S, n_cond 7). The AABB cull
   already made background rays nearly free, so the empty field was costing ~6 s
   of 79.6. `recrop_library` migrates an existing sweep in ~2.5 min with no GPU,
   and the surviving pixels are bit-identical. **Do not change `plan_window` for
   this; there is nothing there.**
2. **`manifest["rendered"]` stays the VIRTUAL window.** `pose_crop`,
   `zoom_limits`, `servable_pose` and `pin_projection.template_mapper` all key
   off it, so keeping it virtual meant none of them changed and the served
   geometry is bit-for-bit what it was. Where a frame's pixels sit inside that
   window rides on the frame record as `content_origin_px` / `content_size_px`.
   Absence of those fields means "the stored image IS the window", which is what
   keeps every pre-crop library working.
3. **No `_BUILD_KEYS` entry was added.** A build key is only an operator banner —
   a stale library still serves — and adding one would have marked all three
   tracked libraries stale and started ~2 h rebuilds. The real interlock is
   `_frames_complete`, the one gate that returns `"missing"` and genuinely
   refuses: it now compares the file against the DECLARED stored size, which
   catches cropped-frames-read-as-window and window-frames-declared-as-cropped
   symmetrically, for free.
4. **The crop is derived from the rendered pixels, not from `content_window()`.**
   That function scouts 8 angles at 320x240 with `n_cond=1` and a PSF below its
   own minimum sigma — 32x coarser than a template pixel. It under-measures,
   which is invisible where it is used (`plan_window` unions it with a much
   larger field) and would clip real sample here.

**The camera stage came along for the ride.** `field.to_sensor`'s 640→704
resample was 6.7 ms of a frame as a float64 fancy-indexed gather; PIL does the
identical resample on the uint8 the template path already holds in 1.3 ms. Both
place output centre i at `(i+0.5)*n_src/n_dst - 0.5`, so this is the same maths
in a faster loop. It is applied **only on the template path**, after the defocus
blur exactly as before — so the blur still happens in square-pixel space, and
`field.to_sensor` is untouched and still serves the live render path.

An earlier version of this also special-cased the outermost columns, on the
theory that `_axis_weights` clips its sample coordinate where PIL clamps filter
support. **Measured: unnecessary** — with the fix removed the two agree to 1
level at every column. It was deleted rather than kept as insurance.

**`--mono` now defaults to `off`.** The simulator is a colour instrument and
scenes are allowed to be coloured; defaulting to a delivery-stage flatten meant
no coloured scene could ever be seen, and hid a scene bug rather than paying it
down. It costs nothing — `apply_camera` is *faster* without the luma matmul and
channel repeat (3.04 vs 4.35 ms) — and on the shipped libraries it changes at
most **20/21/46 levels on 0.003–0.42% of pixels**. What it exposes is real: a
material's `colour` is an ABSORPTION spectrum, so `crystal: [0.7,0.9,1.0]`
renders blue. **The repair is a scene change** (`colour: [1,1,1]` with the
absorption in `mu_optical`) and scene files are inside `render_sha`, so it
rebuilds all three libraries — deliberately left for a later pass.

**Acceptance.** The old path (full window + `to_sensor`) and the new one (crop +
PIL stretch) were compared pre-JPEG across every servable zoom, nine angles and
three pan offsets: **max 1 level, zero pixels differing by more than 1, out of
64 million compared.** The delivered JPEG can differ by more at a hard edge,
because a lossy codec reacts non-linearly to a one-LSB change spread across a
block — that is a property of comparing two encodings, not of the resample, and
it is why the gate is applied before the encoder rather than after.

### 2026-08-13 — voltron measured on both halves: build there is a wash, serve there needs the cache

**Why this is a decision and not just a benchmark.** "Deploy loop-sim to voltron"
was one phrase covering two unrelated questions — can it BUILD libraries, and can
it SERVE them — and the answers point in different directions, so treating them as
one would get the deployment wrong either way.

**Rendering: the TITAN V is 4% faster than the dev box, and the reason matters
more than the number.** 74.55 s/frame mean against 77.9, on `hampton_300um_realistic`
at supersample 4 with the same 1,000,000 tile on both. A full build is ~7.45 h
against 7.48 — no reason to move builds there for speed, and a positive reason not
to: it is a shared node, so a 7.5 h job competes with real work.

**The double-precision hypothesis is refuted.** The tracer is deliberately float64
(the correctness fix), and a TITAN V runs FP64 at 1:2 of FP32 where consumer Ada
runs 1:64 — roughly 8x on paper. It bought 4%. Critically the card was at **100%
utilisation** throughout, so this is not the CPU-dispatch-bound regime the 640x480
tube scene showed at ~29% GPU busy: the mesh render is genuinely GPU-bound and an
8x FP64 advantage still did nothing, which points at bandwidth or occupancy (652
vs 736 GB/s, favouring the newer card) rather than double-precision ALU. **Do not
reach for an FP64-strong card to accelerate this workload.**

**Serving is where the deployment decision actually lives.** Cold, voltron slews at
265.5 ms / 3.77 fps — a third of the 10 fps goal, on the machine most likely to host
the viewer. With `--template-cache auto` it is 67.25 ms / 14.87 fps. The proof that
the mechanism is doing what it claims: the warm slew and the pan agree to **0.05 ms**
(67.25 vs 67.30), so a rotating frame costs exactly what a translating one costs and
the decode is eliminated rather than merely reduced. Predicted 71.4 ms from the stage
split before running it; measured 67.25.

**So the deployment conditions are:**

- The viewer on voltron **requires `--template-cache auto`** (14.4 GiB of its 251 GB).
  Without the flag it runs at a third of the goal and looks merely sluggish rather
  than misconfigured.
- Library builds gain nothing there and cost a shared node hours. Keep them on a
  workstation unless there is a reason beyond speed.
- The VRAM margin for a build is thin: the preflight measured a **7.55 GB budget
  against a known ~7.3 GB build peak**, ~250 MB of headroom. It held across both
  heavy poses, but only four of 360 were sampled — a full build should be watched,
  and `--vram-fraction` lowered before `--supersample` if it OOMs.

**A method note worth keeping.** Both figures above were wrong the first time for
configuration reasons rather than measurement error: the serve benchmark warmed on
the poses it was about to time (reporting a slew at 78 ms whose p10 was, exactly, the
pan number) and then benched `mono=off` where the server defaults on. A benchmark is
only as good as its agreement with the thing it measures, and that agreement is worth
asserting rather than assuming — `bench_serve.py`'s output is now verified
byte-identical to `CameraServer._render_frame()` at the same pose.

### 2026-08-12 — the glint is projected from the scene; the silhouette fit is deleted

**The accepted limitation was hiding a real bug.** On 2026-08-10 the glint's
silhouette inference was booked as "a small incorrectness": it vanished when the
pin's side left the frame, and followed the tip's curve on a tip-only view.
Both were true. What nobody had looked for is what happens when the pin is not
in the frame **at all**. `_streak_patch` thresholded every dark pixel, eroded
anything under 13 px and fitted a bar to whatever survived — with **no
connected-component step anywhere in the file**, so `_fit_one_orientation` took
first-and-last opaque row per column across the *whole image* and two separate
bodies read as one. `specular_streak`'s own docstring had said so since it was
written: "ONE body is assumed."

The erosion was the only discriminator, and it does not survive zoom. At 1x the
loop fiber is ~2.7 px and dies; at 4x the loop-plus-droplet is ~279 x 145 px and
lives. Measured on the shipped `hampton_300um_realistic` library, fraction of
streak PIXELS off the pin (metal starts at lab x = 1.000 mm):

| zoom | φ=0 | φ=15 | φ=45 | φ=90 | φ=180 |
|---|---|---|---|---|---|
| 1.0 | 0.0% | 9.2% | 6.0% | 0.0% | 0.0% |
| 1.5 | 18.8% | 19.4% | 13.3% | 5.1% | 18.3% |
| 2.0 | 43.4% | 43.9% | 33.2% | 22.9% | 42.6% |
| 2.5 | 100% | — | — | — | — |
| 4.0 | 100% | 100% | 100% | 100% | 100% |

At zoom ≥ 2.5 the whole glint was on the droplet at full strength (peak
0.46–0.47, the same as a correct one) at every angle from 0 to 315. **It was
never zoom-only**: at 1x the fit merged pin and drop into one body and `half_l`
jumped 100.5 → 209.5 px.

**A SECOND, INDEPENDENT DEFECT, and it is the one to remember.** The ridge was
masked by `opaque[r0:r1, c0:c1]` — the *global* threshold over the bounding box
of *everything* dark — not by the body that had been fitted. So even a correctly
fitted pin sprayed its ridge across every dark pixel the band crossed. The
comment claiming it "cannot leak onto the background or onto the loop" was half
wrong: not the background, but the loop and the drop freely. Both halves had to
be fixed; either alone leaves a leak.

**No image-only rule separates a pin from a droplet.** Three were measured.
Aspect was already recorded as a failure, and the loop-plus-drop reads 1.92
against `min_aspect` 1.8 — it passed by 6%, which is luck. Bar-likeness was
already recorded as a failure. **Solidity** (opaque fill inside the fitted bar)
was new and fails in the *wrong direction*: a pin view contaminated by the drop
reads 0.508 against a pure droplet's 0.582, so the gate kills correct glints
first. That is what makes this a projection problem rather than a tuning one.

**The fix is the one `_pin_axis`'s docstring had prescribed since August.**
`renderer/pin_projection.py` finds the object the code declares shiny
(`SHINY = {("pin", "metal")}`), clips its cylinder by the half-spaces that bevel
it, maps the axis through `gonio.transform()`, projects orthographically onto
`camera_fast`/`camera_slow`, and converts to delivered pixels. ~150 lines of
heuristic out, ~90 of geometry in, and the mask and erosion that were most of
the stage's 3.4 ms/frame go with them.

**Verified against the silhouette it replaces**, on the shipped library —
predicted vs measured centre row, half-width and start column:

```
  zoom 1.0    240.4 / 47.3 / 501.1   vs   240.0 / 46.5 / 508
  zoom 1.5    240.3 / 70.9 / 575.3   vs   240.0 / 70.5 / 582
  zoom 2.0    240.2 / 94.6 / 649.6   vs   240.0 / 93.3 / 656
  zoom 4.0    240.0 / 189.2 / 589.8  vs   240.0 / 187.5 / 596   (tx = -0.6)
```

Row to ≤ 0.4 px, half-width to ≤ 1.7 px, and the start column short by 6.2–6.9
px — **exactly the (k-1)/2 = 6 px the erosion took off**. The projection is the
accurate one; the silhouette was the approximation.

**Three constraints shaped where the code could go**, and all three are about
not triggering a rebuild. The declaration of which body shines had to be in
CODE, because `scene_sha256` is a build key and a `specular:` flag in the YAMLs
would invalidate all three libraries. The new module had to go in `renderer/`,
because `_RENDER_SOURCES` globs `scene/*.py` and names `motors/goniometer.py` —
a helper in either would have cost the same rebuild. And `to_sensor` runs before
`apply_camera`, so the projection lands on the 704x480 grid and the columns
carry the 704/640 scale. `render_sha` is unchanged and all three libraries still
read `current`.

**Two guards survive, with exact inputs instead of measured ones.** A shank
seen end-on has no side to run a ridge along, and one wider than 0.90 of the
frame's short side leaves the ridge's position undefined — `offset` and `width`
are both fractions of the pin's width. Together they are what keeps
`mitegen_200um` off at all 24 angles: its pin is `axis [0,0,1]`, the beam axis,
so it is end-on at φ=0 and 500 px wide against a 480-row frame everywhere else.
It is now DECLARED not to shine rather than guessed at from its shape.

**The clip margin is not symmetric, and getting that wrong is invisible.** The
ridge sits off the axis by up to `half_w`, so the frame is inflated by that much
when clipping the axis segment — but **only perpendicular to the axis**.
Inflating along it too kept the pin "in frame" for a third of a millimetre after
it had left, which held `hampton_300um_realistic` alive at zoom 2.5 and 3.0
where the crop ends 0.2 mm short of the metal, and would have re-created the bug
in a smaller form.

**What this changes that an operator will notice.** The two limitations accepted
on 2026-08-10 are closed: the glint no longer vanishes when the pin's side
leaves the frame, and on a tip-only view it follows the shank rather than the
tip's curve. Frames at high zoom now show a glint where they used to show none —
and, at zoom ≥ 2.5 with no pan, no glint where they used to show a wrong one.

**Guarded by** `test_streak_never_lands_on_the_droplet` (48 real poses, zero
off-pin pixels — the assertion that failed at 100% before),
`test_projected_pin_matches_the_rendered_silhouette` (architecture-independent:
projection against the render, not against another computation on the same box),
`test_streak_stays_off_a_second_dark_body` (the mask half),
`test_streak_refuses_mitegen_at_every_angle` (re-pointed at the new path), and
`test_render_sha_covers_the_tracers_and_not_the_delivery_stage`, extended to
assert `renderer/pin_projection.py` stays out of the hash.

### 2026-08-11 (latest) — the VRAM budget is enforced, not assumed

**Why this exists as a decision rather than a fix.** The project is handed to a
team who will re-render scenes on a 12 GB TITAN V and must not have to think
about memory. "It worked on the 16 GB dev box" is not a guarantee, and twice in
one day it was actively wrong: a survivor-chunk floor that silently demanded
16.7 GB, and `fit_tile_size` spending an afternoon as `min(total_rays, 1e6)`
without ever consulting the card.

**The mechanism, in four parts.**

- `memory_budget()` is the single authority. It derives from `mem_get_info`'s
  **free**, not total, because voltron is a shared 8-GPU node -- a neighbour's
  allocation must reduce ours rather than surface as an OOM at frame 300 of
  360. One GiB is held back for the CUDA context and allocator slack, because
  `nvidia-smi` routinely reads ~1 GB above torch's own `max_memory_allocated`.
- `install_vram_ceiling()` makes the budget a **hard allocator limit**. Without
  it the budget is only advice, and on WSL2 an overrun spills to host RAM: a
  10-50x slowdown that looks like a hang rather than a failure. With it,
  overruns are catchable `OutOfMemoryError`s.
- `fit_tile_size()` consults the card again.
- `check_render_fits()` renders ONE frame and reads the real peak before a
  build commits.

**Why the preflight measures rather than predicts.** Peak memory here is a sum
of mesh temporaries, resident ray arrays and O(W x H) buffers that no tile
shrinks, and the measured points do not fit a clean linear model -- a formula
would be a guess wearing a safety factor. One frame costs seconds against a
build that costs hours, and it is the only thing that can catch a term nobody
thought of.

**Refuse, do not downgrade.** A too-large request fails with the largest
`--supersample` that would fit rather than quietly building something smaller.
A library that differs from what was asked for would pass every staleness check
in the system, so the failure would be invisible in exactly the place the
project keeps its integrity.

**Deliberately NOT done: a bigger trace tile.** Measured at 14.4 Mpx, tiles of
1M / 2M / 4M / 6M rays run 18.6 / 17.5 / 17.3 / 17.1 s, all byte-identical. Six
times the tile buys **8%** and costs 1.8 GB of peak. The 1M default stays: on a
12 GB card shared with other tenants that memory is worth more than 8%. The
curve is steep in the other direction (~130 tiny passes cost ~10x), which is
why the preflight scales the tile *proportionally* on a miss instead of
dropping to its floor -- overshooting downward converts a memory problem into a
speed problem.

**Testability is the guarantee.** `LOOPSIM_VRAM_BUDGET_GB` overrides the
measured budget, so 12 GB behaviour is asserted in the suite on whatever card
CI has. `set_per_process_memory_fraction` cannot do this on its own: it
constrains the allocator while `mem_get_info` keeps reporting the real device,
so the sizing never sees it. Both are used -- the env var to drive the sizing,
the fraction to enforce the ceiling.

**Two lessons worth keeping.** A test for a memory guard must never itself be
the allocation that breaks the machine: an early version proved "oversized gets
refused" by attempting a 40000x20000 render, whose accumulator alone is 17.9 GB,
and took the WSL2 VM down with it. Provoke the refusal by shrinking the BUDGET.
And a preflight that measures by rendering needs the ceiling installed BEFORE it
measures, or the measurement itself overshoots -- it spilled to 13.8 GB while
"checking" whether 12 GB was enough.

### 2026-08-11 (later still) — the mesh path never culled, and that was 43x

**`hampton_300um_realistic` builds in 11.2 minutes instead of 8.06 hours, and
every frame is byte-identical.** 80.6 -> 1.86 s/frame. Two changes, both in
`engine_torch.py`, neither of which touches a pixel.

**What was wrong.** `TSurfaceMesh.ray_intersect` brute-forced every ray against
every face. The numpy `SurfaceMesh` it is a port OF has always run an AABB slab
test and fed only survivors to Moller-Trumbore; `TTube` has the same cull
(`_aabb_survivors`). The torch mesh was the one class that diverged. On the
shipped droplet scene the mesh's AABB covers **0.284% of the render window**
(mean over the 360-frame sweep, measured from the scene: a 0.484 x 0.283 x
0.171 mm box in a 10.33 x 4.766 mm window), so **99.7% of rays were being tested
against 5472 triangles they could not possibly hit.**

**Why that cost more than the wasted arithmetic.** The 160 B/ray/face law made
`fit_tile_size` divide the frame down until `tile x faces x 160 B` fitted: 6800
rays, **133 passes per frame**. So the missing cull was also buying 133x the
per-pass overhead. Culling alone gave 4.05x (19.90 s); letting the tile return
to a single pass gave a further **10.2x**. The second half was the larger one.

**The fix is a parity restoration, so byte-exactness is provable, not hoped
for.** Every triangle point lies inside the vertex AABB, so a ray the slab test
rejects provably missed every face -- brute force returned INF for exactly those
rays. Verified three ways: a direct culled-vs-brute-force equality test
(`test_tile_sizing.py`), new CPU+CUDA render-parity cases on this scene, and a
**full 360-frame rebuild diffed against the shipped library -- 360/360
byte-identical, zero differing pixels**, with git confirming only
`manifest.json` changed.

**The budget moved rather than disappeared.** `TSurfaceMesh` now chunks its own
survivors (`_mesh_survivor_chunk`), so the mesh's working set is bounded where
the mesh is instead of by shrinking every caller's tile. `fit_tile_size` lost
its mesh term entirely and mesh scenes get the same flat default tube scenes
always had.

**One trap found by watching, not by testing.** Sizing that chunk as a fraction
of FREE VRAM took ~8 GB on an idle 16 GB card and drove a build to **13.7 GB** --
inside the WSL2 spill zone, on a GPU shared with a desktop. Capped absolutely at
2 GiB (`_MESH_CHUNK_MAX_BYTES`): the build then held **5.9 GB and ran no slower**
(2.01 vs 2.15 s/frame). A fraction of free memory is not a bound on a shared
card. Test-guarded.

**Measured, all at n_cond 7, f64:**

| scene | before | after | note |
|---|---|---|---|
| `hampton_300um_realistic` (5472 tris, build res) | 80.6 s | **1.86 s** | 43x; 8.06 h -> 11.2 min |
| `mitegen_200um` (234-tri ThinShell, build res) | 17.0 s | **3.00 s** | 5.7x |
| `hampton_300um` (no mesh) | 7.93 s | 7.97 s | unchanged, as intended |

**Two things this retires.** The `--tile-size 6800` incantation the RUNBOOK
required for WSL2 mesh builds is no longer needed -- the spill hazard it worked
around was a consequence of the missing cull. And the `TSurfaceMesh` AABB cull
stops being an open item that three separate DECISIONS entries deferred as
"blocking" then "an optimisation".

**What was NOT done, and why.** A per-face BVH or a CUDA port of the CPU's
uniform grid: dead by Amdahl once the cull lands (the mesh term is ~0.2 s of a
~20 s frame at the old tile, and less now), and the CPU grid is dead code
anyway -- `_grid` is written and never read, `_intersect_ray_triangles` has no
callers. df64 and rasterisation were investigated and rejected; both have
entries under "Already Tried".

### 2026-08-11 (later) — the NA fork resolves to 0.28, and the confound was the
### SPACE the comparison was made in, not the zoom stop

**Answer first: NA 0.28. At the hi stop, camera-space crystal/background is
0.677 against the photograph's 0.691 — a 2% gap. NA 0.10 gives 0.421, 39%
short.** Harnesses: `scratch/na_fork.py` (renders + measures) and
`scratch/d01_measure.py` (the photograph, with an auditable region overlay at
`scratch/d01_regions.png`). No beamline access, no capture, no library rebuild;
seven direct renders, ~4 min.

**The reference was re-measured first, and it holds.** Independent hand-placed
boxes on `D01` reproduce the recorded numbers: drop/bg **0.882** (recorded
0.871), crystal/bg **0.691** (0.696), crystal/solvent **0.783** (0.799). So the
target was never in doubt; only the render side of the comparison was.

**The fix that mattered was not the one the plan proposed.** The docs called for
rendering at the hi stop, on the reasoning that every drop photograph is hi mag
while the Hampton scenes model the mid stop. That is true and it was worth
doing — but it moves the answer by about **1%**:

| | crystal/bg, camera space |
|---|---|
| mid stop, NA 0.10 | 0.416 |
| hi stop, NA 0.10 | 0.421 |
| mid stop, NA 0.28 | 0.671 |
| hi stop, NA 0.28 | 0.677 |

What moved the answer by **2.3×** was the SPACE. Every previous comparison put a
render's **transmittance** ratio next to a photograph's **grey** ratio.
`field.apply_camera` is affine — `out = (e − B)·t + B` with a black floor
B = 0.1765 — so it does **not** preserve ratios; it lifts dark things hard. The
same NA 0.10 render reads 0.186 in transmittance and 0.416 in camera space. The
recorded 0.218-vs-0.696 gap was therefore roughly **half units and half
physics**, and the NA question looked more dramatic than it was.

**Which normalisation is like-for-like had to be measured, not assumed.** Both
D01 and the render take crystal at frame centre against background at the
edges, so a raw ratio carries whichever vignette each one has. `field.py`'s
modelled field is a vertical bowl at **41.6% peak-to-trough**, and its own
docstring warns the 2020 session it was fitted to is 5–7× stronger than every
other epoch. **D01 is one of the flat ones:** five sky boxes span **2.8%** of
level, and centre-column sky against corner sky is **−0.5%**. So the render must
be divided by its own clear-field level (removing a vignette D01 does not have)
and D01 needs no correction at all. Skipping that step reads NA 0.28 as 0.818
against 0.691 and would have pointed at ~0.17 instead — the same class of error
as comparing the two spaces.

The full grid, camera space, vignette-removed, against D01's 0.691 / 0.783:

| stop | NA | drop/bg | crystal/bg | crystal/solvent |
|---|---|---|---|---|
| real `D01` | — | 0.882 | **0.691** | **0.783** |
| mid | 0.10 | 0.894 | 0.416 | 0.465 |
| mid | 0.17 | 0.921 | 0.546 | 0.593 |
| mid | 0.28 | 0.963 | 0.671 | 0.697 |
| hi | 0.10 | 0.821 | 0.421 | 0.513 |
| hi | 0.17 | 0.923 | 0.556 | 0.603 |
| hi | 0.28 | 0.946 | **0.677** | **0.704** |

crystal/solvent comes in at 0.704 against 0.783, a 10% gap — but D01's own
solvent reads 0.667 / 0.700 / 0.832 across three interior boxes, so the
photograph's spread covers it. crystal/bg is the tighter constraint and it is
the one that lands.

**Two negatives worth not re-deriving.** (1) `template.yaml`'s 0.8233 µm and the
correct square-pixel hi stop of **0.9056 µm** give the same tone to 0.3%
(crystal/bg 0.679 vs 0.677). The pixel-size half of the old three-cameras puzzle
does not touch NA; it only ever mattered dimensionally. For the record the
square-pixel hi stop is built exactly like the mid stop's 7.4 µm —
704 × 0.8233 / 640 = 0.9056, covering the true field to 0.0% / 0.9% — so a
future hi-stop scene should use 0.9056, not `template.yaml`'s value. (2) A
luma-threshold crystal mask is sampling-dependent and cannot be used across
stops; all regions here are projected from the scene's own geometry (the
crystal's half-space box, the solvent mesh's extent), which is why the mid and
hi rows can be compared at all.

**What this costs, unchanged:** switching the scenes to NA 0.28 moves the
supersample ceiling and invalidates all three frame libraries, including the
8.06 h `hampton_300um_realistic` build. That is the owner's call and nothing
here has been changed — this entry is the evidence, not the switch.

**One thing the hi stop did surface, and it is new:** at 0.9 µm pixels the
solvent mesh's **tessellation is visible** — the drop's edge inside the loop
reads as a staircase of flat facets (5472 faces). It is invisible at the mid
stop. Any future hi-mag fidelity work needs a denser drop mesh, which also
makes the missing `TSurfaceMesh` AABB cull cost more than it does today.

### 2026-08-11 — the glint met an operator: four defects that only motion shows

Every one of these passed the test suite and looked right in a still frame.
They were found by turning the spindle and the zoom, which is worth recording
as a method: a camera model has properties that no single frame can expose.

**It sloped with phi.** The pin's silhouette was giving up its axis via second
moments, and `hampton_300um`'s 45-degree chisel sweeps up and down as the
spindle turns — so the fitted axis followed it, **-6.40 to +6.39 degrees,
sinusoidal in phi**, tilting the ridge 54.5 px across the pin. A horizontal pin
lit from a fixed direction shows a horizontal glint at every angle. The axis is
now fitted from the body's two long SIDES, across slices at least 90% of the
widest, which drops the tapering tip: worst tilt 0.42 degrees, measured
end-to-end slope <= 3.6 px.

**Its grain read as parallax.** It had been hashed in the pin's own frame so it
would ride with the pin — right for a static surface texture, wrong for this.
A machined shank is rough at the wavelength scale, so as it turns, different
micro-facets enter the specular condition and the glint TWINKLES. And the
anchor was the eroded mask's centroid, which drifts differently from the pin as
parts of it leave frame, so the pattern slid against the shank. It now re-rolls
per POSE: scintillation, and the anchor problem disappears. Determinism holds
because the phase comes from the pose rather than a clock — a held pose is
byte-identical, which the settle-parity guard requires, and any real move
re-rolls it.

**It vanished above ~1.5x zoom.** Past that the pin's in-frame piece is WIDER
THAN IT IS LONG (211 x 59 px at 2.77x), so the moments called it vertical and
the fit measured the shank's width as its length — half_w came out 17.8 px on a
187 px pin. Both orientations are now fitted and the consistent one wins,
judged on whether the body's sides run off frame and whether a clipped end is a
clean sever (hampton reads 1.00-1.05 at every zoom; mitegen 0.63).

**TWO THINGS THAT DO NOT WORK, recorded so they are not retried.** Separating a
zoomed-in pin from `mitegen_200um` (a mount wider than the frame) cannot be
done on **aspect** — the zoomed pin is 0.28 against mitegen's 0.85 — nor on
**bar-likeness**, the fraction of slices holding a constant width: 35% against
57%, so the pin is the LESS bar-like of the two. What separates them is that
mitegen's fitted body is 528 px wide in a 480 px frame. Nothing wider than the
frame's short side is a pin whose sides can be seen.

**The background was invisible, and the earlier measurement was wrong.** It had
been fitted to a residual of 2.6-2.9% of level, taking "background" to mean
brighter than the 60th percentile — which clips the dark half of every cloud
and biases the spread down. Masking by dilating the dark body instead gives
**3.38 / 3.48 / 3.92%**. Worse, the energy is spread across scale (3.0-3.8% at
16-64 px, 4.6-7.0% at 128-256) where it had all been put at one 200 px cell,
which reads as a smooth wash. Now six octaves of fBm from ~280 px to ~9 px at
gain **0.90** — not the textbook 0.5, which leaves sub-33 px detail at 0.39%
against the real ~3.2%. Two constraints the tests caught: the mottle must be
zero-mean AND renormalised or it moves the field's LEVEL rather than its shape
(6.7%, then 0.24% residual), and it lives on normalised coordinates so the
field still describes the same illumination at any render size.

**The glint now defocuses with the sample.** It is light off the pin's surface,
so it softens with everything else on that plane; it had stayed sharp because
the template crop blurs the silhouette and the glint was added afterwards.
Measured: grain sd falls 5.70 -> 0.26 (22x) from focus to 1 mm of depth, peak
54 -> 38 as the energy spreads. **Not done by swapping the stage order**, which
is the obvious fix and is wrong — that would also soften the illumination
field, whose finest octave is ~9 px, and the background is not imaged from the
sample plane.

**Where the frame time actually goes.** A rotating frame is 97.2 ms: template
decode **73.2 ms (75%)**, camera stage 15.8 ms, rest 8.2 ms. The socket
delivers 10-12 fps; a browser shows about half. The levers are a prefetch
decode pool (~30 fps, no rebuild) or `--supersample 2` (~24 fps, 47 min, zoom
ceiling 4x -> 2x). Neither taken.

**ACCEPTED LIMITATION — and it was hiding a real bug. SUPERSEDED 2026-08-12.**
All of the above still infers the pin from its
SILHOUETTE, so the glint stays a function of what is in frame: it disappears
when the pin's side leaves the frame, and follows the tip's curve on a tip-only
view. Judged a small incorrectness and accepted. The fix is not a better
inference — it is to take the pin's axis and radius from the SCENE, which the
server already knows, deleting ~150 lines of heuristic for ~40 of projection.

**What that judgement missed** is what happens when the pin is not in frame at
ALL. The fit has no notion of a BODY, so the loop-plus-droplet took the glint
instead — 100% of it at zoom ≥ 2.5, at every angle. The scene-driven projection
was built on 2026-08-12 and everything above is now history; see §2026-08-12.

### 2026-08-10 — the renders became photographs: camera emulation, the sensor
### raster, the pin's glint, and the scene fixes that needed a rebuild

The renderer had never been compared against a **photograph**, because no
reference set existed. One now does (`real_images/`, 44 frames). Side by side
the renders did not look like the camera, and the gap was not what anyone
named first.

**The dominant gap was TONE, not the background.** Measured on the delivered
hampton frame: **84.6% of pixels exactly 255, 14.4% exactly 0, 1.1% anything
else.** Real frames carry **19–27% genuinely intermediate tone** (C07 27.4%,
D03 19.0%), an empty field at ~0.65 of full scale and an opaque pin at ~0.18 —
neither rail ever reached. The render was a binary silhouette and the
photograph is continuous-tone; compositing a mottled background under a binary
cut-out reads *more* uncanny, not less. So tone landed first, as

```
observed = (E(x,y) − B) · T(x,y) + B
```

a lerp between the only two anchors actually measured. Both rails become
unreachable **by construction** — "40–226, nothing clipped" for free, with no
clamp and no tone curve. `E` is an analytic vignette (6 quadratic
coefficients, 83.5% of the field's variance), **not** a captured image.

**Why not a captured field: the between-session control falsified it.** Within
one session the background correlates at r = 0.93–1.00 across four spindle
angles and a sample translation, which is what suggested capturing it. Against
other epochs it correlates 0.11 / −0.31 / −0.18 / +0.35 (2005 / 2021 / 2025 /
2026), and the 2020 amplitude (sd/median 0.090) is **5–7× larger** than every
other epoch measured. There is no permanent pattern to capture. A plane
explains under 1% of the field (the linear terms cancel by symmetry, which is
why an earlier gradient fit looked like it failed); a 6-coefficient quadratic
explains 74.6%. Six floats replace a stored image, a schema key and a rebuild.

**Placement is the decision that made everything else cheap.** The whole camera
stage lives in `loop_sim/renderer/field.py` and runs at **serve time**, called
only from `encode_frame` — downstream of `pose_crop`, inside neither tracer.
That single choice is why `content_window`, `test_torch_render_parity`, the
crop-match tests and "must not pan with the sample" all never fire, and why
**no library rebuilt**. Templates keep storing raw transmittance. It is also
why `field.py` is deliberately excluded from `render_sha` below.

**640 vs 704 was settled, and 640 was right.** The open worry was that the
shipped 640-wide render was a silent 10% horizontal scale error against
704-wide photographs. It is not: the BL831 pixels are **1.110 non-square**, so
640 × 7.4 µm covers 4736.0 µm where the real 704 × 6.7324 covers 4739.6, and
480 × 7.4 covers 3552.0 against 3587.0 — **0.08% and 0.98%**. 704/640 = 1.100
cancels the pixel aspect. The feared error is real but points the other way:
rendering 704 wide at 7.4 µm would over-cover by **+9.92%**. `template.yaml`
(hi stop, 704 at the horizontal pitch) is 9.91% short vertically and is the one
place a genuine 10% error lives; nothing uses the file.

What 640 does not reproduce is the frame SHAPE, and dcss stores a µm-per-pixel
constant for this camera — so a stand-in emitting 640 columns reads 10% wide.
`field.to_sensor` resamples to 704×480 in the same camera-space stage, **after**
the defocus blur (the optical PSF is isotropic; it is the SENSOR that samples
at two pitches, so blurring in square-pixel space and resampling after
reproduces that for free) and **before** the field (so illumination is evaluated
on the delivered grid, and pixel-scale terms land in true camera pixels).
Measured on a served frame: pin still 95 px tall (703.0 µm, the documented
dimensional check, unmoved because the resample is horizontal), implied
horizontal pitch 6.7273 µm/px against the real 6.7324.

**The pin's specular streak was measured before it was written, and the plan's
spec for it was wrong twice.** The plan carried "specular peak 219 = 1.25×
local bg" from C07 — that is the frame MAXIMUM, and it is **three pixels**
(0.001% of the frame), not a streak. Painting the pin at 1.25× background
would have been badly wrong. And its "grain sd 0.6–2.5 levels" is the pin
BODY (measured 0.50 and 1.30); the ridge is ~10× grainier, because surface
slope modulates what is *reflected*, not what is absorbed. What the two frames
that clearly show a pin actually say:

| | A01 | E02 | shipped default |
|---|---|---|---|
| ridge centre, half-widths off axis | +0.48 | −0.42 | −0.45 |
| ridge FWHM, fraction of pin width | 0.150 | 0.112 | 0.13 |
| peak above the floor, × background | 0.82 | 0.15 | 0.35 × the field |
| grain sd on the ridge | 6.6 lv | 10.0 lv | 5.5 lv |
| grain correlation length | 2 px | 2 px | 2 px |

Served hampton frame: peak at f = 0.274, 2.40× the pin floor and 0.89×
background — between the two references on every axis. The sign of the offset
is an illumination property, not a pin property, hence signed and defaulting to
E02. Geometry comes from the IMAGE (second moments of the eroded opaque mask),
never the scene, so the glint tracks the pin through any pose without this
stage seeing the goniometer. **(REVERSED 2026-08-12: geometry now comes from
the SCENE. Inferring it from the image is what painted the glint on the
droplet — see §2026-08-12.)** Grain is value noise hashed on the **pin's own
frame** — roughness belongs to the pin, so it rides with it rather than
crawling across the shank as the stage pans, which would also be a
localisation shortcut for anything trained on these frames.

**Three defects in it were found by driving it, not by writing it**, and each
is a rule rather than a patch: `mitegen_200um` blinked the glint on and off six
times a revolution (1 µm pixels put a 0.7 mm pin wider than the frame, so its
moment aspect wanders 1.1–2.2 and any bare threshold cuts through it) → a body
whose **side** the frame cuts has no measurable width, and half_w sets both the
ridge's position and its FWHM, so it draws nothing; the end taper faded the
streak over the last 8 px of every hampton frame → an end the **frame** cut is
not an end; and a compact blob has no long axis worth finding → `min_aspect`
1.8, chosen because a pin only ever enters from one side (2.87 on hampton, 2.0
on E02, against 4.1 on A01).

Cost 3.4 ms/frame after two optimisations worth recording: the box erosion is a
**doubling shift-and** (0.15 ms against 1.6 ms for an integral image and ~8 ms
for a minimum filter, byte-identical to both), and the ridge is evaluated only
inside its own 3.5σ band and scattered into the pin's ~6% of the frame rather
than added frame-wide. `acceptance_voltron.py` times `render_torch` and never
reaches this stage, so the 11.9 fps TITAN V figure is untouched.

**`render_sha` closes the last staleness hole.** `scene_sha256` catches a
changed scene and `_BUILD_KEYS` catches changed settings, but a RENDERER edit
left every manifest reading `current` while the frames had been traced by code
that no longer existed. Hashed: `renderer/microscope.py`,
`renderer/engine_torch.py`, `renderer/optics.py`, `scene/*.py`,
`motors/goniometer.py`. **Deliberately not hashed:** `renderer/field.py`
(serve-time, never enters a template — being able to change it without a
rebuild is the entire reason it was placed there), `renderer/beam.py` (X-ray),
and `library/` and `server/` (delivery — `pose_crop` lives in `library/`, so
hashing it would invalidate every library for a change to how frames are
*cropped*). Getting the set wrong is silent both ways, so
`render_source_paths` is public and a test asserts both halves of it. The three
shipped manifests were stamped, which is honest: `git log` shows no change to
any hashed file since the oldest of the three builds. Without the stamp all
three would grade stale and the LAUNCH path rebuilds a stale library before it
binds the socket — 47 min / 102 min / 9.5 h. That launch-path rebuild is now
reachable by editing the renderer, where before only `mitegen_200um`'s
supersample got there; it is the already-open "should launching on a stale
library behave like switching to one?" question, unchanged but much easier to
hit.

### The scene changes, and the NA evidence they produced

Three changes to `hampton_300um_realistic.yaml` only —`hampton_300um.yaml` is
untouched, because every fps number and the 11.9 fps acceptance figure are
measured on it.

**Crystal `[0.7,0.9,1.0]`/0.02 → `[1,1,1]`/4.09.** Colour is an absorption
spectrum, so the old value was (9.02, 3.02, 0.02)/mm and rendered the crystal
strongly blue against neutral reference frames. Deriving the replacement needs
one non-obvious step: the crystal refracts (n 1.52) and light bent past the NA
gate darkens it whether or not anything absorbs. The BLUE channel measures that
floor directly — its mu is 0.02/mm, essentially nothing — at T = 0.8686. Divide
it out and red and green finally agree on the path length (0.0809 / 0.0832 mm,
2.8% apart) where inverting them raw does not. Control render (old drop,
neutral crystal): luma 0.6170 against the 0.6207 it replaced. **Skipping the
floor gives 3.56 and a crystal three times too dark** — that was the first
value tried, and the control is what caught it.

**Pin bevel 45° → 0.** The reference photos settle it: the pins in A01 and E02
end square, not chiselled. `pin_geometry.py` still models a scored-and-snapped
tube, so `--pin-bevel 45` restores the old tip.

**Drop 0.002 → 0.00893 mm³**, half-thickness 20.7 → 85.6 µm. Clears all three
validator warnings: aspect 8.23:1 → 1.99:1, rim deflection sin 0.081 → 0.273
against NA 0.10 (so the drop draws a real rim instead of collecting whole), and
the crystal no longer pokes out of the solvent. Edge-on the drop goes 111 →
170 µm, which is the gap the change exists to close.

**The dissent this plan recorded is VOID — both sides had the wrong premise.**
It turned on the drop core being 0.72× background. Measured with the drop
located from its own mesh it is **0.94–0.99×**: the 0.72× came from sampling a
0.15 mm disc **about the origin**, and the drop has sat at x = −0.262 mm since
the 2026-08-07 placement fix. The drop was already at or above the 0.93×
target, so thickening it moves toward that target, not away.

**And the volume change produced the NA evidence the plan asked for.** The drop
stays near-background bright at every NA. What goes dark is the CRYSTAL, once
an 85.6 µm drop immerses a body that used to poke out of a 20.7 µm one:

| | drop/bg | crystal/bg | crystal/solvent |
|---|---|---|---|
| real `D01` (hi mag) | 0.871 | **0.696** | 0.799 |
| new drop, NA 0.10 | 0.989 | **0.218** | 0.220 |
| new drop, NA 0.17 | 0.955 | 0.317 | 0.332 |
| new drop, NA 0.28 | 0.957 | **0.455** | 0.476 |

Per the plan that is **evidence for the NA fork and not a reason to revert the
volume**, and the crystal's mu is deliberately NOT tuned to compensate — doing
so would bury the signal in an absorption coefficient. Two things sharpen it:
the reference frame is **hi mag**, whose own calibration (`template.yaml`) is
NA 0.28; and the drop's own rim deflection lands at **0.273**, which is NA 0.28
almost exactly. **The library rebuild is held** for this reason: 9.5 h that a
switch to NA 0.28 would immediately invalidate.

### 2026-08-07 (later) — the black droplet was two scene-side mechanisms; the
### solver is replaced by the closed form and scenes are validated mesh-back

**This supersedes "recorded, not repaired" below** — the prerequisite that
entry demanded (a validation check measuring volume and rim radius back off
the generated mesh) now exists, so the repair happened the same day. Decided
by a four-member council session (Architect/Skeptic/Pragmatist/Researcher,
independently peer-reviewed); the load-bearing findings and choices:

**Why droplets rendered black — two independent mechanisms, both measured,
neither a renderer bug.** (1) **Material `color` is an absorption spectrum,
not a tint**: `mu_per_ch = mu_optical + 30·(1−color)` per mm
(`microscope.py`, `_COLOR_MU`). The generator's solvent color `[0.2,0.4,0.8]`
therefore absorbed at (24,18,6)/mm despite `mu_optical: 0` — a hard
brightness ceiling of ~0.26 through the drop, which is exactly why the
2026-07-28 NA sweep saturated at 0.2633 instead of approaching 1.0 (a purely
geometric gate opened to NA 0.90 must pass nearly everything). Falsification
measurement, changing only the color field to `[0.97,0.98,1.0]`
(`template.yaml`'s own solvent value): NA 0.90 drop-core 0.2633 → 0.6436;
the baseline reproduced the HANDOFF sweep to four decimals first. Corollary:
**every NA experiment run before the color fix was uninterpretable** — the
ceiling capped them all. (2) **The fallback hemisphere was the worst possible
lens**: flat-bottomed plano-convex, f = R/(n−1) = 441 µm, so at NA 0.10 only
the inner 29% of radius (9% of area) clears the collection gate — the
documented 29%/92% figures reproduce from paraxial algebra. The correct
biconvex lens is ~3× flatter (ρ ≈ 0.42 mm at 2 nL) and passes most of the
aperture. With both fixed, the regenerated scene's drop core measures
**0.67 at NA 0.10** against a 0.93 background — near-background-bright with
a dark rim, the real bright-field appearance.

**Exonerations, recorded so nobody chases them:** `MAX_DEPTH` exhaustion
returns the accumulated partial product and never sees the NA test — it errs
BRIGHT and cannot cause blackness (the HANDOFF open question had it
backwards). TIR-as-total-absorption is desirable: it is what draws the dark
rim. The binary NA gate is correct behaviour for this imaging model and was
not touched; nothing in `loop_sim/renderer/` changed.

**The color fix is data-side only (owner's call):** generator solvent color
is now near-white in `hampton_loops.py`/`mitegen_mounts.py`, with a comment
at each site naming the trap. Splitting `color` from a per-channel
`mu_rgb` in both engines was considered (a backward-compatible opt-in field)
and deliberately not done — `microscope.py` is frozen ground truth; the
overload is now documented instead. Revisit only if a scene needs a colored
but weakly-absorbing material.

**The Bashforth-Adams ODE was deleted, not fixed.** Two independent reasons.
(a) Its specific defect: `_ba_rhs` mixed angle conventions — `dr/dz = tan ψ`
is ψ-from-vertical, but the azimuthal curvature was coded `sin(ψ)/r`, the
ψ-from-horizontal form (should be `cos(ψ)/r` in that convention). At the apex
it produced κ₂ = 0 instead of 1/r, so the profile flattened and turned over —
which is precisely the measured "peaks at 78 µm mid-bracket, falls off both
sides" signature (a correct CMC profile is monotone and cannot have an
interior maximum), and the non-monotone array then silently broke the
`searchsorted` crossing detection. (b) The deeper point: the ODE solves a
problem whose answer is closed-form. Bo ≈ 0.003, so the zero-gravity
Young-Laplace surface pinned on a rim is *exactly* a spherical cap; a loop
droplet is two caps sharing the rim circle, h solved analytically from the
volume. `add_droplet.py` already implemented this correctly
(`_biconvex_lens_profile`); the implementation moved to
`crystal_harvester/droplet.py` and both callers now share it. Consequences:
`--solvent-volume` is honoured **by construction**; `--contact-angle` is
deprecated and ignored, loudly — with a pinned contact line the contact angle
is an output of (volume, rim radius), not an input, so the old CLI promised
something no solver could deliver; the unread `gravity_ms2` parameter is
gone; and an unpinnable volume **raises** — all three silent hemisphere
guards and the silent sphere fallback in `hampton_loops.py` are deleted.
There is no fallback shape any more, anywhere.

**Placement: the droplet is built in the loop's canonical frame and moved to
the aperture.** The old generator wrote solver vertices into the YAML
verbatim, so the drop sat origin-centred at the loop/stem junction while the
aperture is at x ≈ −0.25 mm. The rim now follows the actual loop outline
(ray-cast against a dense elastica polygon, pinned at the inner fiber edge —
measured 6.8–10 µm from the 10 µm-radius fiber axis, all the way around),
straddles the loop plane symmetrically, and rotates with the loop for any
`loop_axis` (previously only the loop rotated — a latent non-coplanarity for
any non-default axis). The crystal is placed at the droplet's **volume
centroid**, not the waypoint centroid: on a teardrop aperture the liquid body
sits toward the wide side, and the first validator run measured the 25 µm
difference between the two.

**Every generated scene is now validated mesh-back**
(`crystal_harvester/validate.py`, run by the CLI on every build; 18 tests in
`tests/test_scene_geometry.py`): watertightness with no zero-area triangles
(the old apex fans were degenerate), divergence-theorem volume within 2% of
requested, rim pinned on the fiber, plane-straddling (a one-sided dome is the
fallback signature), crystal centred and listed before solvent. The corrupted
variants are each tested to FAIL — the validator provably catches the exact
defects that shipped. A crystal thicker than the drop is a **warning**, not a
failure: a real mount's crystal does poke out of the film, but this renderer
shows hard crystal/air interfaces with no wetting film, so the condition is
worth knowing about and wrong to forbid.

**Root pipeline scripts prefer the repo over the legacy path.** All five root
scripts (`render.py`, `add_*.py`, `generate_scene.py`) carry a historical
`sys.path.insert(0, '/home/jamesh/projects/loop_sim/claude')`; on any machine
where that path resolves (this workspace does, via the mirror symlinks) they
silently imported **James's old copy of `loop_sim` and `crystal_harvester`**
instead of the repo they sit in. Each now inserts its own directory ahead of
the legacy path. The legacy line is kept deliberately — on machines where the
repo layout differs it is still the fallback that makes the tools run.

**What did NOT change:** `scene_files/hampton_300um.yaml` (the bare perf
baseline — every fps number depends on it staying dropletless),
`mitegen_200um.yaml`, both frame libraries, and everything under
`loop_sim/`. Suite 149 → **167**, all green.

**Open, and now the gating question: which camera is real.** NA drives how
bright a correct drop renders (paraxial: a 2 nL biconvex drop passes ~17% of
its area at NA 0.10 and ~100% at NA 0.28), so fidelity judgments still hang
on the calibration fork (`template.yaml` 0.82 µm/NA 0.28 vs the Hampton
scenes' 7.4 µm/NA 0.10). Two leads from the council: the circulating
calibrations are Abbe-consistent as **two zoom settings of one objective**
rather than rival cameras (note `eff_px` scales with zoom while NA does not —
worth a look); and the cheapest ground truth is **one photograph of a real
300 µm loop carrying a drop** on the beamline camera (up during the
shutdown; Jacob captures it — also note a real drop's thickness ~h must
satisfy h ≲ a·NA/(4(n−1)) ≈ 11 µm to read near-background-bright at NA 0.10,
so the photo also calibrates the volume default). Costing before any switch
to NA 0.28: supersample ceiling moves (~1.37× at 0.82 µm px) and both frame
libraries would need rebuilds.

### 2026-08-07 — the droplet generator's failure is recorded, not repaired

`crystal_harvester`'s Bashforth-Adams solver silently substitutes a hemisphere for
ordinary inputs (the measurements are in HANDOFF "Scene fidelity"). It was left
unrepaired, deliberately.

**Why record rather than fix.** The failure is a scaling problem in an ODE, and the thing
that makes it dangerous is precisely that its output *looks* like a droplet — a plausible
dome of the right diameter in the right units. Any fix attempted without a way to check
the answer would be one more plausible-looking shape, and this project has already spent
a week on a class of bug (the float32 hairy fiber, the sub-pixel crop errors, the pan that
ignored φ) whose common feature is that the picture looked fine. So the prerequisite for
touching it is **a validation check that measures drop volume and rim radius back off the
generated mesh** and compares them against what was asked for. Nothing does that today,
which is exactly why a hemisphere shipped unnoticed through a scene-fidelity audit that
cited its dimensions approvingly.

**What was corrected instead.** HANDOFF's claim that `crystal_harvester` is "the
dimensionally-trustworthy source of scenes" was measuring the fallback. The claim is now
split: trustworthy for the mount (loop, fiber, stem, pin — all verified), untrustworthy
for the solvent. That distinction is the durable part; whoever fixes the solver can delete
half a sentence.

### 2026-08-07 — the analysis tree is out of git, but stays on the mirror

Two requirements that these docs had collapsed into one. Experiment scratch should not be
in the deliverable's git history — it is not part of what a successor clones, and it
churns. But it *should* reach the gateway, because the team seeing the work in progress is
a large part of what the mirror is for. The docs previously asserted `investigation/` was
"NOT shipped", which was both wrong and not the intent.

Resolved by location rather than by pattern: the tree now sits beside the repo rather than
inside it, so git never sees it, while the mirror pair carries it by default.

**An rsync exclude cannot hide a tracked file** — worth keeping, because it is the part
that is easy to get wrong. While the harnesses lived inside the repo they were tracked,
and the mirror ships `.git/` wholesale, so they travelled inside the pack files regardless
of the exclude (which was anchored a level above the repo and never matched them anyway).
Any future "this must not travel" requirement has to be met by keeping the content out of
the repo; a pattern in `push-all.sh` is not a mechanism for that.

**`loop_sim_MINE` ships as three pairs rather than one, and that split is what makes the
above possible.** The analysis tree and the repo's `scratch/` both carry workspace-local
paths — 133 files in the June bug-hunt, whose hardcoded roots no longer even resolve, and
one build log in `scratch/`. The push's path-leak gate is fail-closed **per pair**, so as
a single pair those tokens would have blocked the deliverable itself from shipping. Split,
each tree carries its own gate decision: `xtal-loop-sim` stays gated, the two scratch
trees are `nogate`. The asymmetry is deliberate and worth preserving — nothing clones or
imports a scratch tree, so an ungated one costs nothing, whereas a silently-shipped path
leak in the deliverable would cost the one mechanical check this protocol has. The
alternative (ungate the whole pair) was rejected for exactly that reason. This mirrors the
existing `goni`/`chain` and `auto_centering` splits, which exist for the same reason.

Consequence to know about: a pair whose destination name contains a slash relies on its
parent directory already existing on the gateway — rsync creates only the last component.
That is pre-existing behaviour (`computer_vision/goni` fails the same way into an empty
mirror), and it is invisible until someone builds a mirror from scratch.

### 2026-08-07 — the default trace tile is calculated, not measured

**Every scene with a solvent droplet was unrenderable at default settings, and
the fix that "retired risk A" never applied to the default path.** `render_torch`
defaulted to a flat `tile_size = 1_000_000`, so a 640×480 frame went through in
one pass: 307200 rays × 2880 faces × 160 B = **19.8 GB, a hard OOM** on a 16 GB
card. The VRAM-aware sizing added on 2026-07-31 only ran when a caller passed
`tile_size=None`, and almost nobody does — `camera_server`, `bench_frame.py`,
`acceptance_voltron.py`, the `investigation/` harnesses and, critically,
`frame_library`'s own **scout sweep** (`content_window`, frame_library.py:336)
all take the default. Only the library's main render loop passed it through. So
a library build for a droplet scene would have died in the scout, before the
auto-sizing it does use ever ran.

**Measured law, and it is a property of the kernel rather than of one scene:**
peak trace memory is linear in `tile_rays × mesh_faces` at **160 B per ray per
face**, stable to ~1% across tiles of 2048–16384 rays *and* across scenes of 234
faces (`mitegen_200um`) and 2880 (a `crystal_harvester` droplet). Every
Möller-Trumbore temporary in `TSurfaceMesh._mt_batch` is `(B, F)` or `(B, F, 3)`
and there is no AABB cull to shrink `F`. Scenes with no mesh carry no such term
at all — measured 0.5 KB/ray total.

**So the default computes the tile instead of probing for it**
(`fit_tile_size`): `free VRAM × vram_fraction ÷ (faces × 160)`, microseconds, no
trial renders. A meshless scene gets exactly the old flat default, so every fps
and parity number in the repo — all measured on the tube scene `hampton_300um` —
is untouched, and both shipped scenes still render in a single pass.

**Why not simply switch the default to the existing probing ramp**, which is the
obvious move: three reasons, each of which would have been a silent regression.
(1) `_probe_peak` **resets torch's global peak-memory counters**, which
`bench_frame.py` and `acceptance_voltron.py` read — so the TITAN V GO/NO-GO
harness would have reported corrupted VRAM figures, with no test to catch it.
(That function's docstring asserted those two were safe "because they pass an
explicit `tile_size`"; they do not, and never did. Corrected.) (2) It would put
a calibration in the live server's first frame. (3) The ramp's upper rungs are
precisely the 12–15 GB allocations **WSL2 spills on instead of failing**, so on
the dev box it does not OOM, it hangs — measured >10 minutes against 3.2 s for a
fixed tile. The ramp is still there for callers that ask for `tile_size=None`
and want the measured answer.

**Guarded by `tests/test_tile_sizing.py`**, including an end-to-end render of a
2880-face scene through the bare default; verified to fail with
`torch.OutOfMemoryError` against the old default.

### 2026-08-06 — runtime scene switching: build off-lock, install under lock

The server held one scene for the life of the process, so comparing the two
shipped scenes meant a restart and a reconnect of every MJPEG consumer. It can
now be switched live. Five decisions are worth keeping:

**The split is the safety argument.** `_build_bundle` does everything that can
fail — load the YAML, resolve or build the library, allocate the `TorchScene`,
construct the goniometer — **off-lock**, and writes nothing to `self`.
`_install_bundle` writes `self` and cannot raise: ~20 stores and a dict copy, no
I/O, no allocation, not even a `print` (which takes the stdout lock and blocks
on a full pipe). So a failed switch leaves the running scene bit-for-bit as it
was, and there is **no rollback path to get wrong**. The outgoing objects are
stashed in a local and `del`d after the locks release, so CUDA frees do not run
inside the critical section.

**Lock order is `_anim_cv > _scene_lock > _gonio_lock`, with `_frame_cv` a
leaf, and the install NESTS.** `_command_move`, `_command_recenter` and
`_animator_loop` read `self._scene`'s camera and geometry under `_anim_cv`
(that is what §"a preempted animation…" below established), and `_target_pose`
lives under `_anim_cv` too — so the swap has to make both new in the same
instant. Writing them under separate locks *is* the race. It is not a subtle
one: hampton is 0.0074 mm/px against mitegen's 0.001, so a torn read is a
**7.4× error** in every pan and recentre, silently clamped by `servable_pose`,
with no exception and no log line. This repo already carries one unresolved
non-deterministic pose-offset bug (click-to-recentre); a second independent
source of the same symptom would make the first undebuggable.

**A second cycle was found while doing it, and it was live.** `_servable` reads
`self._templates`, and `_set_pose_instant` calls it from *inside* `_gonio_lock`.
Giving `_servable` a lock of its own — the obvious move once `_templates` is
scene-guarded — creates `_gonio_lock → _scene_lock`, which deadlocks against
`_render_now` holding `_scene_lock` and then wanting `_gonio_lock` via
`_snapshot_gonio`. Two threads, one `/motor` during one background render, and
the whole server hangs with the socket still accepting. Fixed by making
`_servable` acquire nothing (documented "caller holds `_scene_lock`") and
hoisting `_scene_lock` outside `_gonio_lock` in `_set_pose_instant`.
`tests/test_server_lock_order.py` checks the order **statically**, because both
halves of this are invisible at runtime: the inversion is call-mediated and
shows in neither function's own body, and `threading.Condition` wraps an
`RLock`, so an accidentally re-entrant `_anim_cv` would silently succeed rather
than hang. Verified by reintroducing the bug into a copy: the checker names it.

**`_scene_lock` is acquired in `_render_now`, not `_render_frame`**, because
`tests/test_server_singleflight.py` subclasses the server and replaces
`_render_frame` wholesale — a lock in there would be bypassed by the very tests
that exercise the concurrency. Cost: a switch waits at most one in-flight frame.
Measured worst cases: ~70 ms on the shipped template path, ~1 s for a
`--templates off` settle frame, 18 s on `--templates off --engine numpy` (where
the stream is already one frame per 18 s, so a one-frame wait is proportionate).
No mitigation; the alternative — render unlocked and discard on a generation
change — relies on a fragile property of the current renderer and does not fix
the pose/geometry pairing in `_snapshot_gonio`.

**The animator needs no quiescing, with one exception that had to be closed.**
A cancelled animation provably does not touch the goniometer (§below), so the
swap just bumps `_anim_gen`. But `_run_animation`'s preempt branch *does* write
two fields — it hands its speed and heading to whoever won — and it writes them
after the install has released `_anim_cv`, so clearing them in the install is
always undone. A `_scene_gen` counter, frozen into each animation the way
`W`/`pixel_size` already were, makes the handoff skip when the scene changed
underneath. Otherwise the first jog in the new scene starts at speed, at a
different mm-per-pixel, on a stage that was just reset to home.

Also: `_compiled_ok` is reset (the trace was built against the old `TorchScene`)
and deliberately **not** re-warmed, because `_warmup_compiled_preview` must run
single-threaded and by switch time the server is not — so a
`--templates off --engine torch` server runs eager previews (6.3 vs 11.9 fps)
after its first switch, and says so on stdout rather than degrading silently.
`_frame_gen` is **not** bumped on install: MJPEG consumers hold no scene state,
and `_jpeg_cache` still holds the old scene's frame at that instant, so bumping
would push every client one duplicate stale part for no new information.
`_invalidate()` alone is the right signal.

### 2026-08-06 — a stale frame library is served as-is, never silently rebuilt

`is_current()` returns one bool for two very different situations, and the
switch path needs them separated:

- **missing** — no manifest, frames absent or damaged, or the scene YAML has
  changed since the build (those frames show a different object). Nothing to
  serve.
- **stale** — a *complete* library that simply was not built the way we would
  build it now. It serves perfectly well.

`frame_library/mitegen_200um` is the live example: 360 usable JPEG frames the
server streamed at 19.8 ms/frame on 2026-08-03, whose only sin is a manifest
older than the `format` and `psf` build keys. `ensure_library` rebuilds whenever
`is_current` is false, so a switch that used it would have started a **~1.9 h
rebuild of frames already on disk** — and with only two bundled scenes, that is
the *first* thing anyone would hit on tabbing across. So the switch path never
calls `ensure_library`; it uses a non-building `library_status` / `_pick_library`
and reads the manifest that is actually there, and `library_diff` names what
differs so the operator gets *"stored as jpeg, not png; built without the
objective PSF; 1× supersample, so zoom is capped at 1×"* rather than "stale".
Building is only ever explicit, via `build=preview|full`.

Verified before relying on it: **nothing in the serving path reads `format`,
`psf` or `psf_sigma_px`**. The keys a manifest is read for are `axis`,
`rendered`, `frames`, `supersample`, `step_deg`, `camera` and `window_mm`, all
of which the legacy manifest has — so serving it cannot `KeyError`.

**Preview libraries go to a separate root** (`frame_library_preview/`, untracked)
rather than into the live one. Building in place would overwrite frames the
serving `TemplateSource` is decoding and caching *by filename*, so a viewer
would keep showing whichever mixture of old and new bytes its cache held. A
full-but-stale library beats a current preview when both exist: the preview is
a coarse stand-in with a 1× zoom ceiling, and preferring it because a build key
drifted would be a quality regression nobody asked for.

**`supersample` is not graded unless the operator names one.** It is the only
build parameter documented as *per scene* rather than as policy: it follows each
camera's sampling against the objective's Nyquist limit, which is 4 for
hampton's 7.4 µm pixel and 1 for mitegen's 1.0 µm one (RUNBOOK "Frame
libraries"). The server holds a single `library_kwargs`, so grading every scene
against one default put a permanent "stale" on whichever scene did not match —
and it was unclearable by rebuilding, because the value being called stale is
the correct one for that scene. Rebuilding mitegen at supersample 4 to satisfy
the check would have been optically wrong, ~29 h, and ~430 MB in git. Everything
else (`format`, `psf`, `n_cond`, `step_deg`, `pan_mm`, `axis`, `jpeg_quality`)
is global policy and is still graded; an explicitly-passed `--supersample` is
graded too, since naming one means meaning it.

**CPU builds are refused, and only the CLI has an escape hatch.** A frame is
~179 s on CPU, so even a 72-frame preview is ~3.6 h. `python -m loop_sim.library
--allow-cpu` exists because that is a deliberate act in a terminal you can
Ctrl-C; the viewer offers nothing equivalent, because a wedged daemon thread
with no cancel endpoint is a far worse place to discover you meant something
else. The refusal is checked per scene *after* the already-current
short-circuit, so a no-op run still succeeds on a GPU-less box.

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

### df64 / "two float32s to emulate float64" (Dekker double-float)

Investigated 2026-08-11 against the sibling repo `nanoBragg`, which implements
it properly (`cuda/docs/DF64-ARITHMETIC.md`, merged to main). **Rejected. The
premise that it offers an order of magnitude is false, and it would be slower
here even if it were free.**

- **nanoBragg measures df64 at 1.38x SLOWER than fp32**, and ~2.04x faster than
  native fp64 on ONE sub-computation (`sincos`) on an RTX 5090. There is no
  order-of-magnitude result anywhere in that repo. The 1/32-1/64 figure that
  circulates is the *hardware fp64 penalty on consumer cards*, cited as
  motivation, not a measured speedup. Its own doc says: **"on hardware where
  float64 runs at half the float32 rate, just use float64."**
- **A df64 value is still 8 bytes**, so it saves zero bandwidth and zero VRAM.
  The mesh path's 160 B/ray/face law is unchanged by it, and that law -- not
  arithmetic -- was what actually cost 43x (see the cull entry above).
- **The frame is not ALU-bound.** Profiling records 73-85% self-CPU, GPU ~29%
  busy, ~23,600 kernel launches per n_cond=1 frame. df64 replaces each f64 op
  with 10-20 fp32 ops, i.e. it adds work in the dimension that binds.
- **It would break the byte-identity anchor.** ~48-49 mantissa bits against
  fp64's 53 is a *different number*, so every `max diff == 0` test against the
  numpy reference would have to be renegotiated, and every tracked library
  rebuilt.
- **Honest point in its favour, recorded so it is not lost:** the 2026-07-06
  fp32-preview rejection was caused by f32<->f64 cast traffic at the
  deliberately-f64 tube-kernel boundary. A df64 scheme has no f64 boundary at
  all, so that specific failure mode would not apply to it.
- **A measurement error to avoid repeating.** An fp32-vs-f64 A/B on the real
  scenes gave 1.22x (tube) and 1.02x (mesh), which looks like a 2% ceiling on
  all precision work. It is not: `TTube._kernel` and `TSurfaceMesh` **hard-code
  `torch.float64` regardless of engine dtype**, so both arms ran an f64
  intersection. That experiment measured fp32 *transport* only. The true fp32
  ceiling remains unmeasured -- it just does not matter, because the bottleneck
  was never arithmetic.
- **Would be worth another look if:** the tracer became a single fused kernel
  that is genuinely fp64-ALU-bound on a consumer card. Note also the cheaper
  classical fix for the underlying cancellation, which needs no df64 at all:
  the 50 mm ray-launch distance is an arbitrary constant, and re-originating to
  the AABB entry point collapses `oaoa` from 2500 mm^2 to O(r^2). Rejected here
  only because it changes every `t` value and would invalidate every tracked
  library for a speedup worth ~1%.

### rasterisation instead of ray tracing

Investigated 2026-08-11. **Rejected: wrong machine for this image, and the
motivating cost was somewhere else entirely.**

The camera is orthographic and both post-trace stages (the objective PSF and
the whole `field.py` camera model) are already separable 2-D operations, so a
rasteriser sounds close. It is not, because three of the effects that define
the picture are not z-buffer quantities:

- **The NA gate is a binary kill on a ray's FINAL direction after up to 12
  refractions.** No depth-buffer value encodes it, and it is what makes 97.7%
  of a raw frame pure 0 or 255.
- **The droplet's brightness is a focal-length-versus-aperture calculation.**
  A wrong drop shape gave `f = R/(n-1) = 441 um`, passing only the inner 9% of
  its area and rendering black; the correct one passes most of the aperture.
  That swing comes from ray optics, not from shading.
- **TIR rim width depends on immersion** (nylon/air 40.8 deg critical, nylon in
  solvent 61.0 deg), and crystal-in-nylon is index-matched at dn=0.01 -- a
  geometric edge that must render as optically invisible.

Also: the 8-half-space CSG crystal has no bounding box, and objects resolve by
YAML priority order rather than by depth, which is not a z-buffer rule.

**And the honest framing:** the frame-library path already IS the "don't ray
trace at runtime" answer, serving at 1-16 ms with no GPU. The cost being
complained about was the offline BUILD, and that turned out to be a missing
AABB cull worth 43x with no fidelity cost at all.

### float32 in the camera-delivery stage

Tried 2026-08-11 to claw back some of the 15.8 ms the camera stage costs on a
rotating frame. **No gain: 11.45 ms against float64's 11.14 ms.** The stage is
index-bound (fancy-indexed gathers for the sensor resample and the streak
scatter), not bandwidth-bound, so halving the element width buys nothing. The
quantised output differed in 3 pixels of 1,013,760 by one grey level, so it was
not rejected for accuracy — it simply is not faster. Distinct from the fp32
PREVIEW rejection below, which was about the tracer and about precision.


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
