# DECISIONS: loop-sim (xtal-loop-sim)

Newest first. The only home for rationale and dead ends: why the code is the way it is,
the measurement behind each call, and what would reopen it. Read by date or topic, never
end to end. To change a fact in a live doc, replace it there and log the reasoning here;
never append a dated correction beside the old sentence.

## Decisions

### 2026-09-10: cleanup pass, and the helpers kept apart on purpose

The code cleanup (commits 2af1cef, 0565816, 5dc7f16) deleted 362 lines of dead code,
merged two byte-identical helper pairs and one duplicated table, and re-stamped the six
manifests' `render_sha` because nine hashed files lost dead lines (no pixel-deciding
line changed; the byte-exact parity tests pass unchanged). Nothing a caller can observe
changed. The full suite went 262 to 259 because three tests existed only to test the
deleted `_mesh_face_count`.

Left duplicated, with the measured difference, so the next pass does not re-propose them:

- `_handle_mjpeg` / `_handle_xray_stream` (camera_server.py): identical modulo four
  names. Merging them was tried and reverted: the shared body takes `with cv:` and the
  static lock-order checker (`tests/test_server_lock_order.py`) only recognises
  attribute locks, so `_frame_cv`/`_xray_frame_cv` would vanish from its view.
- `_frames_complete` / `_xray_frames_complete`: identical bodies, documented as
  deliberately separate so the two frame formats may diverge.
- `DEFAULT_MATERIALS` in hampton_loops.py / mitegen_mounts.py: four of five entries
  identical; `nylon` vs `kapton` differ. `DEFAULT_CAMERA`: pixel_size 0.0074 vs 0.005.
- `resolve_target` / `recenter_target`: the same algebra with opposite sign and a
  different input.
- `_has_cuda` in four test files: six lines each.
- The legacy per-object CUDA path (`Tube._intersect_batch_cuda`,
  `SurfaceMesh._mt_batch_cuda`) is reachable only through `scene.load(device='cuda')`,
  used by two parity tests and `profile_gpu.py`. Removing it is a design call, not a
  cleanup.

Found, not fixed: `digitize_fiber.py` imports matplotlib, which requirements.txt omits;
`--youngs-modulus` is wired through to `elastica_loop`, which ignores it; the `#dbg`
readout in `static/index.html` is scaffolding for the closed click-to-recentre bug.

### 2026-08-19: radiograph ships as a push stream

Radiograph switches from client-side polling of `/xray?t=...` to a real MJPEG-style push stream, reversing the same-day "still, not a stream" call below. The polling design had avoided a second stream's single-flight risk (the class of bug the 2026-07-06 `_active_compiled` incident was: an unlocked shared mutable flag, read/written cross-thread). Reversed because that risk is addressable directly: two producers are safe if neither shares state with the other.

The new producer mirrors the optical producer/consumer pattern (`_bg_render_loop`/`_frame_cv`/`_invalidate`/`_handle_mjpeg`) with entirely separate state (`_xray_jpeg_cache`/`_xray_cache_dirty`/`_xray_frame_gen`/`_xray_frame_cv`), registered in `tests/test_server_lock_order.py`'s `RANK` at the same leaf rank as `_frame_cv` so the existing same-rank-non-reentrant check catches either nesting inside the other, no new test logic needed. The optical producer is never stopped by the toggle: it has other consumers (a second tab, an AXIS-protocol poller) with no notion of it. Ships as single-active-viewer, not connection-refcounted; a second tab watching Radiograph freezes on its last frame if the first tab switches away.

X-ray templates prewarm at scene-load (`XrayTemplateSource.prewarm()`, 2 B/px for 16-bit greyscale vs 4.25 for RGBX). Measured on `hampton_300um_realistic` at boot: optical prewarm 360 frames/2.32GiB/5.2s, X-ray prewarm 360/1106.7MB/3.6s. Streaming throughput while moving: ~28fps, close to the optical rate; idle, it falls back to the same ~1Hz MJPEG keepalive. Test suite: 262 passed (was 252, +10).

Rejected: a connection-refcounted multi-viewer upgrade (start on 0→1 connections), not built: nobody has asked for multi-tab radiograph viewing. Reopens if that changes.

### 2026-08-19: three X-ray radiograph libraries built

All three shipped scenes get a current `xray_library/`, built sequentially on the dev box's RTX 4080 SUPER (one GPU job at a time, shared with the desktop) via `python -m loop_sim.library --modality xray --scene <scene>` at each scene's optically-correct supersample.

| scene | supersample | build time | s/frame |
|---|---|---|---|
| `mitegen_200um` | 1 | 6.1 min | 1.01 |
| `hampton_300um` | 4 | 13.1 min | 2.18 |
| `hampton_300um_realistic` (flagship, mesh) | 4 | 81.5 min | 13.58 mean |

39 MB total; every manifest's frame count (360) and raster size matches what's on disk. The flagship lands close to RUNBOOK's ~82 min estimate, well above the earlier 4-frame DECISIONS projection of ~65-70 min: a 4-pose sample at coarse `--step 90` isn't representative of the full sweep's pose-dependent cost (same lesson as the optical build's 1.48x face-on/edge-on spread, 2026-08-13).

Deferred until the illustrative-vs-real `mu_xray` question settled (below); now decided (keep illustrative), so this spends the GPU time on constants that won't change.

Found while checking the build in: `xray_library/` had no `.gitignore` re-include. `frame_library/`'s `!frame_library/**/*.png`/`!frame_library/**/manifest.json` pattern was never mirrored for the new tree, so all 1083 built files (360×3 frames + 3 manifests) matched the blanket `*.png` ignore and would have been silently dropped from `git add -A`: a fresh clone would see three manifests each claiming 360 frames and zero images, reading as `missing`. Fixed by mirroring the re-include lines; verified `git ls-files --others --exclude-standard xray_library | wc -l` reports 1083, matching `find xray_library -type f | wc -l`.

### 2026-08-19: Microscope/Radiograph toggle and /beam panel ship

The viewer ships a Microscope/Radiograph toggle and a `/beam` panel, closing the last open item of the 2026-08-18 work (WORK_LOG). Only `loop_sim/server/static/index.html` changes: the serving side (`/xray`, `/beam`) was already fully wired by the 2026-08-18 lock fix, so no `camera_server.py` change was needed beyond one read-only addition.

`/xray` already renders the server's live goniometer pose with no arguments and clamps identically to the optical view (`pose_crop(..., clamp=True)`), so both of Phase 3's requirements were already satisfied server-side; the toggle just swaps `#cam`'s `src`, reusing the scene tab-strip's CSS and click idiom verbatim.

Refresh fires client-side on pose-settle, with no new server state: `applyState` (already the single sink for every pose-changing action) computes a signature of `(tx,ty,tz,rotx,zoom)` and refreshes only when two consecutive readings match, so it fires once, right after motion stops. `_scenes_json()` now reports each scene's `xray_library.status` (reusing `xray_library_status()`) so the tab can badge "no library, slow" before a click triggers a live render that could take minutes with no feedback; adds no new lock, builds nothing. The `/beam` readout is a `<details>` element, fetched only while open, on its own DOM elements, never `#state` (rewritten every 600ms) or `#notice`.

Verified: `py_compile` clean; `tests/test_xray_serve.py`/`test_scene_switch.py`/`test_beam_attenuation.py` (40 tests) pass unchanged; the new `_scenes_json` field checked against real on-disk state for all three scenes.

### 2026-08-18: X-ray mu_xray stays illustrative

`mu_xray` ships as the illustrative (exaggerated) values; no scene or constant change. Decided as a usefulness call handed to the user with a rendered comparison, not picked on technical grounds.

Measured by rendering `hampton_300um_realistic` at the home pose twice, same code and geometry, only the absorption constants changed: shipped illustrative values vs. literature-sourced real ones (crystal 0.20 mm⁻¹ cited at 12 keV; metal/pin 130 mm⁻¹ from NIST XCOM iron 170.6 cm²/g × 7.874 g/cm³; solvent/nylon ~0.17-0.20 mm⁻¹, estimated, not independently verified against a primary XCOM query).

Two findings survive regardless of which default ships: peak crystal contrast drops from 21% to 3.1% absorbed (6.8×) with real coefficients. The structure survives (a contrast stretch excluding the saturated pin still shows loop, droplet and crystal), but at ~3% native contrast it reads as noise without deliberate tone-mapping. It isn't a uniform "everything is too high" story: crystal/solvent are overstated 2-10×, but the shipped pin value (100) is if anything slightly low against the cited real one (~130). That's invisible today because both saturate the pin to solid black, but it matters to anyone reusing these constants elsewhere.

`xray_library/` (above) now builds against the shipped illustrative constants.

Rejected: real-physics constants with a display-stretch/path-length tone-mapping option, not chosen against on technical grounds (both findings above hold either way). Stays on the shelf. Reopens if someone wants radiographic correctness over readability.

### 2026-08-18: X-ray radiograph frame library module (xray_library.py)

New module `loop_sim/library/xray_library.py`, not a parameter on `frame_library.py`: separate root (`xray_library/`), separate `render_sha` scope (`renderer/xray_torch.py`, `renderer/engine_torch.py`, `renderer/beam.py`, `scene/*.py`, `motors/goniometer.py`), separate build/staleness functions. Reuses `pose_crop`/`zoom_limits`/`frame_for_angle`/`servable_pose`/`content_bbox`/`crop_to_content`/manifest I/O from `frame_library.py` unmodified.

Storage is 16-bit greyscale PNG (Pillow infers `I;16` from a `uint16` array). 8-bit was rejected before writing any code: the pin transmits at ~1e-31 while the biological signal sits in roughly the top fifth of the range, so 8-bit would spend ~50 of 256 levels on the content the library exists to serve.

No depth blur, and no reader-side special case for it: the manifest's `camera.na_condenser` is stored as `0.0`, which makes `pose_crop`'s existing `sigma_px = 0.5 * na_condenser * |w| / eff_px` formula evaluate to exactly zero for any `tz`: a collimated beam's Beer-Lambert integral doesn't change with the ray's start point along its own direction, encoded as data rather than an if-branch.

Verified byte-exact: re-rendered frame 0 live at the build pose and compared against the stored, cropped, 16-bit-quantized frame, 0 of 1,517,000 pixels differing. `tests/test_xray_library.py` (11 tests, CUDA-gated, build in `tmp_path`) plus full suite: 247 passed (236+11), zero regressions.

Measured build cost (supersample 4, 4 frames of the flagship mesh scene): 10.8-13.1 s/frame, extrapolated to ~65-70 min for 360 frames, later found low by the coarse-sample lesson above (real cost 81.5 min).

Not built yet at this point: deferred until the illustrative-vs-real question above settled, so the ~65-70 min wasn't spent three times on constants that might change.

### 2026-08-18: xray_torch split out of engine_torch.py

`render_xray_torch`/`trace_xray` move out of `renderer/engine_torch.py` into a new `renderer/xray_torch.py`, which is not in `_RENDER_SOURCES`. `trace_xray` becomes a free function taking `tscene` explicitly rather than a `TorchScene` method (its only caller, `render_xray_torch`, moved with it).

Reason: `frame_library.py`'s `_RENDER_SOURCES` hashes `engine_torch.py` whole, and that file held the X-ray functions alongside the shared `TorchScene` machinery, so any X-ray-only GPU edit was silently invalidating all three shipped optical libraries (up to 9.5h to rebuild). An eventual X-ray frame library gets its own separate `render_sha` over `xray_torch.py` instead.

Pure code motion: the full 236-test suite, including byte-exact GPU/CPU parity assertions, passes unchanged. But `engine_torch.py`'s bytes did change, so `render_sha()` changed, which read all three shipped libraries as stale against their recorded hash. Stamped `manifest.json`'s `render_sha` to the new value on all three (`hampton_300um`, `hampton_300um_realistic`, `mitegen_200um`), the same pattern as the 2026-08-10 stamping when `render_sha` was introduced. Justified because the test suite confirms nothing that decides a template pixel actually changed, not assumed.

### 2026-08-18: /beam and /xray release _scene_lock before rendering

`_beam_json`/`_render_xray_png` now snapshot `(scene, tscene, scene_gen, pose)` under one `_scene_lock` hold, release the lock, then render, instead of holding `_scene_lock` for the whole render.

Bug found: on the deployed config (`--templates on`, no `_tscene`), one `/xray` or `/beam` request froze the live camera stream for as long as the numpy render took, because `_render_now` (the MJPEG background producer) needs the same lock. Measured on `hampton_300um_realistic` at 640×480: `/beam` 47.2s, `/xray` 27.9s frozen. The fix is safe because `_install_bundle` swaps `self._scene`/`self._tscene` by reference rather than mutating them, so a snapshot stays consistent even if a switch lands mid-render. Verified: 492 forced re-renders through `_render_now` during a live 27.9s `/xray` call all completed in ≤44ms (mean 2ms), matching pre-fix baseline. Also memoized on `(scene_gen, pose_phase)`: a repeat `/beam` at the same pose returned in 22ms, byte-identical to the first.

Second item, closed together: commit `0548868` (2026-06-25) added real, shadowing-aware Beer-Lambert attenuation to `/beam`/`/xray` (an ordered walk over `scene.path_segments()`, energy conservation checked, guarded by `test_stacked_slabs_downstream_is_shadowed`), but shipped with no DECISIONS entry. That's why a paper note calling attenuation unimplemented (accurate before that commit) was never corrected. Hazard: don't trust that note.

### 2026-08-18: render.py --device cuda unified on the resident engine

`render.py --device cuda` now builds a `TorchScene` and calls `render_torch`, replacing the legacy per-object CUDA path in `scene/tube.py`/`scene/surface_mesh.py`. `camera_server` was already the only caller of `engine_torch`; now there's one GPU path, not two. Verified against `--device cpu` on `hampton_300um`, n_cond=1: 14 pixels differ by >10 (max 18): JPEG double-compression noise, not a trace difference, within `run_gpu.slurm`'s documented n_cond=1 tolerance (≤35K). Both paths were already float64 and already measured byte-identical to the CPU reference, so this changes which code runs, not what it computes.

Compile-fallback silence closed: both fallback sites in `camera_server.py` (`_warmup_compiled_preview`, the runtime fallback in `_render_frame`) print a `WARNING:` to stderr and set `self._compile_error`, surfaced via `GET /scene` as `compile_preview: {compiled_ok, error}`. A scene switch clears `_compiled_ok` but leaves `_compile_error` at `None`, distinguishing "not compiled, just switched" from "compilation failed."

`setup_titan_v_env.bash` packages RUNBOOK's five manual TITAN V deploy steps into one idempotent script ending in an `acceptance_voltron.py` GO/NO-GO. Not run on voltron: this session has no beamline-host access, so it's syntax-checked (`bash -n`) only. Whoever runs it next must confirm the acceptance verdict, not assume correctness from a clean parse.

Open: torch 2.6 as a hard requirement vs. silent fallback. `run_gpu.slurm`'s comparison thresholds (≤35K/60K pixels) describe the old legacy-CUDA-vs-CPU gap and are now stale since that path is gone; not rewritten here because it needs an actual SLURM run on voltron to produce a real number.

### 2026-08-14: all three beamline hosts serve the viewer above 10 fps

One run each, graded on the pre-warmed regime (`slew_warm`), 40 frames,
`hampton_300um_realistic`, stock `/programs/pytorch/envs/pt/bin/python`:

| Host | CPUs | slew (pre-warmed) | in a browser | pan | verdict |
|---|---|---|---|---|---|
| dataserver3 | 40 | 29.5 ms / 33.9 fps | ~16.9 fps | 43.0 fps | GO |
| voltron | 48 | 33.4 ms / 30.0 fps | ~15.0 fps | 45.2 fps | GO |
| gateway (bl831) | 16 | 36.0 ms / 27.7 fps | ~13.9 fps | 31.1 fps | GO |

Per-stage (ms), decode / crop+scale / camera / jpeg: voltron 23.8 / 3.9 / 12.5 / 5.4,
dataserver3 25.0 / 3.9 / 14.1 / 5.5, gateway 33.3 / 5.8 / 18.6 / 7.6.

Core count does not decide it: dataserver3's 40 cores beat voltron's 48, and gateway's 16
still cleared the goal by 2.8x: the serve path is single-threaded, so single-core speed
is what matters and a smaller newer box can win. Treat cross-host gaps under ~15% as
noise: voltron measured 33.4 ms here against 37.7 ms an hour earlier with no relevant code
change, which is shared-node load. dataserver3 runs the DCSS stack (the DHSs,
`touch_tcp`), so it is a control host rather than a compute box, a placement question,
not a performance one.

### 2026-08-14: viewer prewarms; decoded-template byte figures corrected

`TemplateSource.prewarm()` decodes the library at boot, blocking, before the socket binds, instead of filling lazily. Measured on the dev box: the first revolution after a restart goes from 28.01ms (35.7fps) to 13.24ms (75.5fps), for 5.3s of startup cost. `--prewarm off` restores lazy filling; it refuses to prewarm when the cache can't hold a whole revolution, since a partial warm gets evicted before use.

Deliberately not in `TemplateSource.__init__`: `bench_serve` needs a constructor that measures cold costs, and one that silently decoded 360 frames would destroy that, adding 10-30s to every run. Blocking, not threaded, because the server's threading rules are strict and a few seconds at boot is the cheaper trade. `_build_bundle` also warms a switched-to scene off-lock, so the old scene keeps serving throughout a switch.

Second finding while verifying it: PIL stores an RGB image as 4-byte-aligned RGBX, so a decoded pixel costs 4.22 B/px measured (40 real 3940×414 templates: 275.2MB actual vs. 195.7MB predicted by `w*h*3`); every `w*h*3` estimate in the cache path was a third low. That broke `plan_template_cache`'s own guarantee to under-promise: with `w*h*3` it over-promised by 41%. Fixed: `_DECODED_BYTES_PER_PX = 4.25` (measured 4.22, rounded up for safety). `tests/test_frame_library.py::test_decoded_bytes_per_px_matches_what_pil_actually_allocates` checks against the allocator, not an arithmetic model: the exact bug class that caused this.

Retroactive correction to every earlier figure naming decoded bytes: the pre-crop cache was ~20.4GiB, not 14.4; a full-window template was 58MiB, not 41; the cropped sweep is 2.32GiB, not 1.76. The ratio is unchanged at 8.8× (same pixel ratio). Older DECISIONS entries below are not rewritten and remain low by a third wherever they name decoded bytes.

### 2026-08-14: voltron measured: crop lands, remaining gap is memory

The crop below projected ~57ms/17.6fps cold on voltron; measured: 71.4ms/14.01fps cold, 37.7ms/26.53fps warm, on 2.32GiB (corrected from the projected 20.4GiB, 8.8x ratio unchanged). voltron now clears the 10fps goal on a cold slew, which it missed by 3x before; the condition "usable there only with `--template-cache auto`" is retired.

Run it twice after a push: the first run read 94.3ms (p10 71.7/p90 130.0, 1.8x spread); the second read 71.4ms (p10 67.7/p90 77.3, 1.14x spread), p10 barely moved. The gap is first-touch I/O off the shared ZFS pool on files rsynced minutes earlier, ~23ms/frame paid exactly once: the first direct evidence cold I/O on that pool is real and large.

The remaining gap is DRAM, not CPU, and a stage split can't see it: the four stages sum to 52.8ms against a 71.4ms slew, because the split re-renders a few angles back to back (cache-friendly) while a real slew streams a different 4.89MB template from DRAM every frame. On the dev box: holding one template resident costs 10.3ms/frame, two costs 10.4, four costs 14.1, then flat to 1762MB, the L3 boundary (30MB on the E5-2650 v4, ~33MB dev box). `bench_serve` now prints and names this stage-sum-vs-slew gap rather than leaving a 26% discrepancy to read as noise.

Reframes two shelved items without closing either: an LoD tier (4.89MB→1.22MB at zoom≤1) is now a bandwidth argument, not just a decode one; the 2026-08-11 float32-camera-stage refutation ("index-bound, not bandwidth-bound") was measured on the dev box and may not hold on voltron's weaker memory system.

Rejected: raw-uncompressed-on-disk storage, stayed on the shelf. This run is the first direct evidence that cold ZFS I/O is large enough to make it a bad trade.

### 2026-08-14: templates store content only; --mono defaults off

Templates now store only their measured content, not the full render window; the viewer fills the surrounding field at read time. `--mono` now defaults to `off`: the simulator delivers colour by default.

Content occupies 10.4% of a template's frame, the same rectangle at all 360 angles for both hampton scenes (`mitegen_200um` varies). Storing only content is exact, not approximate: raw transmittance is stored (every ray born at 1.0); the photographic look applies downstream. On `hampton_300um_realistic`: 28.9→12.9MB on disk, 15.48→1.76GB decoded (voltron throughput corrected upward in the entry above).

Deliberately not done: the render window is unchanged, no library rebuilt. Cropping alone buys only 1.08x on 8.79x fewer pixels (AABB cull already makes empty-field rays nearly free); `recrop_library` migrates an existing sweep in ~2.5 min with no GPU. `manifest["rendered"]` stays the virtual window, so `pose_crop`/`zoom_limits`/`servable_pose` are untouched; a frame's content location rides as `content_origin_px`/`content_size_px`, whose absence means "the stored image IS the window," keeping pre-crop libraries working. No `_BUILD_KEYS` entry was added (would mark all three libraries stale for ~2h rebuilds over a banner only); `_frames_complete` compares the file against the declared stored size instead, catching cropped/window mismatches symmetrically.

Rejected: the crop is derived from the rendered pixels, not `content_window()`, which scouts coarsely (320x240, n_cond=1) and under-measures, risking clipped sample. An outermost-column special case in `_axis_weights`, added to match PIL's filter-support clamping, measured unnecessary (agrees to 1 level at every column), deleted rather than kept as insurance.

`--mono off` costs nothing (`apply_camera` is faster without the luma matmul, 3.04 vs 4.35ms) and changes at most 20/21/46 levels on 0.003-0.42% of pixels. It exposed that a material's `colour` is an absorption spectrum (`crystal: [0.7,0.9,1.0]` renders blue); the fix is a scene change (`colour: [1,1,1]`, absorption moved into `mu_optical`), deferred since scene files sit in `render_sha` and touching them rebuilds all three libraries.

### 2026-08-13: voltron measured on both halves: build there is a wash, serve there needs the cache

Keep library builds on a workstation; deploying the viewer to voltron requires `--template-cache auto`.

The TITAN V renders 4% faster than the dev box (74.55 vs 77.9 s/frame, `hampton_300um_realistic` at supersample 4, same 1M-ray tile), so a build there is a wash (~7.45 h vs 7.48 h) and not worth it on a shared node. The FP64 hypothesis is refuted: a TITAN V runs FP64 at 1:2 of FP32 against consumer Ada's 1:64 (~8x on paper) but bought only 4%, with the card at 100% utilization throughout: the gap is bandwidth or occupancy (652 vs 736 GB/s), not double precision. Don't pick a card for FP64 strength on this workload.

Serving is where deployment turns: cold, voltron slews at 265.5 ms / 3.77 fps, a third of the 10 fps goal. `--template-cache auto` (14.4 GiB of voltron's 251 GB) brings it to 67.25 ms / 14.87 fps; warm slew and pan agree to 0.05 ms, so the decode is eliminated, not reduced. Without the flag the viewer looks merely sluggish rather than misconfigured. Verify the flag is set rather than trust perceived speed.

Build VRAM margin is thin: a preflight measured a 7.55 GB budget against a ~7.3 GB peak (~250 MB headroom), holding across two heavy poses but only 4 of 360 sampled. Watch a full build; drop `--vram-fraction` before `--supersample` if it OOMs.

Both figures were wrong on the first pass for configuration reasons, not measurement error (warming the benchmark on the poses it was about to time; benching `mono=off` against a server default of on). `bench_serve.py` is now verified byte-identical to `CameraServer._render_frame()` at the same pose.

### 2026-08-12: the droplet scene ships at supersample 4 with a Rayleigh-matched mesh

`hampton_300um_realistic` was regenerated with a droplet tessellated to 50,976 faces
(3.28 µm facets against the 3.35 µm Rayleigh limit at NA 0.10) and rebuilt at
`--supersample 4`, the same rule that gives `hampton_300um` its 4, so zoom reaches 4x and
the tessellation is finer than the optics resolve. Build: 360 frames at 5578x2570, 7.48 h
at 74.8 s/frame, 27.6 MB, peak 7.3 GB; bit-identical to a live f64 render at phi = 30.
A bare launch on this scene is safe from here on (its supersample matches the builder's
default).

Cost of the alternatives, measured at supersample 4 and n_cond 7 before choosing:

| droplet faces | s/frame |
|---|---|
| 5,472 | 20.2 |
| 14,508 | 31.1 |
| 22,464 | 37.8 |
| 50,976 | 74.8 |

A denser mesh or a higher supersample would cost more time and more git and stop buying
anything the objective can resolve. Reopen only if the NA changes (see 2026-08-11 NA fork),
which moves the Rayleigh limit and the supersample ceiling together.

### 2026-08-12: the glint is projected from the scene; the silhouette fit is deleted

The glint's axis and radius now come from the scene (`renderer/pin_projection.py`), not the pin's silhouette (deleted).

The 2026-08-10 "small incorrectness" (glint vanishes when the pin's side leaves frame) hid a real bug: with the pin out of frame entirely, the loop-plus-droplet took the glint at 100% strength at every angle past zoom 2.5x, because the old fit had no connected-component step and merged two dark bodies into one. The discriminating erosion doesn't survive zoom (loop fiber ~2.7 px at 1x, loop+droplet ~279x145 px at 4x). A second, independent defect: the ridge was masked by a global dark-pixel threshold over the bounding box, not the fitted body, so even a correctly fitted pin sprayed its ridge onto the loop and drop. No image-only rule separates a pin from a droplet: aspect and bar-likeness fail, and a "solidity" test fails backwards (0.508 vs 0.582, contaminated pin vs pure droplet).

The fix declares which object is shiny in code (`SHINY = {("pin","metal")}`), not scene YAML, since `scene_sha256` is a build key and a YAML flag would invalidate all three libraries; the module lives in `renderer/`, not `scene/*.py`, to dodge the same rebuild via `_RENDER_SOURCES`. Verified against the silhouette it replaces: row to within 0.4 px, half-width to 1.7 px, start column short by exactly the 6 px the old erosion removed.

Hazard: the clip margin inflates only perpendicular to the pin's axis. Inflating along it too let a pin read "in frame" a third of a mm after leaving, recreating the bug at zoom 2.5-3.0.

`mitegen_200um`'s pin (axis along the beam) is always end-on or wider than the frame; a body seen end-on, or wider than 0.90 of the frame's short side, is declared not to shine, keeping it off at all 24 angles.

### 2026-08-11: the VRAM budget is enforced, not assumed

`memory_budget()` derives a hard ceiling from the card's free VRAM, not total, since voltron is a shared 8-GPU node; `install_vram_ceiling()` makes it an allocator limit rather than advice, because on WSL2 an unenforced overrun spills to host RAM: a 10-50x slowdown that looks like a hang, not a failure.

Two real bugs justified this over trusting "it worked on the 16 GB dev box": a survivor-chunk floor silently demanded 16.7 GB, and `fit_tile_size` had been `min(total_rays, 1e6)` without consulting the card at all. `check_render_fits()` renders one real frame and reads the peak before a build commits, since peak memory is a sum of mesh temporaries, ray arrays and O(W x H) buffers that doesn't fit a clean formula.

A too-large request refuses and reports the largest `--supersample` that would fit, rather than silently building something smaller: a library that differs from what was requested would pass every staleness check, making the failure invisible exactly where the project's integrity checks live.

Rejected: a bigger trace tile. At 14.4 Mpx, 1M/2M/4M/6M-ray tiles ran 18.6/17.5/17.3/17.1 s, byte-identical: 6x the tile buys 8% for 1.8 GB more peak, not worth it on a 12 GB shared card. `LOOPSIM_VRAM_BUDGET_GB` overrides the measured budget so 12 GB behavior is asserted in CI on any card; `set_per_process_memory_fraction` alone can't do this, since `mem_get_info` still reports the real device.

Hazard: never test a memory guard by attempting an oversized render. An early version tried a 40000x20000 frame (17.9 GB accumulator alone) and took the WSL2 VM down. Provoke the refusal by shrinking `LOOPSIM_VRAM_BUDGET_GB` instead; a preflight that measures by rendering needs the ceiling installed first, or the measurement itself overshoots (one run spilled to 13.8 GB while checking whether 12 GB was enough).

### 2026-08-11: the mesh path never culled, and that was 43x

`TSurfaceMesh.ray_intersect` was testing every ray against every face; adding the AABB slab test the numpy reference and `TTube` already used cuts `hampton_300um_realistic` from 80.6 to 1.86 s/frame (8.06 h to 11.2 min per build), every frame byte-identical.

The mesh's AABB covers 0.284% of the render window (mean over a 360-frame sweep), so 99.7% of rays were tested against 5472 triangles they could not hit. The 160 B/ray/face memory law had forced `fit_tile_size` down to 6800 rays (133 passes/frame) to fit the untested faces, so culling alone gave 4.05x and letting the tile return to a single pass gave a further 10.2x, the second effect being larger.

Correctness is provable: every triangle lies inside the vertex AABB, so a ray the slab test rejects provably missed every face, meaning brute force returned INF for exactly those rays. A full 360-frame rebuild diffed against the shipped library came back byte-identical, git confirming only `manifest.json` changed.

The mesh now bounds its own working set (`_mesh_survivor_chunk`) rather than shrinking every caller's tile. Hazard: sizing that chunk as a fraction of free VRAM took ~8 GB on an idle 16 GB card and drove a build to 13.7 GB, inside the WSL2 spill zone: a fraction of free memory is not a bound on a shared card. Capped absolutely at 2 GiB instead, the build held 5.9 GB at no speed cost.

Measured at n_cond 7, f64: `mitegen_200um` 17.0 -> 3.00 s (5.7x); `hampton_300um` (no mesh) 7.93 -> 7.97 s, unchanged as intended. This retires the `--tile-size 6800` RUNBOOK workaround for WSL2 mesh builds. A per-face BVH or CUDA grid was not built: dead by Amdahl once the cull lands. df64 and rasterization were investigated and rejected; see Already Tried.

### 2026-08-11: the NA fork resolves to 0.28: the confound was the space, not the zoom stop

Answer: NA 0.28. Camera-space crystal/background at the hi stop is 0.677 against the photograph's 0.691 (2% gap); NA 0.10 gives 0.421, 39% short. Method: `scratch/na_fork.py` and `scratch/d01_measure.py`, seven direct renders, no beamline access or library rebuild, ~4 min.

The reference was re-measured first and holds: hand-placed boxes on photograph `D01` reproduce the recorded numbers (crystal/bg 0.691 vs 0.696 recorded, crystal/solvent 0.783 vs 0.799).

Rendering at the hi stop instead of mid (the plan's proposed fix) only moves the answer ~1%. What moved it 2.3x was comparing the wrong spaces: earlier comparisons put a render's raw transmittance ratio against a photograph's grey-level ratio, but `field.apply_camera` is affine (`out = (e-B)*t + B`, floor B=0.1765) and doesn't preserve ratios: it lifts dark things hard. The same NA 0.10 render reads 0.186 in transmittance and 0.416 in camera space; the old 0.218-vs-0.696 gap was roughly half unit-mismatch, half physics.

Hazard: the render must also be divided by its own clear-field level before comparing to D01. `field.py`'s vignette is a 41.6%-peak-to-trough bowl fitted to a 2020 session its own docstring calls 5-7x stronger than every other epoch measured; D01 is one of the flat ones (five sky boxes span 2.8% of level). Skipping the correction reads NA 0.28 as 0.818 against 0.691, pointing at NA ~0.17 instead.

Not to re-derive: `template.yaml`'s pixel size (0.8233 um) and the geometrically correct square-pixel hi stop (0.9056 um, built as 704x0.8233/640) give the same tone to 0.3%: pixel size doesn't touch NA, but a future hi-stop scene should use 0.9056. A luma-threshold crystal mask is sampling-dependent across stops; regions here are projected from the scene's own geometry instead.

Switching to NA 0.28 invalidates all three frame libraries, including the 8.06 h `hampton_300um_realistic` build. That switch is the owner's call and is not made here.

### 2026-08-11: the glint met an operator: four defects that only motion shows

All four passed the test suite and looked right in a still frame; each was found only by turning the spindle and zoom through their full range: a camera model has properties no still frame can expose.

The glint's grain had been hashed in the pin's own frame, as if a static texture. That's wrong: a machined shank's micro-facets enter and leave the specular condition as it turns, so the glint should TWINKLE. It now re-rolls per pose: a held pose stays byte-identical, any real move re-rolls it.

The modeled background had been measured wrong: fitting a residual to pixels brighter than the 60th percentile clips the dark half of every cloud and biases the spread down. Masking by dilating the dark body instead gives 3.38/3.48/3.92% of level (was measured 2.6-2.9%), spread across scale rather than one 200 px cell; six octaves of fBm at gain 0.90 reproduce it.

The glint now defocuses with the sample plane (grain sd 5.70 -> 0.26 across 1 mm of depth), since it is reflected light. Reordering the render stages instead of giving this its own step is wrong: it would also soften the illumination field and background, neither imaged from the sample plane.

Separating a zoomed-in pin from a wide mount (`mitegen_200um`) cannot be done on aspect (0.28 vs 0.85) or bar-likeness (35% vs 57%, the pin is LESS bar-like): the mount's fitted body is simply wider than the frame.

Frame time: a rotating frame is 97.2 ms, 75% of it template decode; a browser shows about half the socket's 10-12 fps. Neither a prefetch decode pool (~30 fps) nor `--supersample 2` (~24 fps, zoom ceiling 4x -> 2x) was taken.

The pin-location method here (image-silhouette fitting) was superseded by the scene projection; see §2026-08-12.

### 2026-08-11: click-to-recentre closed by observation, not by a code fix

The operator reports click-to-recentre landing correctly in ordinary browser use of the
viewer, across the sessions since the template path shipped. That is the browser-side
check a 2026-08-06 latency measurement could not supply, and the two readings agree: on
the template path, which serves a frame in ~35 ms instead of the ~1 s an `n_cond=7` live
render took, a recentre driven over HTTP landed 2.5 px from target (centroid from the
server's own served frame, target 320,240 -> 317.5,240.0; part of the residual is the
centroid including an asymmetric stem stub). Both readings are consistent with a
frame/pose lag hypothesis (the displayed frame lagging the live goniometer pose), and
with the move to templates having removed the lag that caused it.

No code was changed to close this, so the standing evidence is two observations rather
than a root cause. If it returns, expect it on `--templates off`, where the slow
`n_cond=7` render still exists and the lag is still possible.

### 2026-08-11: real_images has no mid-stop photograph, and the D set is the dimensionally reliable one

`real_images/` holds 13 scale-carrying frames of loops with drops (B01-B06, B08-B11,
D01/D03/D05) but none at the mid stop, so a mid-stop photograph of a loop with a drop
would let the shipped Hampton scenes be judged at their own stop instead of by transfer
from the hi stop.

Prefer the D set for anything dimensional: its `fov` is 0.579607 x 0.438657 mm, the hi
stop's true non-square pitch (built the same way the mid stop's 7.4 µm is: 704 x 0.8233 /
640). The B set implies square pixels and is 1.1% / 9.8% off. The square-pixel hi stop
used elsewhere is 0.9056 µm, not `template.yaml`'s 0.8233 µm; the two give the same tone
to 0.3%, so the difference only matters dimensionally, not for the NA-fork tone
comparison in the entry above. Loop sizes are unrecorded in both sets, so these
photographs settle tone and NA well and drop volume poorly.

### 2026-08-10: the renders became photographs: camera emulation, sensor resolution, and what gets hashed for staleness

The dominant gap against real photographs was tone, not background: the render was 84.6% pixels at exactly 255 and 14.4% at exactly 0 (1.1% anything else), where real frames carry 19-27% intermediate tone. Fixed as `observed = (E-B)*T + B`, an affine lerp with an analytic vignette `E` (6 quadratic coefficients, 83.5% of the field's variance) rather than a captured image: within one session the background correlates r=0.93-1.00 across angle and translation, but across epochs only 0.11/-0.31/-0.18/+0.35, and the 2020 session an earlier fit used has 5-7x the amplitude of every other epoch: there is no permanent pattern to capture.

The camera stage lives in `renderer/field.py` and runs at serve time (`encode_frame`, downstream of `pose_crop`, outside both tracers) rather than baked into a template. That placement is why no library needed a rebuild, templates keep storing raw transmittance, and `field.py` is deliberately excluded from `render_sha`.

640-wide rendering was confirmed correct against 704-wide photographs: BL831 pixels are 1.110 non-square, so 640x7.4um and 480x7.4um cover the true 4739.6x3587.0um field to 0.08%/0.98%, while rendering 704 wide at 7.4um would over-cover by 9.92%. `template.yaml` is the one place a genuine ~10% vertical error lives; nothing reads that file.

`render_sha` hashes `renderer/{microscope,engine_torch,optics}.py`, `scene/*.py`, and `motors/goniometer.py`, catching a renderer edit that would otherwise leave a stale manifest reading `current`. Deliberately excluded: `field.py` (serve-time), `beam.py` (X-ray), and `library/`/`server/` (hashing `pose_crop` would invalidate every library for a crop-only change). Hazard: a library graded stale rebuilds on the LAUNCH path before the socket binds, 47 min to 9.5 h depending on the scene.

The pin's specular streak got its own geometry-from-silhouette fit here; that method was later superseded by projecting the pin from the scene (§2026-08-12).

### 2026-08-10: the pin's specular streak, as measured on A01 and E02

The tracer models the pin as purely opaque, so it rendered as a flat silhouette; real
pins carry a bright, broken glint along the shank. The stand-in is drawn in camera space
(`renderer/field.py`, `STREAK`), so it costs no library rebuild. Its appearance was
measured on the two reference frames that show a pin at the mid stop
(`real_images/A01_nylonloop_pinleft_mid.jpg`, `E02_digitize_source_mid.jpg`), taking the
pin's edges per column and resampling each column onto a normalised cross-section:

|                        | A01       | E02       |
|---|---|---|
| pin width              | 85 px     | 100 px    |
| floor / background     | 46 / 167  | 60 / 254  |
| ridge centre (fraction of width) | 0.738 | 0.288 (opposite sides of the axis) |
| ridge centre (half-widths off axis) | +0.48 | -0.42 |
| ridge FWHM             | 0.150 w   | 0.112 w   |
| peak above the floor   | 0.82 x bg | 0.15 x bg |
| grain sd on the ridge  | 6.6 lv    | 10.0 lv   |
| grain sd on the body   | 0.50 lv   | 1.30 lv   |
| grain correlation      | 2 px      | 2 px      |

Three spreads are kept rather than averaged: the ridge sits ~0.24 half-widths off the
axis on opposite sides in the two frames (an illumination property, so `offset` is
signed and defaults to E02's side); the peak spans 0.15 to 0.82 of background and the
default sits near the quiet end, because a blown-out glint reads worse than none; the
grain lives on the specular term (applied multiplicatively to the ridge, never the
body). Not modelled: the bright rim at the pin's far edge (f ~ 0.95 in both frames),
edge diffraction plus the grazing return. Where the pin is comes from the scene since
2026-08-12 (`pin_projection.py`); only the appearance is fitted here.

### 2026-08-10: the scene changes that produced the NA evidence

Three changes to `hampton_300um_realistic.yaml` only (`hampton_300um.yaml` is untouched, since the 11.9 fps acceptance figure is measured on it).

Crystal colour `[0.7,0.9,1.0]`/0.02 -> `[1,1,1]`/4.09: the old value rendered the crystal strongly blue against neutral frames. The blue channel's mu (0.02/mm) measures the NA-gate darkening floor directly (T=0.8686); dividing it out, red and green agree on path length (0.0809 vs 0.0832 mm, 2.8% apart). A control render (old drop, neutral crystal) caught the error: skipping the floor division gives a crystal three times too dark.

Pin bevel 45 degrees -> 0: reference photos `A01`/`E02` show square-cut pin ends, not the chiselled tip the geometry had been modeling.

Drop volume 0.002 -> 0.00893 mm3 (half-thickness 20.7 -> 85.6 um) clears all three scene validator warnings and stops the crystal poking out of the solvent. The recorded dissent against this change is void: it argued the drop core was 0.72x background, but that came from sampling a disc about the origin rather than the drop's actual position (x=-0.262 mm since the 2026-08-07 placement fix); measured correctly the drop is already 0.94-0.99x background.

The volume change produced the NA evidence: the drop stays near-background bright at every NA, but the crystal goes dark once the thicker drop immerses a body that used to poke out. Crystal/bg came in at 0.218/0.317/0.455 for NA 0.10/0.17/0.28 against the photograph's 0.696 (transmittance-space numbers; see §2026-08-11 for the camera-space re-measurement that resolves NA to 0.28). Switching to NA 0.28 would invalidate all three frame libraries, including the 8.06 h `hampton_300um_realistic` build, not made here; this entry is the evidence for it.

### 2026-08-07: the black droplet was two scene-side mechanisms; the solver is replaced by the closed form and scenes are validated mesh-back

Supersedes "the droplet generator's failure is recorded, not repaired" below, once its prerequisite (a mesh-back validator) existed.

Two measured causes of black droplets, neither a renderer bug. (1) Material `color` is an absorption spectrum, not a tint (`microscope.py`, `_COLOR_MU`): solvent color `[0.2,0.4,0.8]` absorbed at (24,18,6)/mm despite `mu_optical: 0`, capping brightness at ~0.26 regardless of NA, so every earlier NA experiment is uninterpretable. Near-white `[0.97,0.98,1.0]` moved the NA 0.90 drop-core from 0.2633 to 0.6436. (2) The fallback hemisphere was the worst possible lens (f=441 µm, passing only 29% of radius / 9% of area at NA 0.10). Fixed, drop core reads 0.67 at NA 0.10 against a 0.93 background. `MAX_DEPTH` exhaustion is exonerated (errs bright, not black). Fix is data-side only; a per-channel `mu_rgb` split was rejected to keep `microscope.py` frozen.

The Bashforth-Adams droplet ODE was deleted, not fixed: a coding bug (mixed angle conventions), but more fundamentally the problem has a closed form (Bo≈0.003, a pinned zero-gravity surface is two spherical caps, now shared via `crystal_harvester/droplet.py`). `--solvent-volume` is honored by construction; `--contact-angle` is deprecated and ignored (an output, not an input); an unpinnable volume now raises.

Placement moved to the loop's own frame: the rim follows the real loop outline, pinned 6.8-10 µm from the fiber axis. The crystal sits at the droplet's volume centroid, not waypoint centroid (measured 25 µm difference).

Every generated scene is validated mesh-back: watertight, volume within 2% of request, rim pinned, plane-straddling, crystal before solvent. A too-thick crystal is a warning, not a failure. Root scripts now insert their own directory ahead of a legacy path that silently shadowed the repo. Suite 149→167, green.

Open: camera calibration, NA 0.10 vs 0.28.

### 2026-08-07: the droplet generator's failure is recorded, not repaired

`crystal_harvester`'s Bashforth-Adams droplet solver silently substituted a hemisphere for ordinary inputs, a plausible dome of the right diameter, which made it dangerous. Left unrepaired rather than patched: fixing an ODE with no way to check its answer risks shipping another plausible-looking wrong shape, the same failure class as the float32 hairy fiber and the φ-ignoring pan. The prerequisite was a validator measuring drop volume and rim radius back off the generated mesh; nothing did that, which is why the hemisphere passed a scene-fidelity audit that cited its dimensions approvingly. Also corrected: `crystal_harvester` is dimensionally trustworthy for the mount (loop, fiber, stem, pin), not the solvent. Resolved by the entry above.

### 2026-08-07: the analysis tree is out of git, but stays on the mirror

Experiment scratch (`investigation/`) must stay out of the deliverable's git history (not part of what a successor clones; it churns) but must still reach the gateway mirror, since the team seeing work in progress is most of what the mirror is for. Resolved by location, not an rsync pattern: the tree sits beside the repo, not inside it, so git never sees it while the mirror pair still carries it.

Hazard: an rsync exclude cannot hide a tracked file. While the harnesses lived inside the repo they were tracked, and the mirror ships `.git/` wholesale, so they traveled in the pack files regardless of any exclude. Anything that must not travel has to be kept out of the repo, not excluded by pattern.

`loop_sim_MINE` ships as three push-pairs because the path-leak gate is fail-closed per pair: the analysis tree and the repo's own `scratch/` carry workspace-local paths (133 files with roots that no longer resolve) that would have blocked the deliverable itself as one pair. `xtal-loop-sim` stays gated, the two scratch trees are `nogate` (mirrors the `goni`/`chain` and `auto_centering` splits). A pair whose destination name has a slash needs its parent directory to already exist on the gateway; rsync creates only the last component.

### 2026-08-07: the default trace tile is calculated, not measured

`render_torch` defaulted to a flat `tile_size = 1_000_000`, so a full 640×480 droplet-scene frame went through in one pass: 307200 rays × 2880 faces × 160 B = 19.8 GB, a hard OOM on a 16 GB card. The VRAM-aware sizing added 2026-07-31 only ran when a caller passed `tile_size=None`; almost nothing does, including `frame_library`'s own scout sweep (`content_window`), so a droplet-scene library build died in the scout before auto-sizing ever ran.

Measured law: peak trace memory is linear in `tile_rays × mesh_faces` at 160 B per ray per face, stable to ~1% across tile sizes and scenes (`TSurfaceMesh` has no AABB cull to shrink face count). Meshless scenes carry 0.5 KB/ray total.

Fix: the default now computes the tile (`fit_tile_size`: free VRAM × vram_fraction ÷ (faces × 160), microseconds, no trial renders). A meshless scene still gets the old flat default, so every fps/parity number measured on `hampton_300um` is untouched.

Rejected: switching the default straight to the existing probing ramp. It resets torch's global peak-memory counters that `bench_frame.py`/`acceptance_voltron.py` read (would corrupt the TITAN V GO/NO-GO figures); it would add a calibration step to the server's first live frame; and its upper rungs are exactly the 12-15 GB allocations WSL2 spills on instead of failing (measured >10 min against 3.2 s for a fixed tile). The ramp still runs for callers that explicitly pass `tile_size=None`.

Guarded by `tests/test_tile_sizing.py`, verified to fail with `torch.OutOfMemoryError` against the old default.

### 2026-08-06: runtime scene switching: build off-lock, install under lock

The server held one scene for the process lifetime; it can now switch live. `_build_bundle` does everything that can fail (load YAML, resolve/build the library, allocate `TorchScene`, build the goniometer) off-lock, writing nothing to `self`; `_install_bundle` writes `self` and cannot raise (~20 stores, no I/O, no allocation). A failed switch leaves the running scene bit-for-bit unchanged.

Lock order is `_anim_cv > _scene_lock > _gonio_lock`, and the install nests all three: `_command_move`/`_command_recenter`/`_animator_loop` read the scene's camera and geometry under `_anim_cv`, so writing the new scene and `_target_pose` under separate locks is itself a race. Hampton is 0.0074 mm/px against mitegen's 0.001, so a torn read is a 7.4× error in pan/recenter, silently clamped with no exception or log.

A second, live deadlock surfaced while wiring this: `_set_pose_instant` called `_servable` from inside `_gonio_lock`; giving `_servable` its own lock would invert against `_render_now` and hang the whole server on one `/motor` during one background render. Fixed by making `_servable` acquire nothing and hoisting `_scene_lock` outside `_gonio_lock`. `tests/test_server_lock_order.py` checks the order statically, since a `threading.Condition`-wrapped `RLock` would silently tolerate reentry rather than hang.

`_scene_lock` is acquired in `_render_now`, not `_render_frame`, because `test_server_singleflight.py` replaces `_render_frame` wholesale. Cost: a switch waits at most one in-flight frame (~70 ms typical, up to 18 s on `--engine numpy`).

The preempt branch in `_run_animation` writes speed/heading to the winning move after `_install_bundle` releases `_anim_cv`, undoing any clear. Fixed with a `_scene_gen` counter frozen into each animation. `_compiled_ok` resets and is deliberately not re-warmed post-switch (eager previews, 6.3 vs 11.9 fps, logged to stdout); `_frame_gen` is not bumped, since `_invalidate()` alone is the correct signal.

### 2026-08-06: a stale frame library is served as-is, never silently rebuilt

`is_current()` conflated two situations the switch path needs kept separate: missing (no manifest, damaged frames, or a changed scene YAML) and stale (a complete library, just not built the way it would be today, and it serves fine). `mitegen_200um` was the live example: 360 usable frames at 19.8 ms/frame, whose only fault was a manifest older than the `format`/`psf` build keys. `ensure_library` rebuilds on any `is_current()` false, so a switch using it would trigger a ~1.9 h rebuild of frames already on disk. Fix: the switch path never calls `ensure_library`; it reads the existing manifest via a non-building `library_status`/`_pick_library`, and `library_diff` reports what differs instead of a bare "stale". Building stays explicit (`build=preview|full`). Verified: nothing in the serving path reads `format`, `psf` or `psf_sigma_px`; only `axis`, `rendered`, `frames`, `supersample`, `step_deg`, `camera`, `window_mm` are read, all present in the legacy manifest.

Preview builds write to a separate, untracked root (`frame_library_preview/`), since the serving `TemplateSource` caches decoded frames by filename and would show a mix of old and new bytes mid-build. A full-but-stale library beats a current preview when both exist.

`supersample` is graded only if the operator names one explicitly: a per-scene physical value (4 for hampton's 7.4 µm pixel, 1 for mitegen's 1.0 µm one), not global policy. Grading every scene against one default made mitegen permanently and un-fixably "stale" (rebuilding it at supersample 4 would be optically wrong, ~29 h, ~430 MB in git). Everything else (`format`, `psf`, `n_cond`, `step_deg`, `pan_mm`, `axis`, `jpeg_quality`) stays graded.

CPU builds are refused (~179 s/frame; a 72-frame preview is ~3.6 h); only the CLI has `--allow-cpu`, since a wedged daemon thread with no cancel is worse than a blocked terminal command.

### 2026-08-06: motion is a velocity profile, and the stage speeds were halved

`_run_animation` interpolated linearly (instant full speed, instant stop); replaced with a trapezoidal velocity profile (`velocity_step`): ramp at fixed, distance-independent acceleration (`DEFAULT_RAMP_S = 0.15 s`), hold, brake. Measured on a live 180° move, angular velocity by third: 149 → 170 → 143 °/s.

Speed is carried as state, not derived from elapsed time. A preempted move hands its current speed and heading to its replacement. Deriving position from a clock would restart every profile at v=0, so a burst of jog clicks would brake and reaccelerate at each click, the stutter this was meant to fix. Speed is inherited only when the new move continues the old heading (a reversal starts from rest); measured across a 20-click burst, the pose never stands still longer than 38 ms.

The 0.25 s `min_time` floor (a stand-in for the missing ramp) was removed. A 15° jog is ~0.22 s on its own now.

Rates halved (`cross_time` 2→4 s, `rot_rate` 360→180 °/s, `zoom_rate` 4→2/s): the old speeds were ~2× the real goniometer. `move_duration` now returns the constant-speed time, not wall-clock duration.

For future tuning: the old jog harness's 80 ms frame-gap threshold was calibrated for the JPEG library's ~35 ms/frame; the PNG library's ~68 ms/frame makes it measure decode cost, not motion. Sample `/motor` pose instead.

### 2026-08-06: a preempted animation could write its pose after losing the race

`_run_animation` checked `_anim_gen` and wrote the goniometer in two separate critical sections, and the settle block had no generation check at all. A preempt landing between check and write could stamp the loser's pose over the winner's. Reachable today via `/motor` or a second `/move`; would become a guaranteed corruption under runtime scene switching (a cancelled animation writing the old scene's pose onto the new scene's goniometer).

Fixed as single `_anim_cv` acquisitions. Consequence: a cancelled animation now provably touches nothing, so a scene swap never has to join or quiesce the animator thread.

Same class, fixed alongside: `camera_cfg` reads moved inside `_anim_cv`; `_handle_recenter` now passes the click as a fraction for the server to scale under its own lock, so a target can no longer be resolved against one scene's pixel size and another's axes.

### 2026-08-06: realism pass: the objective PSF, and lossless templates

The ray-traced image is sharper than any real objective: a binary NA gate produces edges no lens can form (measured on hampton, 97.7% of a frame is pure 0 or 255; a silhouette resolves in ~1 template pixel where NA 0.10 at 550 nm has a 3.35 µm Rayleigh limit). Fix: convolve the traced image with a Gaussian approximating the objective's Airy PSF, σ = 0.21λ/NA (λ=550 nm hardcoded), sized in object space so it scales correctly with zoom and supersampling (hampton σ=0.624 template px, mitegen σ=1.155).

The PSF is numpy-only; the torch engine round-trips through it, since two independent blur implementations would differ in truncation and summation order. This surfaced a pre-existing float divergence: the two float64 traces were never bit-identical (up to 3e-8 on ~0.7% of values), invisible while the image was binary but landing on either side of a rounding boundary once the PSF creates intermediate greys. Result: CPU/GPU agree to ±1 grey level with the PSF on, never more.

Templates now store losslessly as PNG: a real AXIS camera applies one JPEG compression, and JPEG templates re-encoded on the wire applied two. PNG is also smaller here (28.7 MB vs the 84.9 MB it replaced), at the cost of decode time (68 ms/frame, 14.7 fps through a sustained spin, still above the 10 fps goal).

Auto-rebuild of a stale library was deliberately left alone rather than blocked behind an explicit command. An un-rebuilt library costs the next person the ~45 min wall-clock.

Caveat: `na_condenser/na_objective = 0.70` is partially coherent imaging; a Gaussian PSF will not reproduce real edge overshoot/ringing. Don't read a measured edge off a rendered frame as exact.

### 2026-08-06: the interactive path: what the operator sees must be what the server means

Five defects surfaced only once a person drove the interactive path (frames were cheap, 13-46 ms, but nobody had watched a browser):

- `--fps-limit` default 5.0 → 30.0. Measured: at 5, every gap is 200 ms (5.12 fps); at 30, median gap 34 ms (28.1 fps). The template work's headline 24 fps was a render-cost number the shipped default couldn't deliver.
- MJPEG parts close as written, and new content gets one prompt resend. Closing a part immediately fixes clients that finalize on the boundary; it can't fix clients that finalize on the next part's headers, since `Content-Length` is unknown until the next frame exists, which needs a duplicate frame sent promptly after new content. Both are required; this regressed twice in one session. Guarded by `test_new_content_is_followed_promptly`.
- Screen-space pan now resolves through Rᵀ. The old code added pan straight into motor axes, coinciding with image axes only at φ=0; the same pan produced an identical `ty` at every angle: pure defocus at φ=90, backwards at φ=180. Verified by FFT phase correlation, 0 wrong of 14. A screen pan now writes `tz`, so any "return to origin" must zero it too; the recenter button didn't.
- The commanded pose is clamped to what the library can serve via `servable_pose()`, inverting the same `pose_crop` box so the two cannot disagree. Invariant: re-requesting the reported pose reproduces the image byte-for-byte.
- Non-zero moves get a 0.25 s duration floor (a 15° jog was 42 ms, ~1 frame, reading as a jump). A/B on a 20-click burst: 18 stalls >70 ms → 1.

Lesson: all five were invisible from render timings and unit tests. Measure the delivered stream, not just the renderer.

### 2026-07-31: templates serve every frame; VRAM stops limiting resolution

The camera server now serves every frame from a pre-computed rotation sweep (`--templates on`, default): nearest spindle angle, crop, scale, blur, encode. Live rendering (`--templates off`) stays available as the correctness reference. Measured cost: 1-16 ms/frame with no GPU at all.

`supersample=4` is a physical number: NA 0.10 gives a 3.35 µm Rayleigh resolution; Nyquist against the 7.4 µm native pixel wants 4.41×. A calibration switch to `template.yaml` (0.82 µm px, NA 0.28) would put the answer at 1.37×.

The render window is measured from the scene, not centred on the goniometer origin: `content_window()` scouts a coarse wide-field sweep and measures where the image differs from background, then a fixed `tx` offset centres it. Hampton needed a 10.32×4.76 mm window against a symmetric-margin guess that left over half the pin unrendered.

Depth is a Gaussian blur (`σ_px = 0.5·NA_cond·|Δz|/eff_px`), not a focus stack; it won't reproduce the discrete 7-replica ghosting a real `n_cond=7` render shows.

Crop math depends on the goniometer's `T = Rz·Ry·Rx·T_trans`: which motor is lateral rotates with φ (at φ=90 `ty` gives zero image shift, `tz` gives all of it). Getting this backwards leaves the picture looking plausible while showing the wrong part of the sample.

Three sub-pixel bugs surfaced only in review: `pose_crop`'s float box needs a half-source-pixel offset for PIL's resize semantics (else every frame is off by 0.375 camera px at 4×); rounding that box to integers makes magnification flicker at pan boundaries; and the template must share the camera's pixel parity or every crop inherits a half-pixel offset.

The zoom floor is measured from the sample window's nearer edge, not `camera/template`; clamping always slides the crop, never squeezes an axis.

### 2026-07-31: the tile clamp was the VRAM ceiling, and it was never a correctness one

`render_torch` already traced in batches, but `tile_size = max(tile_size, W*H)` forced every tile to be at least a whole frame, so peak memory scaled with resolution. Supersedes the 2026-07-28 entry below, which concluded the clamp made tiling "unreachable": it was only a performance floor (verified: tracing the same pose in tiles from 1000 to 307200 rays is byte-identical to a single-pass trace).

A second ceiling was also removed: ray buffers were allocated at the full `n_cond × W × H` before any tracing (5.4 GB at 10.7 Mpx, n_cond=7). Tracing one condenser sample at a time and accumulating drops that 7×; all 62 pre-existing tests stayed green.

Measured law: peak ≈ 0.12 + 1.13 × tile_Mrays GiB on this tube scene, scene-dependent.

The tile is sized by a measured doubling ramp, not an extrapolated slope: a first attempt fitted from two 64k-ray probes and extrapolated 150× to a ~10 M-ray tile, under-predicting by ~2.7 GB on the mesh path and pushing `mitegen_200um` to 15.7/16 GB, a WSL2 spill. Fix: probe against reserved, not allocated, memory, and measure each doubling rung rather than extrapolate. A stride-sampling bug (one fixed stride, re-measuring the same rays past a point) caused one ramp to attempt an 87 GiB allocation before the stride was made per-rung.

Mesh scenes are memory-bound on `TSurfaceMesh`, which still has no AABB cull: at 2880 faces a single ray costs ~69 kB, so `mitegen_200um` renders slower than the 3.4× larger `hampton_300um`.

Sizing must be predictive, never try-and-retry: under WSL2 there is no OOM to back off from. Past capacity the driver silently spills to host RAM and the render crawls 10-50×. A sustained per-frame slowdown during a build is the only spill signal.

### 2026-07-28: deliver a pre-computed rotation sweep, not a faster live renderer

For the AXIS-camera use case, ship a pre-rendered frame library (`loop_sim/library/`, output in `frame_library/`) instead of pushing live frame rate further: a full 360° sweep about the spindle is rendered once and replayed.

Why this collapses the problem: the camera is orthographic, so translating the sample sideways shifts the image by an exact whole number of pixels and changes nothing else (measured: `tx` of 5 px and 20 px reproduce the un-translated frame rolled by exactly +5/+20 px, max pixel difference 0.000000). Panning is therefore a crop, not a render. Rotation is the only motor that changes image content. At 1° steps that's 360 frames, ~12-19 kB JPEG each, single-digit MB total.

Constraints: frames render `margin`× larger than the camera (default 1.5) so there's material to pan into; `crop_window()` raises rather than silently clamping past the limit. The manifest carries a SHA-256 of the scene YAML, so an edited scene rebuilds on first use. Build failures raise rather than degrading quality silently.

Not covered: `zoom` and `tz` (focus) aren't free the way lateral translation is. `.gitignore` ignores `*.jpg` globally but explicitly re-includes `frame_library/**/*.jpg`: the library is a deliverable, not build output.

### 2026-07-28: scene-fidelity audit: the physics is sound, the bundled scene is not

The geometry-to-image chain is dimensionally correct: a pin with a 700.0 µm ground-truth diameter measures 703.0 µm at four independent columns of the rendered image (0.4%, the half-pixel edge threshold), an architecture-independent check comparing against physics rather than another render on the same machine.

But `scene_files/hampton_300um.yaml`, the scene every test, benchmark and the TITAN V acceptance number runs on, is not realistic: its `solvent` is a sphere of `radius: 0.0` (no droplet), it has no crystal, and its `loop_fiber` waypoints span 69×200 µm despite the "300um" name. The radius-0 sphere was optimised around rather than questioned (commit `77fa545`).

Not filled in directly: a real droplet is a `SurfaceMesh`, the exact path that exhausts VRAM, so populating the benchmark scene would make it unrenderable at full resolution and invalidate every fps number measured on it. Prefer a second, generated scene for fidelity work instead.

`crystal_harvester` itself is trustworthy: its 300 µm loop measures 300.4×300.0 µm, its droplet mesh spans 300×300×150 µm, its pin is exactly 700 µm. The hand-built bundled scenes are the outlier, not the generator.

### 2026-07-28: the mesh VRAM law, and why the documented tiling fix is unreachable

Measured law: mesh peak memory ≈ `tile_rays × faces × 24 bytes`, ×~6 for Möller-Trumbore temporaries, driven by tile size and face count, not image resolution. `render_torch` did `tile_size = max(tile_size, W*H)`, so at 640×480 the tile could never be smaller than 307200 rays: ≈19.8 GB predicted, 19.78 GiB observed as an OOM. Corrected the 2026-07-17 entry's safety note (`tile_size=32768` did nothing at full resolution, since the clamp raised it back to `W*H`).

Reducing resolution doesn't rescue it: 320×240 still OOMs; only 160×120 renders. Half-resolution is a real speed lever (3.5× faster) but a weak memory one (only 1.7× less VRAM, resident scene geometry a fixed floor), a poor trade regardless, since the loop fiber is only ~3 px wide and sub-sampling destroys it.

The durable fix identified here was the `TSurfaceMesh` AABB cull, attacking `faces`, the term that actually needs to fall.

Corrected 2026-07-31 (above): the clamp was only a performance floor. Removing it fixed the OOM directly, without needing the cull.

### 2026-07-17: TITAN V measured: 10 fps confirmed (11.9 fps), gated on the software stack

The 10 fps target reproduces on a real voltron TITAN V: compiled preview 11.9 fps median / 10.1 fps p90 (GO), eager fallback 6.3 fps, measured by `acceptance_voltron.py`. The GPU and VRAM were never the bottleneck; the beamline's default software stack is. Supersedes two predictions in the 2026-07-14 analysis entry: the mesh scene does not hard-OOM (fits torch 2.6 at ~11.1 GB, a knife's-edge margin; OOMs on torch 2.0.1), and the 2016 Xeon does not drag the compiled path (compile fuses ~8k launches into ~8 graphs, GPU-bound; eager stays CPU-bound at 6.3 fps).

`torch.compile` needs torch ≥2.x and a modern C compiler. Voltron's defaults block both: `pt` ships torch 2.0.1 (Inductor `pkg_resources` failure), system gcc is 4.8.5 (too old for Inductor's codegen). torch 2.6 + devtoolset-7 clears both; the recipe is in RUNBOOK "Deploy on the TITAN V".

Hazard: `camera_server` catches any compile failure and silently runs eager, so the server looks healthy while missing 10 fps. Verify with `acceptance_voltron.py`, whose verdict is GO only when compile actually engaged and beat eager.

Mesh scenes remain a knife's-edge fit on 12 GB even on torch 2.6; the AABB cull is the safety margin, not yet built.

### 2026-07-14: deploy target is a TITAN V; GPU speed is NOT the bottleneck (analysis)

The beamline runs loop-sim on voltron's TITAN V (Volta, 12 GB), but the 10 fps result was only ever measured on an RTX 4080 SUPER. Profiling showed the frame is CPU-dispatch-bound (73-85% self-CPU, GPU ~29% busy), so the TITAN V's 8.5× FP64 advantage is a red herring: the dominant ops are low-intensity elementwise, not FP64-ALU-bound. Deployment risks identified: (A) 12 GB mesh OOM, (B) silent `torch.compile` fallback, (C) unknown voltron CPU speed.

Measurements (don't re-derive): hampton `n_cond=1` f64, worst-case pose: eager 147.5 ms (6.8 fps), compiled 88.7 ms (11.3 fps); per-launch ~8.25 µs on WSL2 GPU-PV (native Linux typically 3-5 µs). L2 inversion: the f64 chain is L2-served on the 4080 (64 MB L2) but a 7 MB working set won't fit the TITAN V's 4.5 MB L2, falling to HBM2, for a real GPU regression ≈1.5× (thin margin; the GPU would need to be ~2.6× slower to actually bottleneck).

Hazard: byte-exact parity gates are architecture-blind: a Volta box could pass every gate while producing wrong images. Add a numpy-anchored golden-image gate before trusting a new target.

Tooling: CUPTI works under WSL2; `torch.profiler` reports zero GPU time from a `CLOCK_REALTIME` vs `CLOCK_MONOTONIC` mismatch (fix: env `CuptiUseRawGpuTimestamps=false`). `bench_results/` is gitignored.

Superseded 2026-07-17 (above): the mesh scene does not hard-OOM and the CPU does not drag the compiled path once the software stack is fixed.

### 2026-07-06: 10 fps interactive via a flag-gated torch.compile preview path

Motion/preview frames route through `torch.compile` (`--compile-preview`, `--preview-mode`, both default on); settled frames stay eager. Why: a frame is CPU-dispatch-bound (~8k kernel launches/frame, ~11 fps dispatch ceiling for pure PyTorch), and `torch.compile` fuses the launch explosion (~1.7-1.86×), which is what clears the 10 fps bar.

Constraints: CUDA-only (Inductor's CPU backend miscompiles the mesh/CSG path: mitegen came out 1734/6912 px wrong); `mode="default"` with `dynamic=True`; the compiled function must warm single-threaded in `start()` before any worker thread exists (concurrent first-compile crashes dynamo); the compiled/eager choice is threaded explicitly through the call chain, never via shared mutable state (a shared `_active_compiled` flag caused a measured `/xray`-vs-background-render data race); sticky eager fallback on any runtime failure.

Hazard: Inductor reorders float arithmetic at tangent boundaries, shifting `best_t` by ~1 ULP and flipping up to ~16 tangent CUDA pixels. The strict `max==0` parity gate is relaxed to a ≤16 px flip budget on the compiled (CUDA) path only; CPU stays exact. Do not tighten that gate back to 0 on the compiled path.

### 2026-07-06: dev-box eager/compiled render timings (RTX 4080 SUPER)

Measurement from the optimization pass that added the flag-gated `torch.compile` preview
path (`--compile-preview`). Hardware: RTX 4080 SUPER, 640x480, hampton scene, byte-exact
unless noted. The qualitative design (compile once single-threaded before any worker
thread, never `mode="reduce-overhead"`, threaded compiled/eager choice) is in DECISIONS
§2026-07-06; these are the numbers behind it, not reproduced there.

Eager engine: n_cond=1 ~154 ms, n_cond=7 ~988 ms (was 180 ms / 1193 ms before this pass).

Compiled preview path (moving-pose frames only, settled/`/xray`/offline renders stay
eager): ~25 fps to two MJPEG clients during animated motion (median render ~36-41 ms);
worst case 10 Hz `/motor` stream with the loop centered, ~9.8 fps.

Reproduce with `bench_frame.py --compiled` (RUNBOOK "Verify").

### 2026-07-06: converge-on-idle preview policy (approximate in motion, exact at rest)

During motion the server renders fast `n_cond=1` previews (optionally compiled); a full-`n_cond` bit-exact f64 frame renders automatically once the pose settles (`--settle-delay`, default 0.5 s). Offline renders and `/xray` are always exact. Why: loop-sim's purpose is a drop-in AXIS camera at ~10 image/s, where motion smoothness matters more than per-frame exactness during motion, and settled output is never degraded (test-gated: `test_server_settle_parity`). Hazard: making motion frames "exact" collapses fps; the invariant is exact-when-settled, not exact-always.

### 2026-06-25: GPU-resident torch engine, byte-identical to the numpy reference (f64)

`loop_sim/renderer/engine_torch.py` runs the whole trace on-device (`TorchScene`, `render_torch`); the numpy `microscope.py` path is frozen as ground truth and kept as the CPU fallback. The active ray set uses compaction, not masking. Why: profiling showed ~84% of the render was numpy on the CPU (only the intersection touched CUDA, leaving the GPU ~84% idle). Keeping all ray state resident (one upload/download per frame, condenser batched) was the real ~6-8× speedup, not faster kernels. Do not let the torch engine diverge from the numpy reference: it is both the correctness anchor and the no-GPU fallback, enforced by `tests/test_torch_render_parity.py` (including full-res CUDA).

### 2026-06-25: float64 in the GPU intersection quadratic (THE correctness fix)

`TTube`/`TSurfaceMesh` compute the intersection quadratic in float64 internally, always, even on CUDA (`tube.py`, `surface_mesh.py`, `engine_torch.py`). Why: float32 catastrophically cancelled in `c_ = baba*oaoa - baoa² - r²*baba`, amplified by the microscope's 50 mm ray-march origin offset. For a ~7.5 µm fiber 50 mm upstream, `oaoa ≈ 2500 mm²` swamps the r² signal (~6e-9, below the float32 noise floor ~1.6e-8), flipping the discriminant sign, producing a median ~9 µm t-error, poisoning normals (~19-32°), sending refracted rays ~20° off, and tripping the binary NA/TIR gates into full 0↔255 pixel flips: the speckled "hairy/spikey" fiber. Decisive isolation: forcing only the tube intersection to f64 (rest stays CUDA) dropped the hampton roty45 diff from 203 px to 0.

Hazard: down-casting the intersection geometry or its inputs to float32 brings the artifact straight back; this is also why an fp32 preview path was rejected. Guarded by `tests/test_gpu_cpu_parity.py`.

### 2026-06-25: threaded camera server

`CameraServer` inherits `ThreadingHTTPServer` (one thread per connection). A plain single-threaded `HTTPServer` was starved by a browser's idle preconnect socket: the MJPEG stream "loaded forever" and `/motor` couldn't be served while streaming.

### 2026-05-22: CPU-vs-GPU diff thresholds on the legacy per-object CUDA path

Historical measurement of the retired legacy per-object CUDA path, from the same commit
(`e9e72c7`) that fixed the material-probe bug and switched the tube kernel to float64.
Superseded 2026-08-18, when `render.py --device cuda` moved onto the GPU-resident
`engine_torch` (float64, measured byte-identical to the CPU reference); see DECISIONS
§2026-08-18. A fresh run of `run_gpu.slurm` should read far tighter than the numbers below.

Timing (704x480, TITAN V on voltron):

| n_cond | CPU     | GPU   | speedup |
|--------|---------|-------|---------|
| 1      | ~179 s  | ~9 s  | 20x     |
| 7      | ~1230 s | ~23 s | 53x     |

Acceptable pixel-diff thresholds measured on the same runs (PNG, lossless, 704x480):

| n_cond | pixels>10 | notes |
|--------|-----------|-------|
| 1      | <= ~35K   | baseline; nearly all diff pixels are >60 (TIR/NA flip) |
| 7      | <= ~60K   | higher than n_cond=1 because 7 angles sample more edge cases |

Almost all differing pixels are binary flips (TIR or NA cutoff crossing due to float32
geometry), not gradual noise. The n_cond=7 count is ~1.6x the n_cond=1 count because
independent condenser rays can each flip a different set of edge pixels.

### (inherited from James's model: recorded so they aren't "cleaned up")

- The twisted-pair stem must be two separate tube objects (`stem_1`, `stem_2`). A single helical tube models only one fiber; the crossover shadows disappear and the stem looks wrong.
- Crystal must be listed before droplet in scene priority. A crystal voxel is also inside the droplet; if droplet is listed first, every crystal voxel is assigned `solvent` and the crystal renders invisible.

## Already Tried

### probe-ray material lookup after an interface (`_obj_index_at_points_batch`)

The original `next_interface` found the material a ray enters by casting a probe
ray from a point just past the hit. Probe-point ambiguity made it wrong on ~50% of
crossings; the interval check on the already-computed `all_te`/`all_tx` (see
`Scene.next_interface`'s docstring in loop_sim/scene/scene.py) replaced it. The probe-ray methods sat unused until
2026-09-10, when they were deleted as dead code. Do not reintroduce them.

### three template-serving speedups shelved after the crop (prefetch pool, raw on disk, mip tiers)

Measured 2026-08-13/14 on the dev box, before and around the content crop; all three were
made unnecessary by holding the cropped library in RAM (`--template-cache auto`), which
takes voltron to 26.5 fps warm with no threads and nothing to predict.

- Prefetch decode pool: driven through the server's own velocity profile, neighbour
  prefetch hits 6-38%, extrapolated prefetch ~50%. A perfect prefetch would reach 34 fps on
  voltron; the resident cache gets there without it. The AXIS consumer's `/motor` is
  instant and absolute, so it has no predictable slew to prefetch along.
- Raw uncompressed templates on disk: raw beats decoding even cold (17.5 vs 76.7 ms for a
  full frame, ext4 NVMe), but a 12.9 MB cropped PNG library serves faster (12.1 ms) than a
  15.48 GB raw full-window one (17.5 ms). Cropped-and-raw would be 0.61 ms/frame at
  1.76 GB, worth it only if the crop alone were not enough. voltron's `/home` is ZFS on a
  shared pool; first-touch there measured ~23 ms/frame, paid once per host.
- Level-of-detail mip tiers: `Image.reduce` decimations cost no render time (20 s and
  +7 MB for two levels) and buy a further ~1.6x, but a zoom that moves 2800-4300 px by
  5-16 levels shows a visible tier-switch pop. On the shelf.

Reopen if a host cannot hold a whole revolution in RAM: the cache declines rather than
half-filling, because LRU on a cyclic sweep evicts each frame just before it comes round
again (measured: cache 16 of cycle 32 gives lap-2 69.8 ms against 12.9 ms warm).

### df64 (Dekker double-float) to emulate float64

Rejected 2026-08-11 against nanoBragg's df64 implementation
(`cuda/docs/DF64-ARITHMETIC.md`): the speedup premise is false, and df64
would be slower here even if free. There it measures 1.38x slower than
fp32, and only 2.04x faster than native fp64 on one op (sincos, RTX 5090),
with no order-of-magnitude result in that repo, whose doc says just use
float64 on such hardware. A df64 value is still 8 bytes: no bandwidth or
VRAM saved against the 160 B/ray/face law that cost 43x (see the
AABB-cull decision). The frame is not ALU-bound (73-85% self-CPU, GPU
~29% busy, ~23,600 kernel launches per n_cond=1 frame), so df64 adds work
in the wrong dimension (10-20 fp32 ops per f64 op) and breaks
byte-identity (48-49 mantissa bits vs fp64's 53), forcing rebuilds.

An earlier fp32-vs-f64 A/B (1.22x tube, 1.02x mesh) looked like a 2%
ceiling; it is not, since `TTube._kernel` and `TSurfaceMesh` hard-code
float64 regardless of dtype, so it measured fp32 transport only, and the
true fp32 ceiling remains unmeasured.

Reopen only for a single fused kernel fp64-ALU-bound on a
consumer card. A cheaper non-df64 fix exists (re-origin the ray launch to
the AABB entry point) but changes every `t` value, invalidating every
library for a ~1% gain.

### rasterisation instead of ray tracing

Investigated 2026-08-11. Rejected: wrong machine for this image; the
motivating cost was elsewhere.

The camera is orthographic and both post-trace stages (the objective PSF and
the `field.py` camera model) are separable 2-D operations, so a rasteriser
sounds close. It is not: three defining effects are not z-buffer quantities.
The NA gate binary-kills a ray's final direction after up to 12 refractions;
no depth-buffer value encodes it, making 97.7% of a raw frame pure 0 or 255.
The droplet's brightness is a focal-length-versus-aperture calculation: a
wrong drop shape gave `f = R/(n-1) = 441 um`, passing only the inner 9% of the
aperture and rendering black; the correct shape passes most of it, a swing
from ray optics, not shading. TIR rim width depends on immersion (nylon/air
40.8 deg critical, nylon in solvent 61.0 deg), and crystal-in-nylon is
index-matched at dn=0.01, a geometric edge that must render as optically
invisible. The 8-half-space CSG crystal has no bounding box, and objects
resolve by YAML priority, not depth.

The frame-library path is the "don't ray trace at runtime" answer, serving at
1-16 ms with no GPU. The real complaint was the offline build, fixed by a
missing AABB cull (43x, no fidelity loss).

### float32 in the camera-delivery stage

Tried 2026-08-11 to claw back some of the 15.8 ms the camera stage costs on a
rotating frame. No gain: 11.45 ms against float64's 11.14 ms. The stage is
index-bound (fancy-indexed gathers for the sensor resample and the streak
scatter), not bandwidth-bound, so halving the element width buys nothing. The
quantised output differed in 3 pixels of 1,013,760 by one grey level, so it was
not rejected for accuracy: it simply is not faster. Distinct from the fp32
preview rejection below, which was about the tracer and about precision.

### global Lagrange / Neville polynomial waypoints for the fiber path

Already implemented, and deliberately bounded. Every fiber path (the loop,
stem strands, the droplet rim, the Kapton outline) is built by
`neville_sample()` in `loop_sim/scene/tube.py`; Neville's algorithm is
Lagrange interpolation (same polynomial, stabler evaluation), so "use
Lagrange waypoints" is already the design.

The global form is used only for 4 or fewer waypoints (degree <= 3). Above
that it switches to scipy's `CubicSpline`, because global Neville at degree
5+ produces Runge-phenomenon knots on curved paths. Real loop paths carry
40-59 waypoints, so a global fit would be degree ~58, putting visible
oscillations in the fiber. It runs once at scene load, off the render hot
path, offering no speed lever.

Fiber quality is lost elsewhere: the curve is sampled into `n_samples - 1`
capsules, each tested by every ray. At the default `n_samples=50`, a 300 um
loop gives 19.3 um segments against a 20.0 um fiber diameter, capsules as
long as they are wide, rendering as a beaded ring, not a smooth fiber.
Raising `n_samples` fixes the appearance and costs render time roughly
linearly.

Reopen only for a performance goal, not smoothness: the target is an
analytic swept-surface intersection or hierarchical culling over the
capsule chain, not a different interpolant.

### fp32 preview mode

Rejected. On the sync-starved compiled engine, fp32-compiled is about 2x slower
than f64-compiled (approx. 188 ms vs 120-126 ms on worst-case poses), and
fp32-eager also loses on the worst case. The f32/f64 cast traffic at the
deliberately-float64 tube-kernel boundary outweighs the halved memory bandwidth,
and fp32 perturbs the material-probe interval enough to change ray-path work
profiles. The often-cited June "~2x fp32 win" microbench predates this pipeline
and no longer holds. Reproduce with `bench_frame.py --fp32` (the negative result
is recorded in commit `fb38fdb`).

Reopen if the whole engine goes fp32-native. The intersection quadratic has to
stay f64 (see the correctness decision), so any fp32 scheme has to keep that
boundary, which is exactly where the cast cost showed up.

### `torch.compile(mode="reduce-overhead")` / CUDA graphs in the threaded server

Rejected. `reduce-overhead` silently enables CUDA-graph capture, which is not
thread-safe in `ThreadingHTTPServer` ("already recording to mempool_id" /
`CUBLAS_NOT_INITIALIZED`). Whole-frame graph capture is also blocked by the AABB
cull's data-dependent `nonzero` (a graph-break), and dropping the cull to enable
capture gives a catastrophic 1.95 fps (a 50x tube blow-up). The AABB cull and
whole-frame capture are mutually exclusive.

Reopen if a capture strategy that tolerates dynamic shapes is found.

### custom Triton megakernel (one thread per ray, whole bounce loop in registers)

Proven byte-identical but only about 1.1x (54 ms vs torch's 60 ms) on this fp64
thin-fiber workload: it is GPU-compute-bound on fp64 and trades dispatch overhead
for dense-depth compute, a wash on this geometry. Not worth the complexity; it
was a throwaway prototype and no code remains.

Reopen, uncertainly, for fp32 (with its cancellation risk) plus a single-pass
tube and 2D ray tiling.

### dense alive-mask `trace_rays` (instead of compaction)

Byte-identical but regresses: 5.6 fps against compaction's 6 fps, because it runs
12 full-N depths versus about 3. CUDA graphs still do not engage, since the
cull's `nonzero` still graph-breaks. Compaction is the correct design.

### "fix the hairy fiber" via normals, gate-softening, or more condenser rays

All four refuted. The root cause is the float32 intersection `t`, not the
normals or the gates. `recompute_normals_f64` is already optimal given a correct
`t` (0 degree error); forcing CUDA's `best_k` makes normals worse; softening the
NA/TIR gates catches about 0% of the flips; and raising n_cond makes the fiber
hairier (the flip count roughly doubles), not smoother. Float64 at the quadratic
is the only fix.
