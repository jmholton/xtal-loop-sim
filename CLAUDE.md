# CLAUDE.md — loop-sim

Bright-field microscope simulator for protein crystals in cryo-loops.
See README.md for user-facing documentation.

## Python interpreter

Always use `/programs/pytorch/envs/pt/bin/python` — not `python3` or
`/usr/bin/python3`.  The system Python is 3.6, owned by root, and has none of
the required packages.  The PyTorch env has numpy, scipy, PIL, pyyaml, and
PyTorch+CUDA.

## GPU rendering (voltron)

GPU-accelerated rendering requires CUDA — only available on voltron.
Submit via SLURM from the local machine (no SSH needed):

```bash
sbatch run_gpu.slurm          # gpu partition, gres=gpu:1, no --time
squeue --job <jobid>          # check status
```

Do **not** set `--time` in SLURM job scripts — this queue has no time limits
and the flag causes jobs to be cancelled prematurely.

If you need an interactive SSH session (not SLURM): voltron's login shell is
tcsh, which does not parse `&&`.  Always write a bash script and invoke it with
`ssh voltron "cd $PWD ; bash script.bash"`.

`--device cuda` enables the GPU path in `render.py` and `scene.load()`.
`--device cpu` is the reference path (float64, no PyTorch required).

Performance (704×480, TITAN V on voltron):

| n_cond | CPU    | GPU   | speedup |
|--------|--------|-------|---------|
| 1      | ~179 s | ~9 s  | 20×     |
| 7      | ~1230 s| ~23 s | 53×     |

Those are the legacy per-object CUDA path on voltron's TITAN V.  A newer
**GPU-resident engine** (`loop_sim/renderer/engine_torch.py`) runs the whole
trace on-device and is byte-identical to the numpy reference in float64.
`camera_server` uses it automatically when CUDA is present; `render.py --device
cuda` still uses the legacy per-object path.

2026-07-06 optimization pass (RTX 4080 SUPER, 640×480, hampton, all byte-exact
unless noted): eager engine n_cond=1 ≈ 154 ms / n_cond=7 ≈ 988 ms (was 180 /
1193).  The server additionally has a **flag-gated compiled preview path**
(`--compile-preview`, default on): moving-pose frames route `next_interface`
through `torch.compile(mode="default", dynamic=True)`, and the heavy tube
`_kernel` is compiled SEPARATELY with a dynamic survivor dim (the AABB cull's
data-dependent `nonzero` would otherwise re-specialize per pose — the
compiled/eager choice is threaded explicitly through the call chain, never
shared mutable state).  "Moving" = an animated `/move` OR any instant `/motor`
set within `--settle-delay` (0.5 s) — the AXIS-consumer path; one exact frame
auto-renders on pose quiet.  Measured: **~25 fps to two MJPEG clients during
animated motion (median render ~36-41 ms); worst-case 10 Hz `/motor` stream
with the loop centered ≈ 9.8 fps**; settled frames, `/xray`, and offline
renders stay bitwise-exact eager.  Compilation happens once, single-threaded,
in `start()` before any worker thread (never use `mode="reduce-overhead"`
here — its CUDA-graph capture is not thread-safe in this threaded server).
**fp32 preview was tried and REJECTED** (commit fb38fdb: ~2× SLOWER compiled —
the f32↔f64 casts at the deliberately-float64 tube-kernel boundary outweigh
the bandwidth saving; do not re-propose).  Benchmark with `bench_frame.py`
(`--compiled`, `--fp32`); phase-gate the live server with `soak_server.py`, which
lives outside this repo in the analysis tree at
`/home/jadoughty/projects/loop_sim_MINE/investigation/2026-07_scene_and_perf_harnesses/`
(mirrored to the gateway, but not versioned here).

## Condenser sampling (n_cond)

`--n-cond 7` (1 centre + 6-point hex ring) gives smooth edge transitions.
`--n-cond 1` is fast preview mode (hard NA step at the objective edge).

Do **not** use n_cond > 7 without a specific reason — it adds render time with
diminishing returns on edge quality.

## Scene assembly order = rendering priority

The object list in `scene.yaml` is priority-ordered: first entry wins at any
point in space.  When assembling scenes with `generate_scene.py`:

```bash
python3 generate_scene.py loop.yaml crystal.yaml droplet.yaml ...
```

- **Crystal must come before droplet.**  A point inside the crystal is also
  inside the droplet.  If droplet is listed first, every crystal voxel is
  assigned `solvent` and the crystal is invisible.

## Stem topology

The twisted-pair stem **must** be two separate tube objects (`stem_1`,
`stem_2`) output by `add_stem.py`.  Do not merge them into one tube.

The crossover points where the two fibers intersect produce characteristic
dark shadows.  A single helical tube models only one fiber: the second is
absent, crossover shadows disappear, and the stem looks wrong.

## Architecture overview

```
render.py                    CLI: loads scene, drives goniometer, calls render()
loop_sim/
  scene/
    scene.py                 YAML loader; Scene.next_interface(); path_lengths();
                             path_segments() (ordered front-to-back, for X-ray attenuation)
    primitives.py            HalfSpace, Sphere, Cylinder, Box, Capsule, Ellipsoid
    tube.py                  Neville-chain tube; CUDA hot path (_intersect_batch_cuda)
    surface_mesh.py          Möller-Trumbore mesh; CUDA hot path (_mt_batch_cuda)
    csg.py                   Intersection / Union / Difference
    materials.py             Material dataclass; AIR, WATER, NYLON constants
  motors/
    goniometer.py            SE(3) from tx/ty/tz/rotx/roty/rotz/zoom
  renderer/
    microscope.py            Snell's law ray tracer; Beer-Lambert; NA cutoff (numpy reference)
    beam.py                  X-ray grid probe → per-material volume + Beer-Lambert
                             attenuation (absorbed_dose, transmitted_frac,
                             beam_transmission); render_xray_numpy (radiograph, CPU ref)
    engine_torch.py          GPU-resident torch engine (TorchScene, render_torch);
                             byte-identical to microscope.py in float64, ~6-8x faster;
                             render_xray_torch (straight-ray transmission map)
  server/
    camera_server.py         AXIS HTTP server; renders via engine_torch on CUDA, else
                             microscope; /beam (JSON) + /xray (radiograph PNG);
                             control page (/), animated /move + /recenter (daemon
                             animator thread interpolates the goniometer)
    static/index.html        interactive control UI (crosshair, pan/rot/zoom, speed dial)
```

## Server animation & concurrency

`/move` and `/recenter` are **animated**: a daemon `_animator_loop` thread
linearly interpolates the goniometer toward a target by wall-clock time, so the
streamed sample glides instead of teleporting.  Non-obvious bits:

- The live goniometer is shared mutable state, so it gets its **own**
  `_gonio_lock` (separate from the JPEG frame slot's `_frame_cv`).  Rendering
  never reads the live goniometer directly — it takes a `_snapshot_gonio()` (a
  fresh Goniometer at the locked-in pose) to avoid torn reads mid-interpolation.
- Frame production is **single-flight**: `_bg_render_loop` is the sole caller
  of `_render_now` while serving.  `_invalidate` sets a dirty flag and notifies
  `_frame_cv`; the loop claims (clears) the flag *before* rendering, so
  mid-render invalidations coalesce into at most one re-render.  Each published
  frame bumps `_frame_gen`; MJPEG/snapshot handlers are pure consumers (wait
  for a newer generation, send newest-only, clamp to `fps_limit`, resend the
  cached frame after ~1 s idle as a keepalive).
- During motion `_anim_active` forces **n_cond=1** preview; a final full-`n_cond`
  frame is rendered on settle.  (n_cond=7 is far too slow for smooth motion.)
- New moves **preempt** via an `_anim_gen` counter (the running animation checks
  it each step and bails); relative/pan moves resolve against `_target_pose`
  (the last *commanded* target, not the in-flight pose) so rapid clicks
  accumulate.  `/motor` stays **instant** (cancels any animation) for AXIS
  back-compat — the UI uses `/move`.
- Speeds: translation crosses the field-of-view width in ~4 s (zoom-aware via
  `eff_px`), rotation 180°/s (30 rpm), zoom 2/s, scaled by the `speed` dial.
  Halved on 2026-08-06 — the old rates were about twice what the real
  goniometer looks like, so what needed the dial at 0.5x is now 1.0x.
- **Motion is a velocity profile, not a position tween** (`velocity_step`).
  Speed ramps at a fixed acceleration (`DEFAULT_RAMP_S` = 0.15 s to full speed,
  distance-independent), holds, then brakes so the stage arrives at rest;
  moves too short to reach full speed come out triangular.  Speed is STATE, so
  a preempted move hands its speed and heading to its replacement and a burst
  of jog clicks stays one continuous motion — recomputing position from t=0
  would brake to a stop at every click.  Inherited only when the heading
  continues; a reversal starts from rest.  The geometry math (`resolve_target`
  / `move_duration` / `velocity_step` / `recenter_target`) is factored into
  pure module-level functions (unit-tested in `tests/test_server_controls.py`).
  Note `move_duration` returns the CONSTANT-SPEED time — the input to the
  stepper, not the wall-clock duration.
- **Scene switching is a build/install split under a ranked lock order.**
  `_build_bundle` does every fallible thing (YAML load, library resolve-or-build,
  `TorchScene`, goniometer) **off-lock and writes nothing to `self`**;
  `_install_bundle` writes `self` and **cannot raise**. A failed switch therefore
  leaves the old scene bit-for-bit intact and there is no rollback path.
  The order is **`_anim_cv > _scene_lock > _gonio_lock`, with `_frame_cv` a
  leaf**, and the install nests: `/move`, `/recenter` and the animator read the
  scene's camera *and* `_target_pose` under `_anim_cv`, so both must become new
  in one instant (hampton 0.0074 mm/px vs mitegen 0.001 — a torn read is a 7.4×
  error, silently clamped). `_scene_lock` is an **RLock** (`_snapshot_gonio` is
  called both inside and outside `_render_now`'s hold) and is taken in
  `_render_now`, **never** in `_render_frame`, which the single-flight tests
  replace wholesale. **`_servable` acquires nothing — its caller must hold
  `_scene_lock`**: `_set_pose_instant` calls it from inside `_gonio_lock`, so a
  self-locking version deadlocks against the renderer.
  `tests/test_server_lock_order.py` enforces the order **statically**, because
  the inversions are call-mediated and `threading.Condition` wraps an `RLock`
  (so a re-entrant `_anim_cv` would silently succeed rather than hang).
  The goniometer is **rebuilt, never reassigned** — it captures `scene.geometry`
  by reference — and `_scene_gen` stops a cancelled animation's speed/heading
  handoff (the one thing it does write) crossing into the new scene.
- **A stale frame library is served, not rebuilt.** `library_status` splits
  `is_current`'s single bool into `current` / `stale` / `missing`; the switch
  path never calls `ensure_library`, because that rebuilds anything not current
  and `mitegen_200um` is stale only by two build keys — a ~1.9 h rebuild of
  frames already on disk. Builds happen only on an explicit `build=preview|full`
  and are refused without CUDA.
- **Click-to-recentre:** the browser sends the click as a **fraction** `fx,fy ∈
  [0,1]` of the displayed image (taken from `cam.getBoundingClientRect()`); the
  server scales by the true camera W/H.  Do **not** map clicks via
  `<img>.naturalWidth` — it is unreliable/0 for an MJPEG stream and breaks when
  the view is CSS-scaled.  Clicks are captured by a dedicated transparent
  `.clicklayer` (the streaming `<img>` may not deliver clicks reliably).

  **⚠ KNOWN BUG (unresolved, paused — see [[project_loop_sim_phase2]] for the resume
  plan):** in the live browser, click-to-recentre lands ~100–200 px off, sometimes
  in the wrong direction, **non-deterministically**.  What is RULED OUT:
  `recenter_target` is proven correct — an arbitrary point and a rendered feature
  centre to **0.0 px** through both the X-ray parallel projection AND the optical
  renderer (with and without refraction), when the pose handed to it matches the
  rendered pose; and `view ≈ img` (641×481 vs 640×480 is only the 1 px border).
  Leading hypothesis: a **frame/pose mismatch** — the displayed MJPEG frame lags
  the live goniometer pose (n_cond=7 render ~1 s + stream buffering), so
  `_command_recenter` (which reads the LIVE pose) recenters against a pose that
  doesn't match the frame the user clicked.  Offline tests can't catch this
  because they pass a matching pose.  First thing to try next: render at
  **n_cond=1** (minimal latency) and see if recentre becomes accurate → confirms
  the lag race.  A `#dbg` readout in `static/index.html` reports `view/img` sizes
  and the per-click `fx,fy` (debug scaffolding — remove once fixed).

## Key API: next_interface()

`Scene.next_interface(origins, dirs, t_min=1e-6)` returns:
- `t` — (N,) distance to next interface (inf = exited scene)
- `normals` — (N, 3) outward surface normal at the interface
- `mat_out_oi` — (N,) int: object index of material **after** crossing
  (-1 = background/air; 0..n_obj-1 = `scene.objects[oi]`)

`_trace_rays` in `microscope.py` maintains `cur_mat_oi` (same index
convention) and looks up n, mu, color via pre-built numpy tables
(`mat_n_tab`, `mat_mu_tab`, `mat_col_tab`, index 0 = background,
index k = objects[k-1]).

## GPU intersection precision (float64)

The CUDA intersection path computes the cylinder/triangle quadratics in
**float64** (`tube.py`, `surface_mesh.py`).  It used to use float32, which
catastrophically cancelled in `c_ = baba*oaoa - baoa**2 - r²*baba`: with the ray
launched 50 mm upstream, `oaoa ≈ 2500` swamped the r² signal of a ~7.5 µm fiber,
so float32 produced off-surface hits and wrong normals → the "hairy/spikey"
fiber artifact.  Doing the quadratic in float64 fixes it, and the GPU render is
now **byte-identical** to the float64 CPU reference.  **Do not** down-cast the
intersection inputs or geometry to float32.

### Material-after-interface (mat_out_oi)

`next_interface()` picks the material a ray enters after a crossing with an
interval check on the already-computed `all_te`/`all_tx`:

```python
t_probe = best_t + 1e-4
inside  = (te_m < t_probe) & (t_probe < tx_m)
mat_out_oi = first object whose interval contains t_probe, else -1
```

**Do not** revert to `_obj_index_at_points_batch` probe rays — that was the
original broken path (50% failure from probe-point ambiguity).

## Comparison workflow

After any change to the renderer or scene, submit to SLURM:

```bash
sbatch run_gpu.slurm     # renders CPU+GPU at n_cond=1 and n_cond=7, PNG output
cat slurm_<jobid>.log    # check timing and diff stats
```

Acceptable thresholds (PNG, lossless, 704×480):

| n_cond | pixels>10 | notes |
|--------|-----------|-------|
| 1      | ≤ ~35 K   | baseline; nearly all diff pixels are >60 (TIR/NA flip) |
| 7      | ≤ ~60 K   | higher than n_cond=1 because 7 angles sample more edge cases |

Almost all differing pixels are binary flips (TIR or NA cutoff crossing due to
float32 geometry), not gradual noise.  The n_cond=7 count is ~1.6× the
n_cond=1 count because independent condenser rays can each flip a different set
of edge pixels.
