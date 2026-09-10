# RUNBOOK: loop-sim (xtal-loop-sim)

Environment → run → verify → deploy → rollback. `../README.md` is the user guide (full
pipeline, every CLI flag, all HTTP endpoints); this file is the from-nothing path and the
operational detail the README leaves out. Paths are relative to the repo root.

## Environment (from nothing)

**On the beamline, the interpreter already exists. Use it:**

```
/programs/pytorch/envs/pt/bin/python
```

It carries numpy, scipy, PIL, pyyaml, and PyTorch+CUDA. **Do not use `python3` or
`/usr/bin/python3`**: the system Python is 3.6, root-owned, and has none of the required
packages. Every command below spells this as `$PY`:

```bash
PY=/programs/pytorch/envs/pt/bin/python     # beamline
```

**On a machine without that env** (a dev box), build one. Python 3.11 works; the GPU path
needs a CUDA-capable torch matching the local driver:

```bash
conda create -n loopsim python=3.11
conda activate loopsim
pip install -r requirements.txt       # numpy, scipy, Pillow, pyyaml, tifffile
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pytest                    # for the verify command; not in requirements.txt
PY=$(conda run -n loopsim which python)   # or the env's python path directly
```

`requirements.txt` deliberately omits torch: the CPU reference path (`--device cpu`)
runs without it, and the right torch build is site-specific. Confirm the GPU is visible:

```bash
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

There are no environment variables to set and no `.env`.

## Run

```bash
# Render a bundled scene (tube-based; this is the one that exercises the GPU path)
$PY render.py scene_files/hampton_300um.yaml --n-cond 7 --output /tmp/out.jpg

# Same on the GPU (GPU-resident engine, engine_torch.py)
$PY render.py scene_files/hampton_300um.yaml --n-cond 7 --device cuda --output /tmp/out.jpg

# Rotate about the spindle (rotx for the bundled scenes) / translate
$PY render.py scene_files/hampton_300um.yaml --rotx 45 --n-cond 1
$PY render.py scene_files/hampton_300um.yaml --tx 0.05 --ty -0.02 --n-cond 7

# Live AXIS-compatible camera server (uses the GPU-resident engine when CUDA is present)
$PY -m loop_sim.server.camera_server --scene scene_files/hampton_300um.yaml --port 8080
```

Then open `http://<host>:8080/` for the control page, or point an AXIS consumer at
`http://<host>:8080/axis-cgi/mjpg/video.cgi`. `../README.md` documents every flag and
endpoint; `loop_sim/server/camera_server.py`'s docstrings explain the animation and concurrency model.

Notes that save time:

- **`scene.yaml` / `loop.yaml` are pipeline outputs, gitignored and not shipped.** The
  three scenes in `scene_files/` are complete; build your own with the README's "Full
  pipeline".
- **`--device cuda` runs the GPU-resident engine** (`engine_torch.py`), the same one the
  camera server uses. `render.py` builds a `TorchScene` and calls `render_torch` for any
  bundled scene, tube or mesh, not just `hampton_300um`. Falls back to CPU when no CUDA
  device is visible.
- Run from the repo root.
- Which motor is the spindle is scene-dependent: `rotx` for the bundled scenes.

### Frame libraries (pre-computed rotation sweeps)

A frame library is a full 360° spindle sweep rendered once and replayed, so the camera
responds instantly and nothing is rendered at request time. Libraries are **tracked in
git** (part of the deliverable, not build output) and live in
`frame_library/<scene_stem>/` alongside a `manifest.json`.

```bash
# build (or refresh) one scene, and every bundled scene
$PY -m loop_sim.library --scene scene_files/hampton_300um.yaml
$PY -m loop_sim.library --all
$PY -m loop_sim.library --scene <s>.yaml --force        # rebuild regardless
```

Useful flags: `--step` (degrees between frames, default 1.0 → 360 frames),
`--supersample` (render this many times finer than the camera pixel; default 4, and the
hard ceiling on zoom-in), `--pan-mm` (sample travel to allow beyond the scene and the
centred field of view, default 0.6), `--axis` (spindle motor, default `rotx`), `--n-cond`,
`--format` (default `png`, lossless), `--psf` (default `on`), `--quality` (JPEG only),
`--tile-size` (default `auto`), `--vram-fraction` (default 0.80), `--device`.
Every lever, with defaults and what it costs, is tabulated under "Every lever" below.

**Only `mitegen_200um` needs `--supersample 1` on a bare launch**: `hampton_300um` and
`hampton_300um_realistic` are safe bare. The flag passed at launch must match the value
the library on disk was actually built at (`manifest.json`'s `supersample` field); a
mismatch rebuilds the library before the socket binds. `python -m loop_sim.library
--scene <s>.yaml` with nothing else confirms this: it is a no-op when the library is
current.

A rebuild **deletes the manifest first** and overwrites frames in place, so a server
launched at that scene mid-rebuild finds no library. All frames are tracked, so
`git checkout -- frame_library/<scene>/` recovers the previous one, for any of the three
scenes, as long as it is caught before the working tree is committed.

| scene | supersample | frames | size |
|---|---|---|---|
| `hampton_300um` | 4 | 360 | 15 MB |
| `hampton_300um_realistic` | 4 | 360 | 14 MB |
| `mitegen_200um` | 1 | 360 | 9 MB |

~37 MB total, all `current`.

Re-running is a **no-op when the library is current**: the manifest stores a SHA-256 of
the scene YAML, a SHA-256 of the **renderer source** (`render_sha`), *and* the build
parameters, so an edited scene, an edited tracer or a different `--supersample` rebuilds
automatically. From Python, `ensure_library(scene_path)` does the same and returns the
manifest; `frame_for_angle(manifest, deg)` picks the frame and `pose_crop(manifest, tx,
ty, tz, angle_deg, zoom)` gives the crop box, output size and defocus blur.

Notes:

- **The window is measured from the scene, not centred on the origin** (`content_window()`
  scouts a coarse wide sweep first). A symmetric margin would leave most of a long thin
  mount, like hampton's 6.7 mm pin in a 4.7 mm field, unrendered, and panning would scroll
  into blank background. Served: the spindle axis (quantised to `--step`), `tx`/`ty`/`tz`
  as a crop (`tz` becomes a Gaussian defocus blur), and `zoom` between the window floor
  and `--supersample`. `roty`/`rotz` are **not** covered: `--axis` other than `rotx` is
  refused rather than silently building a geometrically wrong library.
- **Out-of-range requests are refused, not clamped**, except in the live server, which
  clamps (and prints what it clamped) so it keeps serving, sliding the crop rather than
  squeezing it. Squeezing would alter the aspect ratio.
- **Which motor is lateral depends on φ.** At φ=0 `ty` moves the image vertically and
  `tz` is pure defocus; at φ=90 they swap. Handled in `pose_crop`, the single easiest
  thing to get backwards.
- **Pick `--supersample` per scene, from the camera's own sampling.** The right value is
  where the template pitch reaches the objective's Nyquist limit, `0.61λ/NA / 2`:

  | scene | pixel | NA | Nyquist | camera is… | supersample |
  |---|---|---|---|---|---|
  | `hampton_300um` | 7.4 µm | 0.10 | 1.68 µm | under-sampling 4.4× | **4** |
  | `mitegen_200um` | 1.0 µm | 0.10 | 1.68 µm | already over-sampling 1.7× | **1** |

  Going beyond that magnifies resolution the optics cannot deliver. It also costs: the
  template grows with the square. For a scene whose content is wider than its field
  (mitegen's is, 1.10 mm against 0.48 mm) the useful zoom direction is *out*, which the
  window already provides, not *in*.
- **A build sizes itself to the GPU and refuses rather than dying half-way**, with no need to
  pass `--tile-size` or `--vram-fraction` on any card:

  1. **The budget comes from free VRAM, not the card's total**: on a shared node like
     voltron another tenant's usage reduces yours instead of surfacing as an
     out-of-memory error mid-build. A gigabyte is held back for CUDA context and
     allocator slack.
  2. **The budget is a hard limit**: an overrun raises rather than silently spilling to
     host RAM (see "Dev-environment caveat").
  3. **A preflight renders one frame and reads the real peak before the build commits**,
     printing a line like
     `[preflight] 1396x644 n_cond=7: 2.26 GB peak against a 10.82 GB budget,
     tile 899024 -- fits`. If it doesn't fit, the trace tile shrinks (no pixel change) and
     re-probes.
  4. **If it still doesn't fit, the build refuses before rendering anything**, naming the
     largest workable `--supersample`:
     `... needs more memory than this GPU has: peak 12.4 GB against a 8.8 GB budget.
     Tiling cannot help -- the cost that does not fit scales with OUTPUT PIXELS. At this
     scene's settings --supersample 4 is the largest that fits (you asked for 8).`
     A too-large request is never silently downgraded.

  `LOOPSIM_VRAM_BUDGET_GB` overrides the measured budget: leave room for another tenant
  on a shared card, or rehearse a smaller card's sizing before deploying to it
  (`=12` on a 16 GB box mimics the TITAN V).

  **Watch `nvidia-smi`, not torch's own counter**: the caching allocator reserves and
  never returns, so `max_memory_allocated()` under-reports what the card is actually
  holding. **Per-frame cost is strongly pose-dependent, so never time one frame** to
  estimate a build: the progress line prints a cumulative average, not a per-frame time.
- **Raising `--supersample` on a mesh scene costs time, not correctness.** See
  DECISIONS §2026-08-12 supersample 4 for the measured cost table.
- **Under WSL2 there is no OOM to catch** (the driver spills to host RAM instead of
  raising), so the builder also warns when frames slow down persistently; see
  "Dev-environment caveat".

### X-ray radiograph library

Same mechanism as the optical frame library, in its own module
(`loop_sim/library/xray_library.py`), root (`xray_library/`, not `frame_library/`) and
`render_sha` scope, so an X-ray-only change never costs an optical rebuild or vice versa:

```bash
python -m loop_sim.library --modality xray --scene scene_files/hampton_300um.yaml
python -m loop_sim.library --modality xray --scene scene_files/hampton_300um.yaml --supersample 4
```

Simpler than the optical build: no `--n-cond`, `--psf`, or `--format`/`--quality` (always
lossless 16-bit greyscale: 8-bit would quantize contrast into ~50 levels), and no
depth-blur approximation (a collimated beam's Beer-Lambert integral doesn't change with
`tz`).

**Serving is read-only and never builds implicitly.** `--xray-library-root` points
`camera_server` at a library root; a complete library there (current *or* stale, stale is
served as-is) is served from it in single-digit ms and **prewarmed into RAM at boot and on
every scene switch** (`XrayTemplateSource.prewarm()`), which is what `GET /xray-stream`
(started by `POST /stream-mode?mode=radiograph`, stopped by `mode=microscope`) needs to run
near the optical stream's own frame rate (~28 fps in motion) instead of paying a disk
decode per frame. If no library exists, both `/xray` and `/xray-stream` keep rendering
live. Building one is always the explicit CLI command above, never a server side effect.

All three shipped scenes have a current X-ray library, built with the illustrative
`mu_xray` values settled 2026-08-18 (see DECISIONS.md), and a future switch to
literature-real coefficients would need a rebuild:

| scene | build time | size |
|---|---|---|
| `hampton_300um` | 783 s (13.1 min) | 12 MB |
| `hampton_300um_realistic` (flagship, mesh) | 4890 s (81.5 min) | 10 MB |
| `mitegen_200um` | 365 s (6.1 min) | 18 MB |

~39 MB total, tracked and committed. `xray_library/` needs the same `.gitignore`
re-include as `frame_library/` (`!xray_library/**/*.png`,
`!xray_library/**/manifest.json`): without it `git add -A` silently ships an empty
library.

### Switching scenes on a running server

The control page carries a tab per scene in `scene_files/`; clicking one swaps
the sample live without restarting or dropping the MJPEG stream. The pose resets
to home. A millimetre does not mean the same thing in two scenes whose pixel
sizes differ 7.4×. Same thing from a terminal:

```bash
curl -X POST 'http://host:8080/scene?path=mitegen_200um'   # 202 accepted
curl -s http://host:8080/scene                             # progress + errors
curl -s http://host:8080/scenes                             # library state of each
```

Each tab is badged with that scene's library state, and only one of them stops a
switch:

- **(no badge)**: matches current build settings; switches immediately.
- **`stale`**: complete and servable, built with different settings, e.g. before a new
  build key existed. **Switches immediately** and the page names what differs; **never
  rebuilt automatically**: that would cost a full build nobody asked for.
- **`preview`**: only a coarse on-demand library exists (5° steps, 1× zoom).
- **`no library`**: nothing to serve; the page offers a preview or full build.

**Builds are refused without CUDA** (~179 s/frame → hours for a full library), in
the viewer *and* the CLI. The viewer has no override by design; build offline on
a GPU host instead, then switch:

```bash
python -m loop_sim.library --scene scene_files/<scene>.yaml            # full, tens of minutes
python -m loop_sim.library --scene scene_files/<scene>.yaml --preview  # coarse, minutes
python -m loop_sim.library --scene <scene>.yaml --allow-cpu            # if you really mean it
```

Two consequences:

- **A switch waits for the in-flight frame**, so the stage briefly stops responding: ~70
  ms on the default template path, up to ~1 s with `--templates off`, one whole frame
  (~18 s) on `--templates off --engine numpy`.
- **`--templates off --engine torch` loses the compiled preview after the first switch**
  (6.3 fps eager instead of 11.9). `torch.compile` warmup only runs at startup; the
  server prints a `[compile-preview]` line rather than degrading silently. Restart to get
  it back. The default template path holds no GPU state and is unaffected.
- **With `--templates off`, a switch holds both the old and the new `TorchScene` until
  the install completes**, so peak VRAM is the sum. That is the price of having no rollback
  path; it does not arise on the default path.

### On voltron (the beamline GPU node)

GPU rendering requires CUDA, which lives on voltron. Submit from the local machine:

```bash
sbatch run_gpu.slurm      # gpu partition, gres=gpu:1
squeue --job <jobid>
cat slurm_<jobid>.log
```

**Never set `--time` in these job scripts**: this queue has no time limits and the flag
gets jobs cancelled early. If you need an interactive session, voltron's login shell is
tcsh and does not parse `&&`: write a bash script and run
`ssh voltron "cd $PWD ; bash script.bash"`.

## Every lever

Everything a user can turn, with its default and what it does. **The right-hand column is
the one to read before a long run:** a lever marked *rebuilds library* changes the stored
template pixels, so touching it invalidates a frame library and the next server launch
silently regenerates it (tens of minutes to hours; DECISIONS §2026-08-11 mesh cull has
the per-scene build times).

### `render.py`: offline single frame

| Flag | Default | Effect |
|---|---|---|
| `<scene.yaml>` | none | scene to render (positional) |
| `--tx` `--ty` | from the scene's `motor:` block | stage translation, mm. The CLI overrides the YAML; there is no `--tz` here |
| `--rotx` `--roty` `--rotz` | 0 | rotation, degrees; `rotx` is the spindle for the bundled scenes |
| `--n-cond` | 1 | condenser angles per pixel; 7 = soft NA edges, >7 buys little |
| `--device` | `cpu` | `cuda` runs the GPU-resident engine (`engine_torch`), same one the camera server uses; falls back to CPU with no CUDA visible |
| `--output` | `<scene_basename>.jpg` | output JPEG path |

`render.py` exposes no `--zoom`; set `zoom` in the scene's `motor:` block, or use the
server, whose `/motor` endpoint takes all seven axes.

### `python -m loop_sim.server.camera_server`: the live/pretend camera

| Flag | Default | Effect |
|---|---|---|
| `--scene` | `scene_files/hampton_300um.yaml` | scene to serve |
| `--host` / `--port` | `0.0.0.0` / 8080 | bind address |
| `--templates` | `on` | serve from the pre-computed sweep (no GPU at runtime). `off` raytraces every frame, the correctness reference |
| `--fps-limit` | 30.0 | MJPEG wire-rate ceiling, a hard clamp on how fast frames go out |
| `--n-cond` | 7 | condenser angles for settled frames |
| `--jpeg-quality` | 85 | quality of frames the server **sends**. Not the stored template: see `--template-quality` |
| `--engine` | `auto` | `torch` (GPU-resident) / `numpy` (reference) / auto-detect |
| `--preview-mode` | `on` | approximate frames while moving, exact on settle |
| `--compile-preview` | `on` | `torch.compile` the preview path (CUDA + preview only) |
| `--settle-delay` | 0.5 s | quiet time after a `/motor` set before the exact frame renders |
| `--camera-emulation` | `on` | map transmittance through the illumination field, so an empty field reads ~0.60 and an opaque body ~0.18 rather than pure black/white. **Serve-time only: no library rebuild** |
| `--mono` | `off` | `on` collapses to grey before the camera stage. Colour here is an absorption spectrum, so a scene declaring a crystal `[0.7,0.9,1.0]` renders it blue, and `--mono on` masks that. To strip colour at the scene level instead, set `colour: [1,1,1]` with the absorption in `mu_optical`, which *rebuilds every library*. Ignored when `--camera-emulation off` |
| `--pin-streak` | `on` | draw the specular glint a real machined pin carries along its shank, projected from the scene (`renderer/pin_projection.py`) through the current pose, exact at any zoom, crop or angle, absent when the pin is out of view. Only objects the code declares shiny get one (`SHINY`: `pin`+`metal`), so `mitegen_200um` never does. Ignored when `--camera-emulation off` |
| `--sensor-pitch` | `on` | deliver on the real camera's **704×480** raster. BL831 pixels are 1.11 non-square and the tracer's are square, so a consumer applying dcss's µm-per-pixel constant to a 640-wide render reads 10% wide. `off` serves the render's own square pixels. Template path resamples in PIL, not `field.to_sensor` (6.7 → 1.3 ms, agrees to 1 level) |
| `--template-cache` | `auto` | decoded templates held in RAM. `auto` takes as much of the library as half of available memory allows; `off` caps at 8 frames; an integer pins the count. **All-or-nothing per sweep**: if the host cannot hold a full revolution, `auto` **declines** rather than half-filling: LRU against a cyclic sweep evicts each frame just before it comes round again, so a partial cache is worth zero rather than a share |
| `--supersample` | builder default (4) | *rebuilds library* |
| `--template-format` | builder default (`png`) | *rebuilds library* |
| `--template-quality` | builder default (90) | JPEG quality of **stored** templates; ignored for png. *rebuilds library* |
| `--scene-dir` | repo `scene_files/` | which `*.yaml` are offered for runtime switching on `/scenes` |
| `--library-root` | repo `frame_library/` | frame-library root to serve from and report on |
| `--preview-root` | repo `frame_library_preview/` | where on-demand **preview** libraries are written. Separate from `--library-root` deliberately: building into the live root overwrites frames the serving `TemplateSource` is caching by filename |
| `--xray-library-root` | repo `xray_library/` | X-ray radiograph library root `/xray` and `/xray-stream` both serve from (and prewarm from at boot/switch). **Read-only**: unlike `--library-root`, a missing or stale library here is never built implicitly; both endpoints just keep rendering live (see "X-ray radiograph library" above) |

### `python -m loop_sim.library`: build a frame library

| Flag | Default | Effect |
|---|---|---|
| `--modality` | `optical` | `xray` builds the radiograph library instead; see "X-ray radiograph library" above. Every flag past this row is optical-only and ignored: with `--modality xray`, only `--scene`/`--all`, `--root`, `--step`, `--supersample`, `--pan-mm`, `--axis`, `--device`, `--force` apply (no `--n-cond`/`--format`/`--psf`/`--quality`) |
| `--scene` / `--all` | none | one scene, or every `scene_files/*.yaml` |
| `--root` | `frame_library/` | output directory |
| `--step` | 1.0° | degrees between frames → 360 frames. *rebuilds library* |
| `--supersample` | 4 | render this many times finer than the camera pixel; the hard ceiling on zoom-in. *rebuilds library* |
| `--pan-mm` | 0.6 mm | travel to allow beyond the scene and the centred field. *rebuilds library* |
| `--n-cond` | 7 | condenser angles. *rebuilds library* |
| `--axis` | `rotx` | spindle motor; anything else is refused rather than built wrong. *rebuilds library* |
| `--format` | `png` | stored template format. png is lossless **and** smaller here. *rebuilds library* |
| `--psf` | `on` | bake the objective diffraction PSF into the templates. *rebuilds library* |
| `--quality` | 90 | JPEG quality; ignored when `--format png`. *rebuilds library* |
| `--tile-size` | `auto` | rays per trace pass; `auto` measures the size by trial renders. Does not change pixels. Note `render_torch`'s own default is different and cheaper: it *calculates* the tile from mesh face count and free VRAM with no trial renders (DECISIONS.md §2026-08-07) |
| `--vram-fraction` | 0.80 | share of free VRAM the auto tile may use. Does not change pixels |
| `--device` | auto | `cuda` when available |
| `--force` | off | rebuild even if current |
| `--preview` | off | build the same coarse library the camera server builds on demand (5° steps, 1× supersample, n_cond 1 → 72 frames) into `frame_library_preview/`. Minutes instead of ~45 min; zoom capped at 1× |
| `--allow-cpu` | off | permit a build with no CUDA. Without it a CPU build is **refused**: ~179 s/frame is ~3.6 h for a preview and ~18 h for a full library. `--device cpu` needs this flag too |

### Scene YAML `camera:` block

| Key | Example | Effect |
|---|---|---|
| `width` / `height` | 640 / 480 | camera resolution in pixels |
| `pixel_size` | 0.0074 mm | mm per pixel at the sample. Sets the field of view and, with NA, how visible the PSF is |
| `na_objective` | 0.10 | collection gate **and** the PSF width (σ = 0.21 λ/NA) |
| `na_condenser` | 0.07 | illumination cone; also drives the template defocus blur |

Per-material properties live on each object: `n` (refractive index), `mu_optical`
(absorption), and colour. Object **order matters**: the list is priority-ordered and the
first entry wins at any point in space, so a crystal must precede the droplet that
contains it, or it renders as solvent.

### Environment

| Lever | Value | Effect |
|---|---|---|
| interpreter | `/programs/pytorch/envs/pt/bin/python` on the beamline | the only Python with numpy/scipy/PIL/pyyaml/torch. The system 3.6 has none of them |
| CUDA present | auto-detected | picks the GPU-resident engine; absent falls back to numpy (minutes per frame) |
| `CC` / `CXX` | devtoolset-7 on voltron | required for `torch.compile`; without it the server silently drops to eager and misses 10 fps |

---

## Verify

```bash
$PY -m pytest tests/ -q
```

Pass is every test green, about 8 minutes on an RTX 4080 SUPER; the count grows with the
work, so a run that collects fewer tests than the previous one is the signal worth
chasing. Benign `divide by zero`/RuntimeWarnings from the numpy reference
primitives are expected in the output, not a failure. On a CPU-only box the CUDA-gated
parity tests **skip** rather than fail, so a green run there is a weaker check. It does
not exercise the GPU engine at all.

The suite covers GPU↔CPU render parity (the correctness fix), torch↔numpy shape parity,
beam attenuation, the compiled preview path, and the server's settle/single-flight
behavior.

For a render-level check after touching the renderer or a scene, use the SLURM comparison
job (`sbatch run_gpu.slurm` renders CPU+GPU at n_cond 1 and 7 and reports diff stats);
DECISIONS §2026-05-22 has the thresholds measured on the retired legacy path; a fresh run on the resident engine should read far tighter.

Benchmarking the **serve** path (no GPU, no socket, no display, safe on a busy shared
node, and verified to import torch not at all):

```bash
$PY bench_serve.py --scene scene_files/hampton_300um_realistic.yaml --frames 40
```

On **voltron**, with all eight cards busy, use the **stock** interpreter, *not* the
`~/projects/loopsim-torch26` venv from "Deploy on the TITAN V" below. That venv exists for
the compiled GPU preview path; this benchmark imports torch not at all, so it needs no
venv, no `CC`/`CXX` and no devtoolset:

```tcsh
cd ~/projects/loop_sim_MINE/xtal-loop-sim
/programs/pytorch/envs/pt/bin/python bench_serve.py --json serve.json
```

It prints the three regimes, a per-stage split, a comparison against the recorded
pre-crop numbers for that host, and a **GO/NO-GO against the 10 fps goal** (exit 0 / 1).
If the host's libraries still store the full window it says so and points at `--recrop`;
otherwise decode reads ~7x slower with nothing to explain why.

**slew** = spindle turning, every frame a fresh decode (the worst case, and what grades
the host); **pan** = fixed angle, decode served from cache; **hold** = the floor.
**Read `slew_warm`, not `slew`.** The server pre-warms the whole library at boot, so a
spindle slew never touches disk once it is up; `slew_warm` is what an operator gets and
what the GO/NO-GO verdict grades. `slew` benchmarks a cold decode, which the server pays
once at startup instead. It is the noisier number, not the one to read.

All three beamline hosts (dataserver3, voltron, gateway) clear the 10 fps goal warm; see
DECISIONS §2026-08-14.

Benchmarking the **render** path: `bench_frame.py` (flags `--compiled`, `--fp32`;
`--modality xray` times `render_xray_torch`/`render_xray_numpy` instead (no
n_cond/PSF/compiled sweep, since the X-ray tracer has none of those); soak the live server with
`soak_server.py`, which lives **outside this repo** in the analysis tree at
`/home/jadoughty/projects/loop_sim_MINE/investigation/2026-07_scene_and_perf_harnesses/`.
That tree is mirrored to the gateway alongside the repo but is **not versioned**, so it
will not come with a `git clone`; the repo is complete without it.

## Other scripts

Root-level scripts, run from the repo checkout with `$PY`:

- **`test_gpu.bash`**: renders `hampton_300um` on CPU then GPU (`--device cpu` /
  `--device cuda`), timed. The base A/B smoke test.
- **`test_optim.bash`**: renders on GPU, timed, and diffs the JPEG against a prior
  `scene_gpu.jpg` to catch a GPU-path regression.
- **`debug_optim.bash`**: renders CPU (`n_cond=7`, reference) then GPU, and reports
  max/mean pixel difference and the count of pixels off by more than 10.
- **`check_diff.bash`**: three-way diff of `scene_ref.jpg` / `scene_gpu.jpg` /
  `scene_optim.jpg` (CPU vs. GPU-original vs. GPU-new); run after the scripts above have
  produced those files.
- **`profile_render.py`** / **`profile_render.bash`**: cProfile the CPU renderer on
  `hampton_300um.yaml` to find hotspots. The `.bash` wrapper installs missing deps first,
  then runs the `.py`.
- **`profile_gpu.py`** / **`profile_gpu.bash`**: same profiling on the GPU path
  (`device='cuda'`), with a warm-up render discarded before timing.
- **`make_beam_image.py`**: generates a synthetic 16-bit PNG beam-profile image
  (Gaussian FWHM, optional pinhole mask) for the X-ray beam entity, with pixel size
  embedded as a PNG text chunk.

## Deploy

There is no install step and no service: loop-sim runs in place from a checkout, as a
CLI (`render.py`), a SLURM job (`run_gpu.slurm`), or the camera server started on a host
reachable by whatever consumes the stream. To "deploy" a change:

1. Confirm `pytest tests/` green on a CUDA box (a CPU-only run skips the GPU tests).
2. Land the code on the target checkout (git, or a copy).
3. Restart the camera server if one is running (it holds the scene + compiled kernels in
   memory; there is no reload).

**Note on branch state:** this work lives on branch `performance-correctness-optimizations`,
not pushed to GitHub. James owns that decision (see HANDOFF "Current state"). Measure how
far ahead of `master` with `git rev-list --count master..HEAD`; don't quote a number here.
Work from `master` or this branch: the GitHub default `main` is a stale, divergent "Initial
commit".

### Deploy on the TITAN V (voltron)

**The viewer does not need any of this**: serving from templates imports torch *not at
all*. To run the viewer on voltron:

```tcsh
cd ~/projects/loop_sim_MINE/xtal-loop-sim
/programs/pytorch/envs/pt/bin/python -m loop_sim.server.camera_server --scene scene_files/hampton_300um_realistic.yaml --port 8080
```

No venv, no `CC`/`CXX`, no devtoolset, no GPU pinned, no compile warmup. Same for
`bench_serve.py`.

Everything below is for **rendering** on the GPU: building libraries, `--templates off`,
and `acceptance_voltron.py`. The 10 fps live-render path is **measured on the real TITAN
V at 11.9 fps**, but only with the stack below. The beamline's default environment (the pt
env's torch 2.0.1, system gcc 4.8.5) cannot run `torch.compile` and falls back to eager at
~6.3 fps. Voltron's login shell is **tcsh** (`setenv`, not `export`); call the venv's
python by full path because venv `activate` is a bash script.

**Scripted:** `setup_titan_v_env.bash` (repo root) runs steps 1–3 below as one idempotent
command: `bash setup_titan_v_env.bash` on voltron, `--force` to rebuild the venv,
`--skip-verify` to skip the final `acceptance_voltron.py` run. Manual steps below for
reference and troubleshooting.

```tcsh
# 1) a torch-2.6 venv (the pt env's torch 2.0.1 has an Inductor pkg_resources bug)
/programs/pytorch/envs/pt/bin/python3.10 -m venv ~/projects/loopsim-torch26
~/projects/loopsim-torch26/bin/python -m pip install --upgrade pip
~/projects/loopsim-torch26/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# pillow 12 has no glibc-2.17 wheel (RHEL7) and won't build on the old gcc -> pin 10.4.0
~/projects/loopsim-torch26/bin/python -m pip install numpy scipy "pillow==10.4.0" pyyaml

# 2) point Inductor at a modern compiler (system gcc 4.8.5 is too old -> stdatomic.h error).
#    devtoolset-7 (gcc 7.3.1) is enough; set these before launching, in the same shell:
setenv CC  /opt/rh/devtoolset-7/root/usr/bin/gcc
setenv CXX /opt/rh/devtoolset-7/root/usr/bin/g++

# 3) confirm the whole stack (voltron is a shared 8-GPU node; the harness auto-picks a free GPU)
cd ~/projects/loop_sim_MINE/xtal-loop-sim        # the repo's location on voltron
~/projects/loopsim-torch26/bin/python acceptance_voltron.py
```

`acceptance_voltron.py` prints a GO/NO-GO and writes `acceptance_report.json`; a GO means
compile actually engaged and beat eager. To launch the camera server on the **live-render**
path (`--templates off`, or a scene with no library yet), use the same venv with `CC`/`CXX`
still set and a free GPU pinned; for the template path use the stock interpreter above
instead:

```tcsh
setenv CUDA_VISIBLE_DEVICES 6      # a free card (check nvidia-smi first)
~/projects/loopsim-torch26/bin/python -m loop_sim.server.camera_server --scene scene_files/hampton_300um.yaml --port 8080 --templates off
```

Operational notes:

- **~1-2 min compile warmup** at server start: Inductor compiles the preview kernels once.
- **A build sizes itself to free VRAM and refuses rather than OOMing** (the preflight under
  "Frame libraries"); the droplet scene fits a 12 GB card (DECISIONS §2026-08-13 12 GB).
- **Pin `CUDA_VISIBLE_DEVICES` to a free GPU**: a busy card OOMs the mesh scene on arrival.
- If a run fails at *import* with `GLIBCXX...not found` (not a compile error), wrap the
  command in `scl enable devtoolset-7 "<command>"` so the runtime libraries match.

## Rollback

The renderer has no state and writes nothing outside its output files, so rollback is
just running older code:

- **A change made things wrong or slow:** `git checkout master` (or the previous commit)
  and re-run. `master` is the pre-Jacob baseline: correct on CPU, "hairy" fiber artifact
  on the GPU, no GPU-resident engine.
- **The compiled preview path misbehaves** (silent failure, or wrong frames during
  motion): start the server with `--compile-preview off`: preview frames then render
  eagerly. `--preview-mode off` goes further: every frame is exact full quality (slow,
  but no preview path at all).
- **Suspect the torch engine entirely:** force the numpy reference:
  `--engine numpy` on the server, or `--device cpu` for `render.py`. Slow (minutes/frame)
  but it is the reference implementation everything else is checked against.

## Dev-environment caveat (WSL2 + consumer GPU)

If a run on a WSL2 box slows 10–50× instead of failing, suspect **VRAM spill**: past the
card's VRAM the Windows NVIDIA driver silently spills into system RAM rather than raising
CUDA OOM; the job crawls and the whole desktop drags. Tell: `nvidia-smi` `memory.used`
pinned near the ceiling (≳15.5 GB on a 16 GB card) plus per-item time *degrading* over the
run. Fix by shrinking the working set (resolution, batch, `n_cond`) until it fits. This is
a Windows/WSL2 driver behavior, not a loop-sim bug; it matters here because the mesh scene
is already known to be close to the TITAN V's 12 GB (DECISIONS.md).
