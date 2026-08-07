# RUNBOOK — loop-sim (xtal-loop-sim)

Environment → run → verify → deploy → rollback. `../README.md` is the user guide (full
pipeline, every CLI flag, all HTTP endpoints); this file is the from-nothing path and the
operational detail the README leaves out. Paths are relative to the repo root.

## Environment (from nothing)

**On the beamline, the interpreter already exists — use it:**

```
/programs/pytorch/envs/pt/bin/python
```

It carries numpy, scipy, PIL, pyyaml, and PyTorch+CUDA. **Do not use `python3` or
`/usr/bin/python3`** — the system Python is 3.6, root-owned, and has none of the required
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

`requirements.txt` deliberately omits torch — the CPU reference path (`--device cpu`)
runs without it, and the right torch build is site-specific. Confirm the GPU is visible:

```bash
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

There are no environment variables to set and no `.env`.

## Run

```bash
# Render a bundled scene (tube-based; this is the one that exercises the GPU path)
$PY render.py scene_files/hampton_300um.yaml --n-cond 7 --output /tmp/out.jpg

# Same on the GPU (legacy per-object CUDA path)
$PY render.py scene_files/hampton_300um.yaml --n-cond 7 --device cuda --output /tmp/out.jpg

# Rotate about the spindle (rotx for the bundled scenes) / translate
$PY render.py scene_files/hampton_300um.yaml --rotx 45 --n-cond 1
$PY render.py scene_files/hampton_300um.yaml --tx 0.05 --ty -0.02 --n-cond 7

# Live AXIS-compatible camera server (uses the GPU-resident engine when CUDA is present)
$PY -m loop_sim.server.camera_server --scene scene_files/hampton_300um.yaml --port 8080
```

Then open `http://<host>:8080/` for the control page, or point an AXIS consumer at
`http://<host>:8080/axis-cgi/mjpg/video.cgi`. `../README.md` documents every flag and
endpoint; `../CLAUDE.md` explains the animation/concurrency model.

Notes that save time:

- **`scene.yaml` / `loop.yaml` are gitignored and not shipped.** A fresh clone has no
  `scene.yaml`, so the README's `render.py scene.yaml` quick-start needs one built first
  (README "Full pipeline") — or just render a complete bundled scene from `scene_files/`.
- **Only tube/mesh scenes have a CUDA path.** `hampton_300um.yaml` (tubes) exercises it.
  `mitegen_200um.yaml` is subtler than this runbook previously claimed: its `micromount` is
  a `ThinShell` wrapping an internal `SurfaceMesh`, so the **torch engine** does run the
  mesh path on it, but `render.py`'s legacy path never hands the shell a device, so there
  `--device cuda` really is a no-op and CPU/GPU agreement is byte-identical by construction
  rather than evidence the GPU ran. See docs/HANDOFF.md "Other traps".
- `render.py` inserts `/home/jamesh/projects/loop_sim/claude` on `sys.path`; harmless when
  absent. Run from the repo root.
- Which motor is the spindle is scene-dependent: `rotx` for the bundled scenes.

### Frame libraries (pre-computed rotation sweeps)

A frame library is a full 360° spindle sweep rendered once and replayed, so the camera
responds instantly and nothing is rendered at request time. Libraries are **tracked in
git** — they are part of the deliverable, not build output — and live in
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

Re-running is a **no-op when the library is current** — the manifest stores a SHA-256 of
the scene YAML *and* the build parameters, so an edited scene or a different
`--supersample` rebuilds automatically. From Python, `ensure_library(scene_path)` does the
same and returns the manifest; `frame_for_angle(manifest, deg)` picks the frame and
`pose_crop(manifest, tx, ty, tz, angle_deg, zoom)` gives the crop box, output size and
defocus blur.

Notes:

- **The window is measured from the scene, not centred on the origin.** A mount is long
  and thin — the hampton pin reaches x=6.7 mm against a 4.7 mm field — so a symmetric
  margin leaves most of the pin unrendered and panning scrolls in blank background.
  `content_window()` renders a coarse wide-field scout sweep to find where content
  actually is, and the sweep is then rendered at a fixed `tx` offset that centres it.
- **What the library serves:** the spindle axis (quantised to `--step`), `tx`/`ty`/`tz`
  as a crop, and `zoom` between the floor the window allows and `--supersample`. Depth
  translation becomes a Gaussian blur approximating condenser defocus. `roty`/`rotz` are
  **not** covered — one sweep is one axis, and `--axis` other than `rotx` is refused
  rather than silently building a geometrically wrong library. The server prints the
  actual zoom range at startup; `zoom_limits(manifest)` returns it. The floor is not
  simply `camera / template` — the window is anchored on the sample rather than centred,
  so at the home pose the camera runs out of room on the near side first.
- **Out-of-range requests are refused, not clamped**, except in the live server, which
  clamps so it keeps serving and prints what it clamped. Clamping slides the crop and
  never squeezes it: squeezing would change magnification per axis and silently alter the
  aspect ratio.
- **Which motor is lateral depends on φ.** The XYZ stage rides on the spindle, so at φ=0
  `ty` moves the image vertically and `tz` is pure defocus, while at φ=90 they swap. This
  is handled in `pose_crop`; it is also the single easiest thing to get backwards.
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
- **Cost** (RTX 4080 SUPER, n_cond 7): `hampton_300um` ~7.2 s/frame at `--supersample 4`
  (5578×2570), a 360-frame sweep in ~45 min. `mitegen_200um` ~18 s/frame at
  `--supersample 1` (1840×2296) — **slower despite being 3.4× smaller**, because it is a
  mesh scene and `TSurfaceMesh` has no AABB cull, so its tile is memory-capped at ~262 k
  rays. ~1.8 h for its sweep.
- **VRAM no longer limits resolution.** The trace is tiled and the tile is sized at
  runtime to fit `--vram-fraction` of free VRAM, so an 8 GB card renders the same
  templates as a 16 GB one, just in more passes. Per-ray results are tile-independent
  (tested byte-exact down to 1000-ray tiles), so the image does not depend on the tile.
  Tiling bounds the *trace* working set; the resident ray arrays are still O(W×H)
  (~1 GB at a 14 Mpx template) and no tile size shrinks them, so peak memory is reduced
  by tiling rather than made independent of resolution.
- **Mesh scenes cost `rays × faces × 160 B` of VRAM, and that sets the tile.** There
  is no AABB cull on the mesh path, so a droplet scene is far heavier than a tube
  one: a 2880-face droplet at 640×480 needs 19.8 GB in a single pass. The trace
  tile is therefore sized from the face count and free VRAM by default, which is
  what makes such a scene renderable at all (3.2 s at 3.6 GB here). Tube scenes
  carry no mesh term and are unaffected. If you add a much denser mesh and builds
  slow down, that is the tile shrinking to fit — the durable fix is giving
  `TSurfaceMesh` the AABB cull `TTube` already has.
- **The build raises on out-of-memory rather than quietly dropping resolution** — a
  library rendered at a degraded setting is indistinguishable from a good one once it is
  on disk. Under WSL2 there is no OOM to catch (the driver spills to host RAM instead), so
  the builder also warns when frames slow down persistently; see "Dev-environment caveat".

### Switching scenes on a running server

The control page carries a tab per scene in `scene_files/`; clicking one swaps
the sample live without restarting or dropping the MJPEG stream. The pose resets
to home — a millimetre does not mean the same thing in two scenes whose pixel
sizes differ 7.4×. Same thing from a terminal:

```bash
curl -X POST 'http://host:8080/scene?path=mitegen_200um'   # 202 accepted
curl -s http://host:8080/scene                             # progress + errors
curl -s http://host:8080/scenes                            # library state of each
```

Each tab is badged with that scene's library state, and only one of them stops a
switch:

- **(no badge)** — matches current build settings; switches immediately.
- **`stale`** — complete and servable, built with different settings. **Switches
  immediately**, and the page names what differs. This is normal, not a fault:
  `mitegen_200um` ships stale because its manifest predates the `format` and
  `psf` build keys, and its 360 frames are fine. **It is never rebuilt
  automatically** — that would cost ~1.9 h nobody asked for.
- **`preview`** — only a coarse on-demand library exists (5° steps, 1× zoom).
- **`no library`** — nothing to serve; the page offers a preview or full build.

**Builds are refused without CUDA** (~179 s/frame → ~3.6 h for a preview), in
the viewer *and* the CLI. The viewer has no override by design; build offline on
a GPU host instead, then switch:

```bash
python -m loop_sim.library --scene scene_files/<scene>.yaml            # full, ~45 min
python -m loop_sim.library --scene scene_files/<scene>.yaml --preview  # coarse, minutes
python -m loop_sim.library --scene <scene>.yaml --allow-cpu            # if you really mean it
```

Two consequences worth knowing:

- **A switch waits for the in-flight frame**, so the stage briefly stops
  responding: ~70 ms on the default template path, up to ~1 s with
  `--templates off`, and one whole frame (~18 s) on `--templates off --engine
  numpy`, where the stream is already that slow.
- **`--templates off --engine torch` loses the compiled preview after the first
  switch** (6.3 fps eager instead of 11.9). `torch.compile` warmup has to run
  single-threaded, which is only true at startup; the server prints a
  `[compile-preview]` line saying so rather than degrading silently. Restart to
  get it back. The default template path is unaffected — it holds no GPU state.
- **With `--templates off`, a switch holds both the old and the new `TorchScene`
  until the install completes**, so peak VRAM is the sum. That is the price of
  having no rollback path; it does not arise on the default path.

### On voltron (the beamline GPU node)

GPU rendering requires CUDA, which lives on voltron. Submit from the local machine:

```bash
sbatch run_gpu.slurm      # gpu partition, gres=gpu:1
squeue --job <jobid>
cat slurm_<jobid>.log
```

**Never set `--time` in these job scripts** — this queue has no time limits and the flag
gets jobs cancelled early. If you need an interactive session, voltron's login shell is
tcsh and does not parse `&&`: write a bash script and run
`ssh voltron "cd $PWD ; bash script.bash"`.

## Every lever

Everything a user can turn, with its default and what it does. **The right-hand column is
the one to read before a long run:** a lever marked *rebuilds library* changes the stored
template pixels, so touching it invalidates a frame library and the next server launch
silently regenerates it (~45 min for hampton, ~1.9 h for mitegen).

### `render.py` — offline single frame

| Flag | Default | Effect |
|---|---|---|
| `<scene.yaml>` | — | scene to render (positional) |
| `--tx` `--ty` | from the scene's `motor:` block | stage translation, mm. The CLI overrides the YAML; there is no `--tz` here |
| `--rotx` `--roty` `--rotz` | 0 | rotation, degrees; `rotx` is the spindle for the bundled scenes |
| `--n-cond` | 1 | condenser angles per pixel; 7 = soft NA edges, >7 buys little |
| `--device` | `cpu` | `cuda` uses the legacy per-object GPU path (tube/mesh scenes only) |
| `--output` | `<scene_basename>.jpg` | output JPEG path |

`render.py` exposes no `--zoom`; set `zoom` in the scene's `motor:` block, or use the
server, whose `/motor` endpoint takes all seven axes.

### `python -m loop_sim.server.camera_server` — the live/pretend camera

| Flag | Default | Effect |
|---|---|---|
| `--scene` | `scene_files/hampton_300um.yaml` | scene to serve |
| `--host` / `--port` | `0.0.0.0` / 8080 | bind address |
| `--templates` | `on` | serve from the pre-computed sweep (no GPU at runtime). `off` raytraces every frame — the correctness reference |
| `--fps-limit` | 30.0 | MJPEG wire-rate ceiling. This is a hard clamp: the old default of 5 capped the stream far below what templates can deliver |
| `--n-cond` | 7 | condenser angles for settled frames |
| `--jpeg-quality` | 85 | quality of frames the server **sends**. Not the stored template — see `--template-quality` |
| `--engine` | `auto` | `torch` (GPU-resident) / `numpy` (reference) / auto-detect |
| `--preview-mode` | `on` | approximate frames while moving, exact on settle |
| `--compile-preview` | `on` | `torch.compile` the preview path (CUDA + preview only) |
| `--settle-delay` | 0.5 s | quiet time after a `/motor` set before the exact frame renders |
| `--supersample` | builder default (4) | *rebuilds library* |
| `--template-format` | builder default (`png`) | *rebuilds library* |
| `--template-quality` | builder default (90) | JPEG quality of **stored** templates; ignored for png. *rebuilds library* |
| `--scene-dir` | repo `scene_files/` | which `*.yaml` are offered for runtime switching on `/scenes` |
| `--library-root` | repo `frame_library/` | frame-library root to serve from and report on |
| `--preview-root` | repo `frame_library_preview/` | where on-demand **preview** libraries are written. Separate from `--library-root` deliberately — building into the live root overwrites frames the serving `TemplateSource` is caching by filename |

### `python -m loop_sim.library` — build a frame library

| Flag | Default | Effect |
|---|---|---|
| `--scene` / `--all` | — | one scene, or every `scene_files/*.yaml` |
| `--root` | `frame_library/` | output directory |
| `--step` | 1.0° | degrees between frames → 360 frames. *rebuilds library* |
| `--supersample` | 4 | render this many times finer than the camera pixel; the hard ceiling on zoom-in. *rebuilds library* |
| `--pan-mm` | 0.6 mm | travel to allow beyond the scene and the centred field. *rebuilds library* |
| `--n-cond` | 7 | condenser angles. *rebuilds library* |
| `--axis` | `rotx` | spindle motor; anything else is refused rather than built wrong. *rebuilds library* |
| `--format` | `png` | stored template format. png is lossless **and** smaller here. *rebuilds library* |
| `--psf` | `on` | bake the objective diffraction PSF into the templates. *rebuilds library* |
| `--quality` | 90 | JPEG quality; ignored when `--format png`. *rebuilds library* |
| `--tile-size` | `auto` | rays per trace pass; `auto` measures the size by trial renders. Does not change pixels. Note `render_torch`'s own default is different and cheaper — it *calculates* the tile from mesh face count and free VRAM with no trial renders (DECISIONS.md §2026-08-07) |
| `--vram-fraction` | 0.80 | share of free VRAM the auto tile may use. Does not change pixels |
| `--device` | auto | `cuda` when available |
| `--force` | off | rebuild even if current |
| `--preview` | off | build the same coarse library the camera server builds on demand (5° steps, 1× supersample, n_cond 1 → 72 frames) into `frame_library_preview/`. Minutes instead of ~45 min; zoom capped at 1× |
| `--allow-cpu` | off | permit a build with no CUDA. Without it a CPU build is **refused**: ~179 s/frame is ~3.6 h for a preview and ~18 h for a full library. `--device cpu` needs this flag too |

### Scene YAML — `camera:` block

| Key | Example | Effect |
|---|---|---|
| `width` / `height` | 640 / 480 | camera resolution in pixels |
| `pixel_size` | 0.0074 mm | mm per pixel at the sample. Sets the field of view and, with NA, how visible the PSF is |
| `na_objective` | 0.10 | collection gate **and** the PSF width (σ = 0.21 λ/NA) |
| `na_condenser` | 0.07 | illumination cone; also drives the template defocus blur |

Per-material properties live on each object: `n` (refractive index), `mu_optical`
(absorption), and colour. Object **order matters** — the list is priority-ordered and the
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

**Pass = 62 tests green** (last run 2026-07-16: 62 passed in 67 s on an RTX 4080 SUPER;
43 warnings are expected — benign `divide by zero`/RuntimeWarnings from the numpy
reference primitives). On a CPU-only box the CUDA-gated parity tests **skip** rather than
fail, so a green run there is a weaker check — it does not exercise the GPU engine at all.

The suite covers GPU↔CPU render parity (the correctness fix), torch↔numpy shape parity,
beam attenuation, the compiled preview path, and the server's settle/single-flight
behavior.

For a render-level check after touching the renderer or a scene, use the SLURM comparison
job (`sbatch run_gpu.slurm` renders CPU+GPU at n_cond 1 and 7 and reports diff stats);
`../CLAUDE.md` "Comparison workflow" carries the acceptable thresholds.

Benchmarking: `bench_frame.py` (flags `--compiled`, `--fp32`); soak the live server with
`soak_server.py`, which lives **outside this repo** in the analysis tree at
`/home/jadoughty/projects/loop_sim_MINE/investigation/2026-07_scene_and_perf_harnesses/`.
That tree is mirrored to the gateway alongside the repo but is **not versioned**, so it
will not come with a `git clone` — the repo is complete without it.

## Deploy

There is no install step and no service. loop-sim is run in place from a checkout —
either as a CLI (`render.py`), a SLURM job (`run_gpu.slurm`), or the camera server, which
you start on a host reachable by whatever consumes the stream. To "deploy" a change:

1. Confirm `pytest tests/` green on a CUDA box (a CPU-only run skips the GPU tests).
2. Land the code on the target checkout (git, or a copy).
3. Restart the camera server if one is running (it holds the scene + compiled kernels in
   memory; there is no reload).

**Note on branch state:** this work lives on `performance-correctness-optimizations`, ~20
commits ahead of `master` and not pushed to GitHub — James owns that decision (see
HANDOFF "Current state"). Work from `master`/this branch; the GitHub default `main` is a
stale divergent "Initial commit".

### Deploy on the TITAN V (voltron)

The 10 fps interactive path is **measured on the real TITAN V — 11.9 fps** — but only with
the stack below. The beamline's default environment (the pt env's torch 2.0.1, system gcc
4.8.5) cannot run `torch.compile` and silently falls back to eager at ~6.3 fps. Build a
dedicated environment once. Voltron's login shell is **tcsh** (`setenv`, not `export`); call
the venv's python by full path because venv `activate` is a bash script:

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
compile actually engaged and beat eager. Then launch the camera server from the same venv,
with `CC`/`CXX` still set and a free GPU pinned:

```tcsh
setenv CUDA_VISIBLE_DEVICES 6      # a free card (check nvidia-smi first)
~/projects/loopsim-torch26/bin/python -m loop_sim.server.camera_server --scene scene_files/hampton_300um.yaml --port 8080
```

Operational notes:

- **~1–2 min compile warmup** at server start — Inductor compiles the preview kernels once.
- **Mesh scenes (`mitegen_200um`) are a knife's-edge VRAM fit** on the 12 GB card — fine on a
  free GPU with torch 2.6, but see HANDOFF risk A / DECISIONS.md for the tiling safety fix.
- **Pin `CUDA_VISIBLE_DEVICES` to a free GPU** — a busy card OOMs the mesh scene on arrival.
- If a run fails at *import* with `GLIBCXX...not found` (not a compile error), wrap the
  command in `scl enable devtoolset-7 "<command>"` so the runtime libraries match.

This whole recipe is the current cost of the 10 fps path on RHEL7; a lighter stack silently
gets you 6.3 fps. DECISIONS.md "TITAN V measured" records why each step is load-bearing.

## Rollback

The renderer has no state and writes nothing outside its output files, so rollback is
just running older code:

- **A change made things wrong or slow:** `git checkout master` (or the previous commit)
  and re-run. `master` is the pre-Jacob baseline: correct on CPU, "hairy" fiber artifact
  on the GPU, no GPU-resident engine.
- **The compiled preview path misbehaves** (silent failure, or wrong frames during
  motion): start the server with `--compile-preview off` — preview frames then render
  eagerly. `--preview-mode off` goes further: every frame is exact full quality (slow,
  but no preview path at all).
- **Suspect the torch engine entirely:** force the numpy reference —
  `--engine numpy` on the server, or `--device cpu` for `render.py`. Slow (minutes/frame)
  but it is the reference implementation everything else is checked against.

## Dev-environment caveat (WSL2 + consumer GPU)

If a run on a WSL2 box slows 10–50× instead of failing, suspect **VRAM spill**: past the
card's VRAM the Windows NVIDIA driver silently spills into system RAM rather than raising
CUDA OOM — the job crawls and the whole desktop drags. Tell: `nvidia-smi` `memory.used`
pinned near the ceiling (≳15.5 GB on a 16 GB card) plus per-item time *degrading* over the
run. Fix by shrinking the working set (resolution, batch, `n_cond`) until it fits. This is
a Windows/WSL2 driver behavior, not a loop-sim bug; it matters here because the mesh scene
is already known to be close to the TITAN V's 12 GB (DECISIONS.md).
