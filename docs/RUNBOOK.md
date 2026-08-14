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

> **`hampton_300um_realistic` was rebuilt 2026-08-11** for slice 3 (neutral
> crystal, flat pin tip, half-maximum drop): 360 frames, 1396x644, supersample 1,
> **2.1 MB**, 8.06 h at 80.6 s/frame, verified against a live f64 render at phi=30
> to **0,0 px registration, mean |diff| 0.00056, 99.1% of pixels identical**.
> All three libraries read `current` (re-measured 2026-08-11).
>
> **SUPERSEDED 2026-08-11 by the mesh AABB cull -- `--tile-size` is no longer
> needed here.** The mesh path now culls and chunks its own survivors, so the
> whole reason for hand-picking a tile is gone and the build is **11.2 minutes
> at 1.86 s/frame** rather than 8.06 h. A plain
>
> ```bash
> $PY -u -m loop_sim.library --scene scene_files/hampton_300um_realistic.yaml \
>     --supersample 1
> ```
>
> holds ~5.9 GB throughout. Passing the old `--tile-size 6800` still works but
> now costs **10x** (19.90 s/frame): it forces 133 passes over a frame that fits
> in one. The historical note follows, because the reasoning still applies to
> any scene heavy enough to need a manual tile.
>
> *(historical)* **Both flags in the command below are load-bearing on WSL2.**
> `--supersample 1`
> is this scene's own optically-correct value; the default 4 on a mesh scene is
> days, not hours. `--tile-size 6800` is ~6 GB by the measured 160 B/ray/face
> law -- the `auto` ramp sizes near the VRAM ceiling and WSL2 silently spills to
> host RAM, which took one overnight run to **~45 min/frame** instead of 80.
> The 2026-08-11 run held 14.8 of 16.4 GB throughout, ~700 MB under that cliff.
>
> A rebuild DELETES the manifest first and overwrites frames in place, so a
> server launched at that scene mid-rebuild will not find a library. All 361
> files are tracked (2.1 MB), so `git checkout -- frame_library/hampton_300um_realistic/`
> recovers the previous one -- which is what recovered the 2026-08-10 incident.
> **The same recovery works for any of the three** -- the libraries are tracked
> deliverables, so an accidental rebuild is a `git checkout` away as long as it
> is caught before the working tree is committed.

> **UPDATED 2026-08-12: only `mitegen_200um` still needs the flag.**
> `hampton_300um_realistic` was rebuilt at `--supersample 4`, which matches
> `build_params`' default, so a bare launch on it is now safe. Re-measured:
>
> ```
>   hampton_300um            bare launch -> serves it
>   hampton_300um_realistic  bare launch -> serves it   (was: WOULD REBUILD)
>   mitegen_200um            bare launch -> WOULD REBUILD (~18 min)
> ```
>
> The rule is unchanged and still bites on mitegen -- the flag must MATCH the
> library's own supersample, and there is no blanket-safe launch command. The
> 2026-08-11 measurement follows.
>
> **LAUNCHING ON AN `--supersample 1` SCENE NEEDS `--supersample 1` ON THE
> COMMAND LINE — AND TWO OF THE THREE SHIPPED SCENES ARE ONE.** Re-measured
> 2026-08-11 across every bundled scene, with `build_library` monkeypatched to
> raise so nothing could run:
>
> ```
>   hampton_300um            bare launch      -> serves it, no build
>   hampton_300um            --supersample 1  -> WOULD REBUILD   (~45 min)
>   hampton_300um_realistic  bare launch      -> WOULD REBUILD   (~8 h)
>   hampton_300um_realistic  --supersample 1  -> serves it, no build
>   mitegen_200um            bare launch      -> WOULD REBUILD   (~1.9 h)
>   mitegen_200um            --supersample 1  -> serves it, no build
> ```
>
> **The flag has to MATCH the library, so there is no blanket-safe launch
> command.** `--supersample 1` rescues the two S=1 scenes and destroys the S=4
> one. Pass the value the library on disk was built at — `manifest.json`'s
> `supersample` field, or `python -m loop_sim.library --scene <s>.yaml` with
> nothing else, which is a no-op when the library is current.
>
> `ensure_library` resolves its kwargs through `build_params`, which fills
> `supersample=4` from its own default, so any library built at another value
> grades stale on the launch path and is rebuilt before the socket binds. The
> **serving** path does not do this: `_grading_params` drops `supersample`
> unless the operator passed it explicitly, because supersample is per-scene by
> design (RUNBOOK "Frame libraries"), so the tab strip correctly reads all three
> as `current`. The launch path is the one that disagrees — see HANDOFF "Open
> questions", where reconciling the two is an open item. `--templates off` and
> the runtime tab strip remain safe by construction.
>
> (While `hampton_300um_realistic`'s library was MISSING, between slice 3 and
> the 2026-08-11 rebuild, no flag helped at all; now that it is `current` the
> flag works again.)

Re-running is a **no-op when the library is current** — the manifest stores a SHA-256 of
the scene YAML, a SHA-256 of the **renderer source** (`render_sha`, added 2026-08-10),
*and* the build parameters, so an edited scene, an edited tracer or a different
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
- **Cost after the mesh cull (2026-08-11), measured on an RTX 4080 SUPER at
  n_cond 7.** Mesh scenes dropped by roughly the fraction of the frame their
  bounding box covers:

  | scene | build raster | before | after | 360-frame build |
  |---|---|---|---|---|
  | `hampton_300um_realistic` | 1396x644 | 80.6 s | **1.86 s** | 8.06 h -> **11.2 min** |
  | `mitegen_200um` | 1840x2296 | 17.0 s | **3.00 s** | 1.9 h -> **18 min** |
  | `hampton_300um` (no mesh) | 5578x2570 | 7.93 s | 8.0 s | ~48 min, unchanged |

  Peak VRAM during a droplet build is ~5.9 GB. **Watch `nvidia-smi`, not
  torch's own counter** -- the caching allocator reserves and never returns, so
  `max_memory_allocated()` under-reports what the card is actually holding, by
  8 GB in one measured case.
- **A build sizes itself to the GPU, and refuses rather than dying half-way.**
  Since 2026-08-11 you should not need to pass `--tile-size` or `--vram-fraction`
  on any card. What happens automatically:

  1. **The budget comes from FREE VRAM, not the card's total** — on a shared
     node like voltron another tenant's allocation reduces yours, rather than
     surfacing as an out-of-memory error at frame 300 of 360. A gigabyte is
     held back for the CUDA context and allocator slack.
  2. **The budget is a hard limit**, so an overrun raises instead of silently
     spilling to host RAM (see "Dev-environment caveat" for why a spill is the
     worse failure).
  3. **A preflight renders ONE frame and reads the real peak before the build
     commits.** It prints a line like
     `[preflight] 1396x644 n_cond=7: 2.26 GB peak against a 10.82 GB budget,
     tile 899024 -- fits`. If it does not fit, the trace tile is reduced (this
     changes no pixels) and re-probed.
  4. **If it still does not fit, the build refuses before rendering anything**,
     naming the largest `--supersample` that would work:
     `... needs more memory than this GPU has: peak 12.4 GB against a 8.8 GB
     budget. Tiling cannot help -- the cost that does not fit scales with
     OUTPUT PIXELS. At this scene's settings --supersample 4 is the largest
     that fits (you asked for 8).`
     A too-large request is never silently downgraded: a library that quietly
     differs from what was asked for would pass every staleness check.

  `LOOPSIM_VRAM_BUDGET_GB` overrides the measured budget. Two uses: leaving room
  for someone else on a shared card, and rehearsing a smaller card's behaviour
  before deploying to it (`LOOPSIM_VRAM_BUDGET_GB=12` on a 16 GB box mimics the
  TITAN V's sizing decisions).

  **REHEARSED 2026-08-13, and the answer is yes: the droplet scene fits a 12 GB
  card.** The docs had never said whether it would -- 7.3 GB was a 16 GB
  measurement. Four frames of `hampton_300um_realistic` at `--supersample 4`
  under a simulated TITAN V:

  ```bash
  LOOPSIM_VRAM_BUDGET_GB=12 $PY -u -m loop_sim.library \
      --scene scene_files/hampton_300um_realistic.yaml \
      --step 90 --root /tmp/lib_rehearsal --force
  ```
  ```
  [preflight] 5578x2570 n_cond=7: 3.30 GB peak against a 8.80 GB budget, tile 1000000 -- fits
  ```

  It completed under the hard ceiling, so the true peak is under 8.8 GiB. The
  tile stayed at the full 1,000,000, which means a 12 GB card runs the same
  tiling regime as a 16 GB one and a cross-machine comparison is hardware
  against hardware. **`--step 90` is the cheap probe**: 4 frames through the
  real loop, real scout, real preflight. Use a disposable `--root` -- a build
  deletes the manifest and overwrites frames in place.

  **Per-frame cost is strongly pose-dependent, so never time one frame.**
  Measured (differencing the elapsed column -- the progress line prints a
  CUMULATIVE average, not a per-frame time): phi=0 91.7 s, phi=90 64.3 s,
  phi=180 92.7 s, phi=270 62.7 s. A 1.48x spread, because the drop is edge-on
  at 90/270 and face-on at 0/180. Mean 77.9 s/frame against the shipped
  build's 74.8, so capping to 12 GB costs essentially nothing.
- **Tile size is not worth hand-tuning.** Measured at 14.4 Mpx: tiles of
  1M / 2M / 4M / 6M rays run 18.6 / 17.5 / 17.3 / 17.1 s, all byte-identical.
  Six times the tile buys 8% and costs 1.8 GB of peak, so the default stays at
  1M. The curve is steep in the other direction, though -- a tile small enough
  to force ~130 passes costs ~10x -- which is why the preflight reduces the
  tile proportionally rather than dropping to its floor.
- **Raising `--supersample` on a mesh scene costs time, not correctness.** See
  docs/HANDOFF.md "Open questions" for the measured table: supersample 4 on the
  droplet scene runs 20-85 s/frame depending on droplet tessellation, i.e. a
  2-8 h build. (An earlier note here said a 50,976-face droplet spilled past
  16 GB and could not complete a frame; that was a defect in the survivor-chunk
  sizing, not a limit of the scene, and it peaks at 10.4 GB now.)
- **Cost** *(historical, pre-cull)* (RTX 4080 SUPER, n_cond 7): `hampton_300um` ~7.2 s/frame at `--supersample 4`
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
| `--camera-emulation` | `on` | map transmittance through the illumination field, black floor and tone response (`loop_sim/renderer/field.py`), so an empty field reads ~0.60 and an opaque body ~0.18 instead of the rails. **Serve-time only — costs no library rebuild** |
| `--mono` | **`off`** (was `on` until 2026-08-14) | `on` collapses to grey before the camera stage, masking the fact that colour is an ABSORPTION spectrum here — a scene declaring a crystal `[0.7,0.9,1.0]` renders it blue. Now off by default: the simulator is a colour instrument and scenes are allowed to be coloured, so flattening by default meant no coloured scene could ever be seen. Measured on the shipped libraries, on-vs-off differs by at most 20/21/46 levels on 0.003–0.42% of pixels (realistic/hampton/mitegen) — a tint on loop and droplet edges, not a wash. The scene-side repair is `colour: [1,1,1]` with the absorption in `mu_optical`, which *rebuilds every library*. Ignored when `--camera-emulation off` |
| `--pin-streak` | `on` | draw the specular glint a real machined pin carries along its shank. Its position is PROJECTED FROM THE SCENE (`renderer/pin_projection.py`) through the current pose, so it is exact at any zoom, crop or angle, and there is simply no glint when the pin is out of view. Only an object the code declares shiny gets one (`SHINY`, currently `pin`+`metal`), so `mitegen_200um` never does. Ignored when `--camera-emulation off` |
| `--sensor-pitch` | `on` | deliver on the real camera's **704×480** raster. The BL831 pixels are 1.11 non-square and the tracer's are square, so a 640-wide render covers the same field (to under 1%) on a different grid — and a consumer applying dcss's µm-per-pixel constant to 640 columns reads 10% wide. `off` serves the render's own square pixels. On the template path the resample runs in PIL rather than `field.to_sensor` (6.7 → 1.3 ms, agrees to 1 level); the live path still uses `to_sensor` |
| `--template-cache` | **`auto`** (was `off` until 2026-08-14) | decoded templates held in RAM. `auto` takes as much of the library as half of AVAILABLE memory allows; `off` is 8; an integer pins it. Default now that a template stores only its content — ~4.7 MiB a frame, ~1.8 GiB for a 360-frame sweep, against the 14.4 GiB the full window cost. If the host cannot afford a whole revolution `auto` **declines**: LRU against a cyclic sweep evicts each frame just before it comes round again, so a partial cache is worth zero rather than a share |
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

**Pass = 213 tests green** (last run 2026-08-11: 213 passed in 121 s on an RTX 4080 SUPER;
60 warnings are expected — benign `divide by zero`/RuntimeWarnings from the numpy
reference primitives). On a CPU-only box the CUDA-gated parity tests **skip** rather than
fail, so a green run there is a weaker check — it does not exercise the GPU engine at all.

The count grows with the work; treat the number here as the figure from the last recorded
run rather than a constant, and `docs/HANDOFF.md`'s front-matter `last_verified` as the
authority. A run that comes in *below* it is the signal worth chasing.

The suite covers GPU↔CPU render parity (the correctness fix), torch↔numpy shape parity,
beam attenuation, the compiled preview path, and the server's settle/single-flight
behavior.

For a render-level check after touching the renderer or a scene, use the SLURM comparison
job (`sbatch run_gpu.slurm` renders CPU+GPU at n_cond 1 and 7 and reports diff stats);
`../CLAUDE.md` "Comparison workflow" carries the acceptable thresholds.

Benchmarking the **serve** path (no GPU, no socket, no display — safe on a busy shared
node, and verified to import torch not at all):

```bash
$PY bench_serve.py --scene scene_files/hampton_300um_realistic.yaml --frames 40
```

On **voltron**, from the deployment venv, with all eight cards busy:

```tcsh
cd ~/projects/loop_sim_MINE/xtal-loop-sim
~/projects/loopsim-torch26/bin/python bench_serve.py --json serve.json
```

It prints the three regimes, a per-stage split, a comparison against the recorded
pre-crop numbers for that host, and a **GO/NO-GO against the 10 fps goal** (exit 0 / 1).
If the host's libraries still store the full window it says so and points at
`--recrop`, because otherwise a host that has not picked up the cropped libraries just
reads ~7x slower on the decode with nothing to explain why.

**slew** = spindle turning, every frame a fresh decode (the worst case, and what grades
the host); **pan** = fixed angle, decode served from cache; **hold** = the floor.

Dev box, 40 frames, `hampton_300um_realistic`, before and after the 2026-08-14 crop
(`--mono off` and `--template-cache auto` in the "after" column, matching the server):

| | before (2026-08-13) | **after (2026-08-14)** | gain |
|---|---|---|---|
| slew (cold) | 91.1 ms / 11.0 fps | **27.0 ms / 37.1 fps** | 3.4x |
| slew (warm) | 27.1 ms / 36.9 fps | **13.4 ms / 74.8 fps** | 2.0x |
| pan | 32.4 ms / 30.9 fps | 10.5 ms / 95.2 fps | 3.1x |
| decode | 66.6 ms | **11.8 ms** | 5.7x |
| crop+scale | 15.3 ms | **1.9 ms** | 8.0x |
| camera model | 12.3 ms | **5.8 ms** | 2.1x |
| jpeg encode | 2.7 ms | 2.4 ms | 1.1x |
| RAM to hold the sweep | 14.4 GiB | **1.64 GiB** | 8.8x |

**voltron before the crop, for reference** (2026-08-13, full-window templates): cold slew
265.5 ms / 3.77 fps, warm 67.3 ms / 14.87 fps at 14.4 GiB; stages 180.7 / 32.6 / 33.0 /
5.9. Applying the per-stage dev→voltron ratios to the "after" column projects a cold slew
near **57 ms / 17.6 fps with no cache at all** — but that is a projection, and running the
command above on voltron is what replaces it with a measurement.

**As of 2026-08-13, voltron missed the 10 fps goal on a COLD slew by 3x and cleared it
warm at 14.87 fps** — the viewer was usable there only with `--template-cache auto` and
its 14.4 GiB. The crop is expected to have removed that condition (the projection above is
~17.6 fps cold with no cache), but **that is a projection until someone runs the command
on voltron.** The measured warm slew (67.25 ms) and pan (67.30 ms) agreed to
**0.05 ms** -- with the library resident a rotating frame costs exactly what a translating
one costs, i.e. the decode is gone rather than reduced. All 360 frames fit voltron's RAM,
so a full revolution stays warm; see the LRU caveat below for hosts where they do not.
The cold figure remains the number to design against for a first revolution — it is the likely viewer host, and the beamline node is never idle, so a
figure taken under load is the deployment figure rather than a degraded one. The p90/p10
spread is 1.34 there against 1.31 on the dev box, i.e. the same distribution shape, so
this is systematic CPU speed and not contention noise; re-measuring on a quiet node would
not move it much.

**Addressable 2026-08-13 by holding the library in RAM, not by threading.** Warm, a slew
becomes a pan: measured on the dev box **27.1 ms / 37.0 fps against a cold 106.1 ms**. No
threads, no prefetch, no direction prediction — which also makes it the right answer for
the AXIS consumer, whose `/motor` is instant and absolute and therefore has no predictable
slew to prefetch along.

**Opt-in until 2026-08-14, when the tight crop made it cheap enough to default.** A decoded
template used to be the whole `width*height*3` = 41 MiB window, so the sweep was 14.4 GiB —
nothing against voltron's 251 GB and a great deal on a workstation, hence opt-in. Templates
now store only their content (~4.7 MiB a frame, **1.76 GiB** for the sweep), so
`--template-cache` defaults to **`auto`**. Two consequences worth knowing: the 16 GB dev box
now holds a full revolution where it previously held 182 of 360 and thrashed to zero
benefit; and `auto` **declines rather than half-filling** — see the LRU caveat immediately
below, which is why a partial cache is not worth the memory it costs.

**The caveat, and it is a real one: LRU thrashes on a cyclic sweep.** If the cache is
smaller than a revolution, each frame is evicted just before it comes round again and the
benefit is *zero*, not proportional. Measured:

```
  cache=64 cycle=32   lap1 89.2 ms  lap2 12.9 ms   warm
  cache=16 cycle=32   lap1 76.1 ms  lap2 69.8 ms   thrash
  cache=64 cycle=128  lap1 81.4 ms  lap2 71.0 ms   thrash
```

So the flag is all-or-nothing per sweep length. voltron holds all 360 and is fine. A 16 GB
WSL2 dev box gets 182 frames (7.3 GiB) and so stays cold on a *full* revolution while
benefiting on any sweep under 182 frames — which is what centring actually does.
`bench_serve.py` prints a warning naming both numbers when the cache cannot hold the
library.

Decode is 73% of a cold slew on both machines. Warmup deliberately uses angles the timed run never
revisits: warming on the timed poses reads 78 ms with a p10 of 25.8, and that p10 is the
pan number leaking in.

Benchmarking the **render** path: `bench_frame.py` (flags `--compiled`, `--fp32`); soak the live server with
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

**Note on branch state:** this work lives on `performance-correctness-optimizations`, **61
commits ahead of `master`** (measured 2026-08-11) and not pushed to GitHub — James owns
that decision (see HANDOFF "Current state"). Measure it rather than quoting this line:
`git rev-list --count master..HEAD`. Work from `master`/this branch; the GitHub default
`main` is a stale divergent "Initial commit".

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
