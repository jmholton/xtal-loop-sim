# RUNBOOK: loop-sim (xtal-loop-sim)

Environment, run, verify, deploy, rollback: the operational path for loop-sim, in that
order. `../README.md` is the user guide (full pipeline, every CLI flag, every HTTP
endpoint); this file is the from-nothing setup and the operational detail the README
leaves out. Paths are relative to the repo root.

- [Environment (from nothing)](#environment-from-nothing)
- [Run](#run)
  - [X-ray](#x-ray)
  - [Frame libraries](#frame-libraries)
  - [Switching scenes on a running server](#switching-scenes-on-a-running-server)
  - [On voltron (the beamline GPU node)](#on-voltron-the-beamline-gpu-node)
- [Every lever](#every-lever)
- [Verify](#verify)
- [Other scripts](#other-scripts)
- [Deploy](#deploy)
  - [Deploy on the TITAN V (voltron)](#deploy-on-the-titan-v-voltron)
- [The DHS (xtalLoopSimDHS)](#the-dhs-xtalloopsimdhs)
- [Rollback](#rollback)
- [Dev-environment caveat](#dev-environment-caveat)

## Environment (from nothing)

One recipe, every host:

```bash
git clone <repo> && cd xtal-loop-sim
bash setup_venv.bash
```

`setup_venv.bash` creates `.venv/` from `requirements.txt` (torch 2.6.0 cu124, pillow
10.4.0, numpy, scipy, pyyaml, tifffile, matplotlib, pytest; `--only-binary=:all:`),
prints the torch/CUDA/numpy/pillow versions, and runs `pytest tests/`. `--force`
rebuilds `.venv/` first, `--skip-tests` skips the pytest run, `--acceptance` also runs
`tools/acceptance_voltron.py` (GPU).

Base interpreter: `/home/programs/pytorch/envs/pt/bin/python3.10` when present (the
beamline hosts), else `/usr/bin/python3`. Never a conda python: pillow 10.4.0 has no
wheel for 3.13, so a 3.13 base fails the `--only-binary` install. When devtoolset-7 is
present (voltron) the script exports `CC`/`CXX` to it for the test run, so
`torch.compile` builds against a modern compiler instead of the system gcc.

Every command in this file is `.venv/bin/python ...` from the repo root. Serving from
templates never imports torch; torch is needed only to render.

- **Dev box:** `bash setup_venv.bash`, nothing else.
- **voltron:** login shell is tcsh; run `bash setup_venv.bash` through bash as shown.
  devtoolset-7 is what makes `torch.compile` work here. Pin `CUDA_VISIBLE_DEVICES` to a
  free card before starting a live-render server (voltron is an 8-GPU shared node).
- **dataserver3:** no GPU. Serving from templates needs no torch at all, but the venv
  still installs it; the GPU-gated tests skip on a CPU-only box (see "Verify").

The DHS has its own environment:
[`xtalLoopSimDHS/README.md` "Create the env"](../xtalLoopSimDHS/README.md#create-the-env).

## Run

```bash
# Render a bundled scene (tube-based; exercises the GPU path)
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --n-cond 7 --output /tmp/out.jpg

# Same on the GPU (GPU-resident engine, engine_torch.py)
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --n-cond 7 --device cuda --output /tmp/out.jpg

# X-ray radiograph instead of the optical image
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --xray --output /tmp/out_xray.png

# Rotate about the spindle (rotx for the bundled scenes) / translate
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --rotx 45 --n-cond 1
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --tx 0.05 --ty -0.02 --n-cond 7

# Live AXIS-compatible camera server (uses the GPU-resident engine when CUDA is present)
.venv/bin/python -m loop_sim.server.camera_server --scene data/scene_files/hampton_300um.yaml --port 8080
```

Then open `http://<host>:8080/` for the control page, or point an AXIS consumer at
`http://<host>:8080/axis-cgi/mjpg/video.cgi`. `../README.md` documents every flag and
endpoint; `loop_sim/server/camera_server.py`'s docstrings explain the animation and
concurrency model.

Notes that save time:

- **`scene.yaml` / `loop.yaml` are pipeline outputs, gitignored and not shipped.** The
  three scenes in `data/scene_files/` are complete; build your own with the README's
  "Full pipeline".
- **`--device cuda` runs the GPU-resident engine** (`engine_torch.py`), the same one the
  camera server uses, for any bundled scene, tube or mesh. Falls back to CPU when no
  CUDA device is visible.
- Run from the repo root. Which motor is the spindle is scene-dependent: `rotx` for the
  bundled scenes.

### X-ray

```bash
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --xray --output out_xray.png
```

`GET /xray` on the camera server serves the same thing live; `GET /beam` gives the beam
profile as JSON. Unlike the optical frame library, X-ray has no pre-computed sweep: it
always renders live, single-shot, on both paths. See DECISIONS §2026-09-22 for why.

### Frame libraries

A frame library is a full 360° spindle sweep rendered once and replayed, so the camera
responds instantly and nothing renders at request time. Libraries are **tracked in
git** and live in `data/frame_library/<scene_stem>/` alongside a `manifest.json`.

The camera server **never builds**. At launch and on every scene switch it grades the
library on disk against the flags it was started with: current -> serve; stale -> serve
with one warning naming what differs; no library -> that scene renders live and the
server prints the exact build command:

```bash
.venv/bin/python -m loop_sim.library --scene data/scene_files/hampton_300um.yaml
.venv/bin/python -m loop_sim.library --all
.venv/bin/python -m loop_sim.library --scene <s>.yaml --force        # the only way to rebuild
.venv/bin/python -m loop_sim.library --status --all                  # read-only
.venv/bin/python -m loop_sim.library --verify --scene <s>.yaml [--verify-angle DEG]
```

`--scene` builds only when the library is missing; current or stale is reported and
skipped. `--verify` re-renders one stored frame live (needs CUDA unless `--allow-cpu`)
and reports the worst/mean grey-level difference against disk: PASS at <= 1 level, exit
1 on FAIL, how you find out whether a renderer edit actually moved a template pixel
(CLAUDE.md's invariants). Builds are atomic (`<lib>.new` swapped in, `.old` removed
after): a crash or Ctrl-C mid-build never touches the working library. Every build
flag, with its default and cost, is in "Every lever" below.

Two graders, two possibly different answers, both right: `--status` grades against the
build **defaults** (supersample 4), so `mitegen_200um` reads `stale`; the live server
grades against the flags it was **launched** with, so the same library reads `current`
in `/scenes` when nothing overrode `--supersample`. Pass `--supersample 4` at launch and
the server reports it stale too (and still serves it).

| scene | supersample | frames | size |
|---|---|---|---|
| `hampton_300um` | 4 | 360 | 15 MB |
| `hampton_300um_realistic` | 4 | 360 | 14 MB |
| `mitegen_200um` | 1 | 360 | 9 MB |

~37 MB total. `mitegen_200um` is the one that reads `stale` (built at supersample 1; the
default asks 4; served as-is).

Notes:

- **The window is measured from the scene, not centred on the origin**: a symmetric
  margin would leave most of hampton's 6.7 mm pin unrendered in a 4.7 mm field. Served:
  the spindle axis (quantised to `--step`), `tx`/`ty`/`tz` as a crop (`tz` becomes
  defocus blur), `zoom` between the window floor and `--supersample`. `roty`/`rotz`
  aren't covered: `--axis` other than `rotx` is refused rather than built wrong.
- **Out-of-range requests are refused**, except in the live server, which clamps (and
  prints what it clamped) so it keeps serving, sliding the crop rather than squeezing it.
- **Lateral motor depends on φ**: at φ=0 `ty` is vertical and `tz` is defocus; at φ=90
  they swap (`pose_crop`).
- **Pick `--supersample` per scene**, where the template pitch reaches the objective's
  Nyquist limit, `0.61λ/NA / 2`:

  | scene | pixel | NA | Nyquist | camera is… | supersample |
  |---|---|---|---|---|---|
  | `hampton_300um` | 7.4 µm | 0.10 | 1.68 µm | under-sampling 4.4× | **4** |
  | `mitegen_200um` | 1.0 µm | 0.10 | 1.68 µm | already over-sampling 1.7× | **1** |

  Beyond that magnifies resolution the optics can't deliver, and the template cost
  grows with the square. mitegen's content is wider than its field (1.10 mm vs 0.48
  mm), so the useful zoom direction is *out*, which the window already provides.
- **A build sizes itself to the GPU and refuses rather than dying half-way**: the
  budget comes from free VRAM, not the card's total, and is a hard limit (an overrun
  raises rather than spilling to host RAM, see "Dev-environment caveat"). A preflight
  renders one frame and reads the real peak before committing; if it doesn't fit, the
  trace tile shrinks first, then the build refuses and names the largest workable
  `--supersample` rather than silently downgrading. `LOOPSIM_VRAM_BUDGET_GB` overrides
  the measured budget (`=12` on a 16 GB box mimics the TITAN V). Watch `nvidia-smi`,
  not torch's own counter: the caching allocator reserves and never returns.
- **`--supersample` on a mesh scene costs time, not correctness** (DECISIONS
  §2026-08-12); **WSL2 has no OOM to catch** (see "Dev-environment caveat").

### Switching scenes on a running server

The control page carries a tab per scene in `data/scene_files/`; clicking one swaps
the sample live without restarting or dropping the MJPEG stream, and resets the pose
to home. A millimetre differs between scenes whose pixel sizes differ 7.4×. Same thing
from a terminal:

```bash
curl -X POST 'http://host:8080/scene?path=mitegen_200um'   # 202 accepted
curl -s http://host:8080/scene                             # progress + errors
curl -s http://host:8080/scenes                             # library state of each
```

Each tab is badged with its library state, and only one badge stops a switch:

- **(no badge)**: matches current build settings; switches immediately.
- **`stale`**: complete and servable, built with different settings. **Switches
  immediately** and the page names what differs; never rebuilt automatically.
- **`preview`**: only a coarse on-demand library exists (5° steps, 1× zoom).
- **`no library`**: nothing to serve; the page offers a preview or full build.

**Builds are refused without CUDA** (~179 s/frame → hours for a full library), in the
viewer and the CLI. The viewer has no override; build offline on a GPU host instead:

```bash
.venv/bin/python -m loop_sim.library --scene data/scene_files/<scene>.yaml            # full, tens of minutes
.venv/bin/python -m loop_sim.library --scene data/scene_files/<scene>.yaml --preview  # coarse, minutes
.venv/bin/python -m loop_sim.library --scene <scene>.yaml --allow-cpu                 # if you really mean it
```

Two consequences:

- **A switch waits for the in-flight frame**: the stage briefly stops responding, ~70
  ms on the default template path, up to ~1 s with `--templates off`, one whole frame
  (~18 s) on `--templates off --engine numpy`.
- **`--templates off --engine torch` loses the compiled preview after the first
  switch** (6.3 fps eager instead of 11.9): `torch.compile` warmup only runs at
  startup. Restart to get it back; the default template path is unaffected.
- **With `--templates off`, a switch holds both the old and new `TorchScene`** until
  the install completes, so peak VRAM is the sum.

### On voltron (the beamline GPU node)

GPU rendering requires CUDA, which lives on voltron. Submit from the local machine:

```bash
sbatch tools/run_gpu.slurm      # gpu partition, gres=gpu:1
squeue --job <jobid>
cat slurm_<jobid>.log
```

**Never set `--time` in these job scripts**: this queue has no time limits and the flag
gets jobs cancelled early. If you need an interactive session, voltron's login shell is
tcsh and does not parse `&&`: write a bash script and run
`ssh voltron "cd $PWD ; bash script.bash"`.

## Every lever

Everything a user can turn, with its default and what it does. A lever marked
*library key* differs from what the shipped libraries were built with: touching it
doesn't rebuild anything by itself, but the affected scene now reads `stale` and is
served as-is until you rebuild with `--force`.

### `render.py`: offline single frame

| Flag | Default | Effect |
|---|---|---|
| `<scene.yaml>` | none | scene to render (positional) |
| `--tx` `--ty` | from the scene's `motor:` block | stage translation, mm. The CLI overrides the YAML; there is no `--tz` here |
| `--rotx` `--roty` `--rotz` | from the scene's `motor:` block, else 0 | rotation, degrees; `rotx` is the spindle for the bundled scenes |
| `--n-cond` | 1 | condenser angles per pixel; 7 = soft NA edges, >7 buys little |
| `--device` | `cpu` | `cuda` runs the GPU-resident engine (`engine_torch`), same one the camera server uses; falls back to CPU with no CUDA visible |
| `--xray` | off | render the X-ray transmission radiograph at this pose instead of the optical image: an 8-bit greyscale PNG, encoded exactly as `GET /xray` encodes it |
| `--output` | `<scene_basename>.jpg` (`<scene_basename>_xray.png` with `--xray`) | output path |

`render.py` exposes no `--zoom`; set `zoom` in the scene's `motor:` block, or use the
server, whose `/motor` endpoint takes all seven axes.

### `python -m loop_sim.server.camera_server`: the live/pretend camera

| Flag | Default | Effect |
|---|---|---|
| `--scene` | `data/scene_files/hampton_300um.yaml` | scene to serve |
| `--host` / `--port` | `0.0.0.0` / 8080 | bind address |
| `--templates` | `on` | serve from a pre-computed library when one exists, stale or not; never builds. `off` raytraces every frame, the correctness reference |
| `--fps-limit` | 30.0 | MJPEG wire-rate ceiling |
| `--n-cond` | 7 | condenser angles for settled frames |
| `--jpeg-quality` | 85 | quality of frames the server **sends**; not the stored template, see `--template-quality` |
| `--engine` | `auto` | `torch` (GPU-resident) / `numpy` (reference) / auto-detect |
| `--preview-mode` | `on` | approximate frames while moving, exact on settle |
| `--compile-preview` | `on` | `torch.compile` the preview path (CUDA + preview only) |
| `--settle-delay` | 0.5 s | quiet time after a `/motor` set before the exact frame renders |
| `--prewarm` | `on` | decode the whole library before the socket binds; `off` fills lazily |
| `--template-cache` | `auto` | decoded templates held in RAM; `auto` takes up to half of available memory; `off` caps at 8 frames; an integer pins the count |
| `--camera-emulation` | `on` | map transmittance through the illumination field (empty ~0.60, opaque ~0.18, not black/white). Serve-time only, no library rebuild |
| `--mono` | `off` | `on` collapses to grey, masking that colour here is an absorption spectrum (a crystal `[0.7,0.9,1.0]` renders blue). Fix at the scene level instead: `colour: [1,1,1]` plus `mu_optical` -- *library key*. Ignored when `--camera-emulation off` |
| `--pin-streak` | `on` | draw the specular glint a machined pin carries, projected from the scene onto objects declared shiny only (`mitegen_200um` never gets one). Ignored when `--camera-emulation off` |
| `--sensor-pitch` | `on` | deliver on the real camera's 704×480 raster (BL831 pixels are 1.11 non-square, the tracer's are square). `off` serves the render's own square pixels |
| `--supersample` | builder default (4) | *library key* |
| `--template-format` | builder default (`png`) | *library key* |
| `--template-quality` | builder default (90) | stored-template JPEG quality; ignored for png. *library key* |
| `--library-root` | `data/frame_library/` | frame-library root to serve from and report on |
| `--preview-root` | `data/frame_library_preview/` | on-demand **preview** libraries' root, separate so a build never overwrites cached frames |
| `--scene-dir` | `data/scene_files/` | which `*.yaml` are offered for switching on `/scenes` |
| `--jpeg-receiver` | none | URL that `POST /video-trigger?state=open` pushes each new frame to as `Content-Type: image/jpeg` (what pydhsfw's jpeg_receiver accepts) |
| `--push-fps` | 30 | ceiling on the `--jpeg-receiver` push rate |
| `--camera-zoom` | `1:1.0,2:0.5,3:0.25` | AXIS camera number -> zoom stop for `camera=N` on the MJPEG/snapshot URLs (the beamline's three sample cameras on one AXIS server). Unknown `N` answers 400 |

### `python -m loop_sim.library`: build a frame library

| Flag | Default | Effect |
|---|---|---|
| `--scene` / `--all` | none | one scene, or every `data/scene_files/*.yaml` |
| `--root` | `data/frame_library/` | output directory |
| `--step` | 1.0° | degrees between frames → 360 frames. *library key* |
| `--supersample` | 4 | render this many times finer than the camera pixel; the hard ceiling on zoom-in. *library key* |
| `--pan-mm` | 0.6 mm | travel to allow beyond the scene and the centred field. *library key* |
| `--n-cond` | 7 | condenser angles. *library key* |
| `--axis` | `rotx` | spindle motor; anything else refused rather than built wrong. *library key* |
| `--format` | `png` | stored template format, lossless and smaller here. *library key* |
| `--psf` | `on` | bake the objective diffraction PSF into the templates. *library key* |
| `--quality` | 90 | JPEG quality; ignored when `--format png`. *library key* |
| `--tile-size` | `auto` | rays per trace pass; `auto` measures the size by trial renders. Doesn't change pixels |
| `--vram-fraction` | 0.80 | share of free VRAM the auto tile may use. Doesn't change pixels |
| `--device` | auto | `cuda` when available |
| `--force` | off | the only way to rebuild a library that already exists |
| `--status` | off | print each library's status, provenance and, when stale, what differs. Reads only |
| `--verify` | off | re-render one stored frame live, report worst/mean pixel difference against disk: PASS at <= 1 grey level, exit 1 on FAIL. Needs CUDA unless `--allow-cpu` |
| `--verify-angle` | 0 | spindle angle to verify, degrees; nearest stored frame is checked |
| `--preview` | off | build the coarse library the server builds on demand (5° steps, 1× supersample, n_cond 1 → 72 frames) into `data/frame_library_preview/` |
| `--recrop` | off | migrate an existing full-window library to content-only storage in place, no GPU, ~2.5 min per sweep |
| `--allow-cpu` | off | permit a CPU build. Without it, refused: ~179 s/frame is ~3.6 h for a preview, ~18 h for a full library |

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
| interpreter | `.venv/bin/python`, built by `bash setup_venv.bash` | the only Python with numpy/scipy/PIL/pyyaml/torch. The system Python has none of them |
| CUDA present | auto-detected | picks the GPU-resident engine; absent falls back to numpy (minutes per frame) |
| `CC` / `CXX` | devtoolset-7 on voltron | required for `torch.compile`; without it the server silently drops to eager and misses 10 fps |

---

## Verify

```bash
.venv/bin/python -m pytest tests/ -q
```

274 tests across 21 files, all green in the root `.venv` as of 2026-09-22. Fewer tests
collected than last time is the signal worth chasing. Benign
`divide by zero`/RuntimeWarnings from the numpy reference primitives are expected, not
a failure. On a CPU-only box (dataserver3) the CUDA-gated parity tests **skip**, so a
green run there is a weaker check.

On a 17 GB WSL2 box, run one file at a time instead: a full-suite invocation has
exhausted RAM and crashed WSL2 before.

```bash
for f in tests/test_*.py; do .venv/bin/python -m pytest "$f" -q; done
```

About 80 s per file is torch import on the DrvFs mount. The suite covers GPU↔CPU
render parity, torch↔numpy shape parity, beam attenuation, the compiled preview path,
and the server's settle/single-flight behavior.

For a render-level check after touching the renderer or a scene, `sbatch
tools/run_gpu.slurm` renders CPU+GPU at n_cond 1 and 7 and reports diff stats;
DECISIONS §2026-05-22 has the legacy-path thresholds.

Benchmarking the **serve** path (no GPU, no socket, no display, no torch import):

```bash
.venv/bin/python tools/bench_serve.py --scene data/scene_files/hampton_300um_realistic.yaml --frames 40
```

Prints the three regimes and a **GO/NO-GO against the 10 fps goal** (exit 0/1). If the
host's libraries still store the full window it says so and points at `--recrop`.
**slew** = spindle turning, fresh decode every frame; **pan** = fixed angle, decode
cached; **hold** = the floor. **Read `slew_warm`, not `slew`**: the server pre-warms
the whole library at boot, so a spindle slew never touches disk once it is up;
`slew_warm` is what the GO/NO-GO verdict grades. All three beamline hosts (dataserver3,
voltron, gateway) clear the goal warm; see DECISIONS §2026-08-14.

Benchmarking the **render** path: `tools/bench_frame.py` (flags `--compiled`, `--fp32`;
`--modality xray` times the live X-ray tracers instead). `soak_server.py` soaks the
live server from **outside this repo**, in the analysis tree at
`/home/jadoughty/projects/loop_sim_MINE/investigation/2026-07_scene_and_perf_harnesses/`
(mirrored to the gateway, not versioned).

## Other scripts

James's seven pipeline scripts stay at the repo root, flags unchanged. Everything else
is in `tools/`, run from the repo root:

| script | what |
|---|---|
| `tools/test_gpu.bash` | renders `hampton_300um` on CPU then GPU, timed. The base A/B smoke test |
| `tools/test_optim.bash` | renders on GPU, timed, diffs against a prior `scene_gpu.jpg` |
| `tools/debug_optim.bash` | renders CPU (reference) then GPU, reports max/mean pixel difference and pixels off by more than 10 |
| `tools/check_diff.bash` | three-way diff of `scene_ref.jpg` / `scene_gpu.jpg` / `scene_optim.jpg`, after the scripts above produce them |
| `tools/profile_render.py` / `.bash` | cProfile the CPU renderer on `hampton_300um.yaml` |
| `tools/profile_gpu.py` / `.bash` | same, on the GPU path, with a warm-up render discarded first |
| `tools/bench_frame.py` | render-path benchmark; see "Verify" |
| `tools/bench_serve.py` | serve-path benchmark; see "Verify" |
| `tools/acceptance_voltron.py` | GO/NO-GO acceptance test; see "Deploy on the TITAN V" |
| `tools/run_gpu.slurm` | SLURM job; see "On voltron" and "Verify" |
| `make_beam_image.py` | synthetic beam-profile PNG for the X-ray beam entity. Stays at the repo root with James's other scripts |

Every `.bash` wrapper `cd`s to the repo root itself, so it runs the same from any
directory.

## Deploy

There is no install step and no service: loop-sim runs in place from a checkout, as a
CLI (`render.py`), a SLURM job (`tools/run_gpu.slurm`), or the camera server started on
a host reachable by whatever consumes the stream. To "deploy" a change: confirm
`.venv/bin/python -m pytest tests/ -q` green on a CUDA box (CPU-only skips the GPU
tests), land the code on the target checkout (git, or a copy), then restart the camera
server if one is running (it holds the scene and compiled kernels in memory; there is
no reload).

**Branch state:** this work lives on `performance-correctness-optimizations`, not
pushed to GitHub. James owns that decision (see HANDOFF "Current state"). Measure how
far ahead of `master` with `git rev-list --count master..HEAD`; don't quote a number
here. Work from `master` or this branch: the GitHub default `main` is a stale,
divergent "Initial commit".

### Deploy on the TITAN V (voltron)

Same environment recipe as everywhere else:

```tcsh
cd ~/projects/loop_sim_MINE/xtal-loop-sim
bash setup_venv.bash
```

**The viewer needs nothing beyond that venv**: serving from templates imports torch not
at all.

```bash
.venv/bin/python -m loop_sim.server.camera_server --scene data/scene_files/hampton_300um_realistic.yaml --port 8080
```

Everything below is for **rendering**: building libraries, `--templates off`, and
`tools/acceptance_voltron.py`. The 10 fps live-render path is **measured on the real
TITAN V at 11.9 fps**, with devtoolset-7 exported (`setup_venv.bash` does this itself
when it finds `/opt/rh/devtoolset-7`); without it `torch.compile` falls back to eager
at ~6.3 fps.

```bash
.venv/bin/python tools/acceptance_voltron.py    # GO/NO-GO, writes acceptance_report.json
```

A GO means compile engaged and beat eager (`bash setup_venv.bash --acceptance` runs
this as part of setup). For the live-render path (`--templates off`, or a scene with no
library yet), pin a free GPU:

```tcsh
setenv CUDA_VISIBLE_DEVICES 6      # a free card (check nvidia-smi first)
.venv/bin/python -m loop_sim.server.camera_server --scene data/scene_files/hampton_300um.yaml --port 8080 --templates off
```

Voltron's login shell is **tcsh** (`setenv`, not `export`); call `.venv/bin/python` by
full path, since venv `activate` is a bash script.

Notes: **~1-2 min compile warmup** at server start (Inductor compiles the preview
kernels once). **A build sizes itself to free VRAM and refuses rather than OOMing**
(the droplet scene fits a 12 GB card, DECISIONS §2026-08-11). **Pin
`CUDA_VISIBLE_DEVICES` to a free GPU**: a busy card OOMs the mesh scene on arrival. A
`GLIBCXX...not found` failure at *import* (not a compile error) means wrapping the
command in `scl enable devtoolset-7 "<command>"` so the runtime libraries match.

## The DHS (xtalLoopSimDHS)

```bash
cd xtalLoopSimDHS
python3 -m venv .venv
.venv/bin/python -m pip install --no-deps -e /path/to/pydhsfw
.venv/bin/python -m pip install -r requirements.txt

./xtalLoopSimDHS.sh pretend            # no camera server; the same DCSS traffic
./xtalLoopSimDHS.sh real                # drives the camera server named in the config

.venv/bin/python -m pytest tests -q     # 27 offline tests, ~20 s
```

`real` mode drives the camera server over HTTP; start that server first, with
`--jpeg-receiver URL` if you want it to push frames the way the real AXIS camera does.
Full recipe, wire contract and device table:
[`xtalLoopSimDHS/README.md`](../xtalLoopSimDHS/README.md).

For the dcss/BluIce sandbox (restoring the seed database, running dcss, pointing
BluIce's video at the simulator): see
[`xtalLoopSimDHS/sandbox/README.md`](../xtalLoopSimDHS/sandbox/README.md).

## Rollback

The renderer has no state and writes nothing outside its output files, so rollback is
just running older code:

- **A change made things wrong or slow:** `git checkout master` and re-run. `master` is
  the pre-Jacob baseline: correct on CPU, "hairy" fiber artifact on the GPU, no
  GPU-resident engine.
- **The compiled preview path misbehaves:** start the server with `--compile-preview
  off` (preview frames then render eagerly), or `--preview-mode off` for every frame
  exact and slow.
- **Suspect the torch engine entirely:** `--engine numpy` on the server, or `--device
  cpu` for `render.py`. Slow (minutes/frame) but the reference implementation
  everything else is checked against.

## Dev-environment caveat (WSL2 + consumer GPU)

If a run on a WSL2 box slows 10–50× instead of failing, suspect **VRAM spill**: past
the card's VRAM the Windows NVIDIA driver silently spills into system RAM rather than
raising CUDA OOM, and the job crawls. Tell: `nvidia-smi` `memory.used` pinned near the
ceiling (≳15.5 GB on a 16 GB card) plus per-item time *degrading* over the run. Fix by
shrinking the working set (resolution, batch, `n_cond`) until it fits. Not a loop-sim
bug; it matters here because the mesh scene is already close to the TITAN V's 12 GB
(DECISIONS.md).
