# RUNBOOK: loop-sim (xtal-loop-sim)

How to set up, run, verify and deploy loop-sim. `../README.md`
is the user guide (pipeline, endpoints, scene format); this file is the operational
detail. Every command is `.venv/bin/python ...` from the repo root unless it says
otherwise.

- [1. Environment](#1-environment)
- [2. Run](#2-run)
- [3. Frame libraries](#3-frame-libraries)
- [4. Switching scenes on a running server](#4-switching-scenes-on-a-running-server)
- [5. Flags](#5-flags)
  - [5a. Camera server](#5a-camera-server)
  - [5b. Library builder](#5b-library-builder)
- [6. Verify](#6-verify)
- [7. Deploy](#7-deploy)
  - [7a. voltron (the GPU host)](#7a-voltron-the-gpu-host)
- [8. The DHS (xtalLoopSimDHS)](#8-the-dhs-xtalloopsimdhs)

## 1. Environment

One recipe on every host:

```bash
git clone <repo> && cd xtal-loop-sim
bash setup_venv.bash
```

`setup_venv.bash` builds `.venv/` from `requirements.txt` (torch 2.6.0 cu124 and pillow
10.4.0 pinned, `--only-binary=:all:`), prints the torch/CUDA/numpy/pillow versions, and
runs `pytest tests/`. `--force` rebuilds, `--skip-tests` skips pytest, `--acceptance`
also runs `tools/acceptance_voltron.py` (GPU). The base interpreter is
`/home/programs/pytorch/envs/pt/bin/python3.10` when present (the beamline hosts), else
`/usr/bin/python3`; never a Python 3.13, which has no pillow 10.4.0 wheel.

- **voltron:** the login shell is tcsh, so run it as `bash setup_venv.bash` as shown.
  When `/opt/rh/devtoolset-7` exists the script exports `CC`/`CXX` to it, which is what
  makes `torch.compile` work there.
- **dataserver3 / gateway:** no GPU. Serving from templates never imports torch, so the
  viewer runs fine; the CUDA-gated tests skip.

The DHS has its own venv:
[`xtalLoopSimDHS/README.md` "Create the env"](../xtalLoopSimDHS/README.md#6-create-the-env).

## 2. Run

```bash
# Live AXIS-compatible camera server; GPU engine when CUDA is present, templates otherwise
.venv/bin/python -m loop_sim.server.camera_server --scene data/scene_files/hampton_300um.yaml --port 8081

# One offline frame (CPU); add --device cuda for the GPU engine, --xray for a radiograph
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --n-cond 7 --output /tmp/out.jpg
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --rotx 45 --tx 0.05 --n-cond 1
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --xray --output /tmp/out_xray.png
```

Then open `http://<host>:8081/` for the control page, or point an AXIS consumer at
`http://<host>:8081/axis-cgi/mjpg/video.cgi`. `GET /xray` serves the radiograph live and
`GET /beam` the beam profile; X-ray has no frame library on either path.

`rotx` is the spindle for the bundled scenes. `scene.yaml`/`loop.yaml` are pipeline
outputs, gitignored and not shipped: render one of the three scenes in
`data/scene_files/` or build your own (README §3).

## 3. Frame libraries

A frame library is a 360° spindle sweep rendered once and replayed as crops, so the
camera answers instantly and nothing renders at request time. Libraries are tracked in
git under `data/frame_library/<scene>/` with a `manifest.json`.

The camera server never builds one. At launch and on every scene switch it grades the
library on disk against the flags it was started with: current serves; stale serves with
one warning naming what differs; missing renders live and prints the build command.

```bash
.venv/bin/python -m loop_sim.library --scene data/scene_files/<scene>.yaml   # builds only when missing
.venv/bin/python -m loop_sim.library --all
.venv/bin/python -m loop_sim.library --scene <s>.yaml --force                # the only way to rebuild
.venv/bin/python -m loop_sim.library --status --all                          # read-only
.venv/bin/python -m loop_sim.library --verify --scene <s>.yaml [--verify-angle DEG]
```

| scene | supersample | status |
|---|---|---|
| `hampton_300um` | 4 | current |
| `hampton_300um_realistic` | 4 | current |
| `mitegen_200um` | 1 | stale against the default of 4; served as-is (DECISIONS §2026-08-12 for why 1 is right for it) |

- **`--verify`** re-renders one stored frame live (CUDA unless `--allow-cpu`) and reports
  the worst and mean grey-level difference: PASS at 1 level or less, exit 1 on FAIL. Run
  it after editing any file that changes template pixels; `tests/test_render_sha_frozen.py`
  going red is the reminder.
- **Two graders can disagree and both be right.** `--status` grades against the build
  defaults, so `mitegen_200um` reads stale; the server grades against the flags it was
  launched with, so the same library reads current in `/scenes` unless you pass
  `--supersample 4`.
- **Builds need CUDA.** A CPU build is about 179 s per frame (18 h for a full sweep) and
  is refused without `--allow-cpu`; the viewer has no override. Build on voltron.
- **A build sizes itself to free VRAM and refuses rather than dying half-way**, naming
  the largest `--supersample` that fits. `LOOPSIM_VRAM_BUDGET_GB=12` mimics the TITAN V
  on a bigger card. Builds are atomic: a crash or Ctrl-C never touches the working
  library.
- Each rebuild writes about 15 MB into git history; do not rebuild casually.

## 4. Switching scenes on a running server

The control page has a tab per scene in `data/scene_files/`; clicking one swaps the
sample without restarting or dropping the MJPEG stream and resets the pose to home. From
a terminal:

```bash
curl -X POST 'http://host:8081/scene?path=mitegen_200um'   # 202 accepted
curl -s http://host:8081/scene                             # progress + errors
curl -s http://host:8081/scenes                            # library state of each
```

Tabs are badged with library state: no badge (current) and `stale` switch immediately;
`preview` means only a coarse on-demand library exists (5° steps); `no library` renders
live and the page offers a build, which is refused without CUDA. With `--templates off`
a switch drops the compiled preview (eager speed until restart) and briefly holds two
scenes in VRAM.

## 5. Flags

`--help` on either command lists everything with its default. The tables below add what
help text cannot: which flags are *library keys*. Changing a library key does not rebuild
anything; the affected scene reads `stale` and is served as-is until `--force`.

### 5a. Camera server

`python -m loop_sim.server.camera_server`

| Flag | Default | Effect |
|---|---|---|
| `--scene` | `data/scene_files/hampton_300um.yaml` | scene to serve |
| `--host` / `--port` | `0.0.0.0` / 8081 | bind address. Not 8080: on gateway that port is GitLab's puma |
| `--templates` | `on` | serve from a library when one exists, stale or not; never builds. `off` raytraces every frame (the correctness reference, needs a GPU) |
| `--engine` | `auto` | `torch` (GPU) / `numpy` (reference, minutes per frame) |
| `--n-cond` | 7 | condenser angles for settled frames |
| `--fps-limit` | 30 | MJPEG rate ceiling |
| `--jpeg-quality` | 85 | quality of frames sent, not of stored templates |
| `--preview-mode` / `--compile-preview` | `on` / `on` | approximate frames while moving, exact on settle; the preview path is `torch.compile`d on CUDA |
| `--settle-delay` | 0.5 s | quiet time after a `/motor` set before the exact frame renders |
| `--prewarm` | `on` | decode the whole library before the socket binds |
| `--template-cache` | `auto` | decoded templates held in RAM: up to half of available memory; `off` caps at 8; an integer pins it |
| `--camera-emulation` | `on` | map transmittance through the measured camera model (empty about 0.60, opaque about 0.18). Serve-time only |
| `--mono` | `off` | collapse to grey. Ignored with `--camera-emulation off` |
| `--pin-streak` | `on` | draw the pin's specular glint, projected from the scene. Ignored with `--camera-emulation off` |
| `--sensor-pitch` | `on` | deliver on the real camera's 704x480 non-square raster; `off` serves square pixels |
| `--supersample` / `--template-format` / `--template-quality` | builder defaults (4 / `png` / 90) | *library keys* |
| `--library-root` / `--preview-root` / `--scene-dir` | `data/frame_library/` / `data/frame_library_preview/` / `data/scene_files/` | where libraries, on-demand previews and switchable scenes live |
| `--jpeg-receiver` / `--push-fps` | none / 30 | URL that receives each frame as `image/jpeg` while `/video-trigger` is open (what pydhsfw's jpeg_receiver accepts), and its rate ceiling |
| `--camera-zoom` | `1:1.0,2:0.5,3:0.25` | AXIS camera number to zoom stop for `camera=N`; unknown `N` answers 400. A placeholder until measured (HANDOFF open items) |

### 5b. Library builder

`python -m loop_sim.library`

| Flag | Default | Effect |
|---|---|---|
| `--scene` / `--all` | none | one scene, or every `data/scene_files/*.yaml` |
| `--root` | `data/frame_library/` | output directory |
| `--step` | 1.0° | degrees between frames. *library key* |
| `--supersample` | 4 | render this many times finer than the camera pixel; the ceiling on zoom-in. *library key* |
| `--pan-mm` | 0.6 | translation margin beyond the scene. *library key* |
| `--n-cond` | 7 | condenser angles. *library key* |
| `--axis` | `rotx` | spindle motor; anything else is refused. *library key* |
| `--format` / `--quality` | `png` / 90 | stored template format; quality applies to jpeg only. *library keys* |
| `--psf` | `on` | bake the objective PSF into the templates. *library key* |
| `--tile-size` / `--vram-fraction` | `auto` / 0.80 | rays per trace pass and the share of free VRAM it may use. Do not change pixels |
| `--force` | off | the only way to rebuild an existing library |
| `--status` / `--verify` / `--verify-angle` | off / off / 0 | see §3 |
| `--preview` | off | build the coarse on-demand library (5° steps, 72 frames) into `data/frame_library_preview/` |
| `--recrop` | off | migrate a full-window library to content-only storage in place; no GPU |
| `--allow-cpu` | off | permit a CPU build |

## 6. Verify

```bash
.venv/bin/python -m pytest tests/ -q
```

272 tests across 21 files, green as of 2026-09-23. Fewer tests collected than that is
the signal to chase. numpy `divide by zero` RuntimeWarnings are expected. On a CPU-only
host the CUDA-gated parity tests skip, so a green run there is a weaker check.

On a 17 GB WSL2 box run one file at a time; a single `pytest tests/` has crashed WSL2:

```bash
for f in tests/test_*.py; do .venv/bin/python -m pytest "$f" -q; done
```

Benchmarks, both from the repo root:

```bash
.venv/bin/python tools/bench_serve.py --scene data/scene_files/hampton_300um_realistic.yaml --frames 40   # serve path, no GPU; GO/NO-GO against 10 fps, exit 0/1
.venv/bin/python tools/bench_frame.py [--compiled] [--modality xray]                                        # render path, GPU
```

Read `bench_serve.py`'s `slew_warm` line, not `slew`: the server pre-warms the library at
boot, so warm is what it serves.

## 7. Deploy

There is no install step and no service: loop-sim runs in place from a checkout. To
deploy a change, land the code on the target checkout, run `bash setup_venv.bash` when
`requirements.txt` changed, run the tests, and restart the camera server if one is
running (it holds the scene and compiled kernels in memory; there is no reload). Branch
state is in HANDOFF "Current state".

### 7a. voltron (the GPU host)

```bash
cd <checkout>
bash setup_venv.bash
.venv/bin/python -m loop_sim.server.camera_server --scene data/scene_files/hampton_300um_realistic.yaml --port 8081
```

That is the whole viewer: serving from templates never imports torch. The GPU is for
building libraries, `--templates off`, and the acceptance test:

```tcsh
setenv CUDA_VISIBLE_DEVICES 6      # a free card; check nvidia-smi first, voltron is shared
.venv/bin/python tools/acceptance_voltron.py     # GO/NO-GO: compile engaged and beat eager; writes acceptance_report.json
.venv/bin/python -m loop_sim.server.camera_server --scene data/scene_files/hampton_300um.yaml --port 8081 --templates off
sbatch tools/run_gpu.slurm         # batch alternative: CPU and GPU renders with diff stats; never add --time, the queue has no limit and it cancels jobs
```

- The login shell is tcsh: `setenv`, not `export`; call `.venv/bin/python` by full path,
  since venv `activate` is a bash script.
- A live-render server takes 1 to 2 min to compile the preview kernels at start.
- A `GLIBCXX ... not found` at import means the runtime libraries do not match the
  compiler: wrap the command in `scl enable devtoolset-7 "<command>"`.
- A busy card OOMs the mesh scene on arrival; pin a free one.

## 8. The DHS (xtalLoopSimDHS)

Build its `.venv/` per §1, then from `xtalLoopSimDHS/`:

```bash
./xtalLoopSimDHS.sh pretend            # no camera server; the same DCSS traffic
./xtalLoopSimDHS.sh real               # drives the camera server named in the config
.venv/bin/python -m pytest tests -q    # 27 offline tests, about 20 s
```

`real` mode needs the camera server up first, with `--jpeg-receiver <url>` if something
should receive the pushed frames. Wire contract, device table, the dcss database rows
and the no-BluIce driver: [`xtalLoopSimDHS/README.md`](../xtalLoopSimDHS/README.md).
