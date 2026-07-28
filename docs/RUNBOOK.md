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

Useful flags: `--step` (degrees between frames, default 1.0 → 360 frames), `--margin`
(render this much larger than the camera so panning is a crop, default 1.5), `--axis`
(spindle motor, default `rotx`), `--n-cond`, `--quality`.

Re-running is a **no-op when the library is current** — the manifest stores a SHA-256 of
the scene YAML, so an edited or newly added scene rebuilds automatically on first use.
From Python, `ensure_library(scene_path)` does the same thing and returns the manifest;
`frame_for_angle(manifest, deg)` picks the frame and `crop_window(manifest, tx_mm, ty_mm)`
gives the pan crop box.

Notes:

- **Panning is free, rotation is not.** Lateral translation is an exact image shift under
  this orthographic camera (measured max pixel difference 0.000000), so `tx`/`ty` are
  served by cropping inside the rendered margin. `crop_window()` raises if you ask to pan
  beyond it — rebuild with a larger `--margin` rather than clamping. `zoom` and `tz` are
  **not** covered and still need a live render.
- **Cost:** ~1.1 s/frame for `hampton_300um` at the default margin on an RTX 4080 SUPER,
  so a 360-frame sweep is a few minutes and lands around 6 MB of JPEG.
- **Droplet scenes will fail to build** with an explicit out-of-memory error until
  `TSurfaceMesh` gets its AABB cull (docs/HANDOFF.md risk A). The builder raises rather
  than quietly dropping resolution — a library rendered at a degraded setting is
  indistinguishable from a good one once it is on disk.
- **Mesh-bearing scenes cannot afford the default pan margin yet — same root cause.** The
  margin multiplies pixel count by `margin²`, and mesh memory scales with tile rays, so
  `mitegen_200um` at `--margin 1.5` needs ~2.25× its already-large working set. On a 16 GB
  card that lands at ~15.4/16.4 GB, which under WSL2 **spills to system RAM instead of
  raising OOM** and the build crawls rather than failing (see "Dev-environment caveat"
  below). Build mesh scenes with **`--margin 1.0`** until the AABB cull lands — the library
  is then rotation-only, with no free panning. Tube scenes take the default 1.5 fine.
  `frame_library/*/manifest.json` records the margin actually used, and `crop_window()`
  raises on any pan request a margin-1.0 library cannot serve.

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
`investigation/soak_server.py`. Note `investigation/` is Jacob's analysis tree — excluded
from the beamline mirror, so it may not be present in a copy you receive.

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
