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
- **Only tube/mesh scenes have a CUDA path.** `hampton_300um.yaml` (tubes) exercises it;
  `mitegen_200um.yaml` is primitives/CSG only, so `--device cuda` is a no-op there and CPU
  and GPU output is byte-identical by construction — not evidence the GPU path ran.
- `render.py` inserts `/home/jamesh/projects/loop_sim/claude` on `sys.path`; harmless when
  absent. Run from the repo root.
- Which motor is the spindle is scene-dependent: `rotx` for the bundled scenes.

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

**Before running on the beamline's TITAN V**, read DECISIONS.md "deploy target is a
TITAN V" — the 10 fps result was measured only on an RTX 4080 SUPER, and there are named
risks (VRAM headroom on the mesh scene; `torch.compile` failing silently). Nothing here
is certified on that hardware.

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
