---
project: loop-sim (xtal-loop-sim) — bright-field microscope + X-ray simulator for protein crystals in cryo-loops
status: paused
last_verified: 2026-07-16        # `pytest tests/` = 62 passed on this tree (branch performance-correctness-optimizations)
verify: python -m pytest tests/ -q        # 62 tests; "python" = the torch-enabled project interpreter (see docs/RUNBOOK.md "Environment")
contacts:
  - James Holton — original author & physics owner — owns the optical + X-ray model, the beamline deployment on voltron (TITAN V), and the GitHub repo jmholton/xtal-loop-sim
  - Jacob Doughty — departing contractor — authored the float64 GPU correctness fix, the GPU-resident torch engine, and the 10 fps optimization (all on branch performance-correctness-optimizations)
---

# HANDOFF — loop-sim (xtal-loop-sim)

<!-- Single entry point. Read this first; it links out to ../README.md (user guide) and
     ../CLAUDE.md (deep architecture/engineering notes) rather than duplicating them.
     This file lives in docs/; repo paths below are relative to the repo root. -->

## What this is, and why

loop-sim renders a **synthetic bright-field microscope image** of a protein crystal
mounted in a nylon cryo-loop (Snell refraction at every interface + Beer-Lambert
absorption + Köhler condenser illumination), and separately **grid-casts the X-ray beam**
through the same scene to report illuminated volume and dose per material. Output is a
JPEG **or** an AXIS-camera-compatible MJPEG HTTP stream that can stand in for a real
beamline camera. It exists for beamline alignment-algorithm development, AI training-data
generation, and dose estimation. James Holton wrote it; Jacob's contribution was making
the GPU path **correct** (it was producing a "hairy" artifact on the loop fiber) and
**fast enough to drive as a live camera** (10 image files/s).

## Current state (2026-07-15)

- **Branch `performance-correctness-optimizations`, HEAD `edefca0`, tree clean, ~20
  commits ahead of `master`, NOT pushed to GitHub.** James owns the push/merge decision.
  (`master` itself is 22 commits ahead of the stale GitHub default `main`, which is a
  divergent "Initial commit" — always work from `master`/this branch, never `main`.)
- **Correctness: DONE + committed.** The float32 "hairy/spikey fiber" GPU artifact is
  fixed — the CUDA intersection quadratic now runs in float64, and the GPU render is
  **byte-identical to the float64 CPU reference**. See DECISIONS.md §"float64 GPU
  intersection" for the root cause.
- **Performance: 10 fps interactive goal MET — but only measured on Jacob's RTX 4080
  SUPER, never on the beamline's TITAN V.** A GPU-resident torch engine
  (`loop_sim/renderer/engine_torch.py`) runs the whole trace on-device; the live server
  reaches ~25 fps during animated motion and ~9.8 fps on the worst-case 10 Hz `/motor`
  stream, via a flag-gated `torch.compile` preview path. Settled/offline/`/xray` frames
  stay bit-exact f64.
- **Verify: `pytest tests/` = 62 tests, green** on the local torch env (needs a
  torch+CUDA interpreter; GPU-gated parity tests skip on a CPU-only box).
- **Paused with clear open items** (see below) — nothing half-broken; the engine works.

## How to resume

For a stranger picking this up cold:

1. Build the environment and confirm health: follow **`RUNBOOK.md`** → run `python -m pytest
   tests/ -q` (should be 62 green). "python" is the torch-enabled interpreter — beamline:
   `/programs/pytorch/envs/pt/bin/python`; local dev: a conda env with `torch==2.6.0+cu124`.
   On a CPU-only box the GPU parity tests skip, so green there proves less.
2. Understand the design before editing: **`../CLAUDE.md`** is the deep engineering doc
   (architecture map, the float64 precision rationale, the server concurrency model, the
   `next_interface` API contract, the comparison workflow). `../README.md` is the user guide
   (pipeline, CLI, live server, endpoints).
3. Decide the **push/MR** for branch `performance-correctness-optimizations` (James) — the
   work is durable only on this local tree + the gateway mirror until then.
4. The single most important open question is **beamline (TITAN V) readiness** — see
   Hazards below; the 10 fps result is unproven on the target and has zero margin.

The highest-value open engineering items, in rough priority:
- **De-risk the TITAN V deployment** (the three risks in Hazards). Cheapest first: the OOM
  canary (`torch.cuda.set_per_process_memory_fraction(12/16)` caps the 4080 to Titan-V's
  12 GB *today* and turns "mitegen will probably OOM" into a hard fact), and make the
  silent `torch.compile` fallback loud.
- **Wire `render.py --device cuda` to the resident engine** — it still uses the legacy
  per-object CUDA path; only `camera_server` uses `engine_torch`. Unifying them removes a
  confusing second GPU path.
- **Give `TSurfaceMesh` the AABB cull that `TTube` has** — fixes both the mesh OOM risk and
  its ~1.9 s render (mesh scenes only, e.g. `mitegen_200um`).
- **Click-to-recentre bug** (paused) — lands ~100–200 px off, non-deterministically;
  leading hypothesis is a frame/pose lag race. Full resume plan in `../CLAUDE.md` §"Click-to-
  recentre" and DECISIONS.md.

## Hazards & gotchas

**Deployment risk — the beamline runs on a TITAN V, and the 10 fps result was only ever
measured on an RTX 4080 SUPER.** GPU *speed* is NOT the risk (the frame is CPU-dispatch-
bound, ~73–85% self-CPU; FP64 throughput is a red herring). The three real risks:

- **A — 12 GB VRAM cliff (most likely hard failure).** `TSurfaceMesh` has no AABB cull, so
  a mesh scene (`mitegen_200um`, n_cond=1) brute-forces Möller-Trumbore and measured
  **10.5 GB allocated / 11.2 GB reserved** on the 4080. The TITAN V has **12 GB** → almost
  no headroom → it will OOM the moment anything else touches the GPU. **Fix (verified
  byte-exact): sub-frame tiling** — `render_torch(..., tile_size=32768)` drops mitegen to
  ~1.2 GB (`engine_torch.py:1003` `tile_size = max(tile_size, WH)` is a *perf* clamp, not a
  correctness one — its own comment says per-ray results are tile-independent). The real
  fix is giving `TSurfaceMesh` the `_aabb_survivors` cull `TTube` already has
  (`engine_torch.py:59`).
- **B — silent `torch.compile` fallback (owns the entire 10 fps result).** compiled ≈ 11
  fps vs eager ≈ 6.8 fps = a 32% miss without compile. `camera_server.py`
  `_warmup_compiled_preview` (≈line 714) and the runtime path (≈line 478) catch **any**
  exception and degrade to eager while the server keeps booting and streaming — so on a
  Volta (sm_70) box where compile fails, the beamline silently gets a server that runs but
  misses 10 fps. **Make this loud** (fail hard, or a prominent persistent warning), and
  **pin torch in `requirements.txt`** (`torch==2.6.0+cu124`) — it is currently not listed
  there at all, and CUDA 13 / PyTorch 2.11 both drop Volta. (Partial de-risk: Triton 3.2
  *did* compile an f64 kernel targeting sm_70, so Volta codegen is not fundamentally
  broken — but that's codegen, not proof of the fused-graph perf.)
- **C — voltron's CPU is unknown.** The frame is 73–85% self-CPU, so the host CPU matters
  more than the GPU. Per-launch cost on Jacob's WSL2 box is ~8.25 µs; native Linux is
  typically 3–5 µs, so loop-sim **may be faster** on voltron — but nobody has the numbers.
  See Open questions.

**Other traps:**
- **`--device cuda` only does something on scenes with `Tube` or `SurfaceMesh` objects.**
  Those are the only shapes with a CUDA path, so a scene of pure primitives/CSG
  (half-spaces, cylinders, spheres) renders CPU==GPU byte-identical and `--device cuda` is
  a no-op. Use `hampton_300um` (tube-based) to exercise the tube GPU path; `mitegen_200um`
  (mesh-based) exercises the `SurfaceMesh` path (and is the OOM risk in A above).
- **Never down-cast the intersection geometry to float32** — it reintroduces the hairy-fiber
  artifact (DECISIONS.md).
- **Never use `torch.compile(mode="reduce-overhead")` in the server** — its CUDA-graph
  capture is not thread-safe in this threaded server and crashes.
- **SLURM: do not set `--time`** in `run_gpu.slurm` — voltron's queue has no time limit and
  the flag cancels jobs prematurely (`../CLAUDE.md`).
- **`scene.yaml`/`loop.yaml` are gitignored, NOT shipped** — render a `scene_files/*.yaml`
  (e.g. `hampton_300um.yaml`, `mitegen_200um.yaml` are complete) or build one via the
  pipeline (README).
- **The `investigation/` dir is NOT shipped** (excluded from the gateway push) — see Open
  questions; the soak-test and profiling harnesses live there.

## Open questions

- **Ask the beamline for voltron's `lscpu` + `nvidia-smi -q`** (a text paste). The frame is
  CPU-bound; this is the cheapest way to firm up risk C. (Per policy, the beamline relays
  it — never fetched directly.)
- **Push/merge decision for `performance-correctness-optimizations`** — owner: James. Until
  pushed, the branch lives only on this tree + the gateway mirror.
- **The perf-validation tooling lives in the push-excluded `investigation/` dir** (e.g.
  `soak_server.py`, the profiling experiments). If the team wants to re-validate perf or
  build the planned Titan-V acceptance harness, those need moving into the repo (e.g. a
  `tools/` dir). `bench_frame.py` (repo root, `--compiled`/`--fp32`) IS shipped.
- **The planned TITAN V profiling suite is designed but unwritten** (zero code). Its intent:
  a self-contained `acceptance_voltron.py` the beamline runs to produce one JSON that makes
  the perf prediction falsifiable. Design captured in DECISIONS.md §"TITAN V deployment".

## Decisions

See **[DECISIONS.md](DECISIONS.md)** — it carries the float64 correctness fix, the
GPU-resident-engine and 10 fps architecture, the TITAN V deployment analysis, and an
**"Already tried"** record holding the measurements behind fp32 preview, CUDA-graph /
reduce-overhead in the threaded server, the Triton megakernel, and dense-mask tracing, so
those numbers don't have to be re-derived.

## Map of the repo

<!-- Paths relative to the repo root. One line per top-level thing;
     ../CLAUDE.md §"Architecture overview" has the full tree. -->

- `render.py` — CLI: load scene, drive goniometer, render to JPEG.
- `loop_sim/` — the package: `scene/` (YAML loader, `next_interface`, primitives, `tube.py`,
  `surface_mesh.py`, CSG), `motors/goniometer.py`, `renderer/` (`microscope.py` numpy
  reference tracer, `beam.py` X-ray, **`engine_torch.py`** GPU-resident engine),
  `server/camera_server.py` (AXIS HTTP server + control page).
- `digitize_fiber.py → add_stem.py → add_droplet.py → add_crystal.py → generate_scene.py`
  — the pipeline that builds a scene from a real loop image (README).
- `scene_files/` — complete example scenes (`hampton_300um.yaml` tube-based;
  `mitegen_200um.yaml` mesh-based). `template.yaml` — camera/material properties.
- `bench_frame.py` — warm-frame benchmark (`--compiled`, `--fp32`). `run_gpu.slurm` — voltron
  GPU job (no `--time`!). `tests/` — 62 tests (the verify command).
- `README.md` — user guide (repo root). `CLAUDE.md` — deep engineering notes (repo root:
  architecture, precision, concurrency, the recentre bug). `docs/` — the handoff docs
  (this file + `RUNBOOK.md`, `DECISIONS.md`, `DATA.md`). `investigation/` — **not
  shipped**; experiment scratch + perf harnesses.

## Work log (append-only)

- **2026-07-16** — RUNBOOK.md written (env from nothing → run → verify → voltron/SLURM →
  rollback), completing the doc set. Verify command re-run on this tree: **62 passed in
  67 s** (RTX 4080 SUPER). **Next:** de-risk the TITAN V deployment (risks A/B/C above);
  James's call on pushing the branch.
- **2026-07-15** — Scaffolded HANDOFF.md + DECISIONS.md as the first knowledge-transfer
  flush of the loop-sim memories (bug-hunt root cause, 10 fps architecture, rejected
  approaches, TITAN V risks — none of which were written down in the repo before; they
  lived only in the contractor's agent memory + the push-excluded `investigation/` dir).
  Verified: git HEAD `edefca0`, tree clean, `pytest tests/` green (62 tests) on the local
  torch env.
- **2026-07-06** — 10 fps interactive goal met (RTX 4080 SUPER): compiled preview path,
  single-flight render owner, sync-starved trace loop; fp32 preview tried and rejected.
  62 tests green. (James owns push.)
- **2026-06-25** — float64 GPU correctness fix (hairy fiber gone, GPU byte-identical to CPU)
  + GPU-resident torch engine + threaded camera server. Branch created off `master`.
