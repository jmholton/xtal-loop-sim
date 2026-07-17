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

## Current state (2026-07-17)

- **Branch `performance-correctness-optimizations`, HEAD `edefca0`, tree clean, ~20
  commits ahead of `master`, NOT pushed to GitHub.** James owns the push/merge decision.
  (`master` itself is 22 commits ahead of the stale GitHub default `main`, which is a
  divergent "Initial commit" — always work from `master`/this branch, never `main`.)
- **Correctness: DONE + committed.** The float32 "hairy/spikey fiber" GPU artifact is
  fixed — the CUDA intersection quadratic now runs in float64, and the GPU render is
  **byte-identical to the float64 CPU reference**. See DECISIONS.md §"float64 GPU
  intersection" for the root cause.
- **Performance: 10 fps interactive goal MET — and now confirmed on the beamline's TITAN V
  (11.9 fps).** A GPU-resident torch engine (`loop_sim/renderer/engine_torch.py`) runs the
  whole trace on-device; on the 4080 the live server reaches ~25 fps during animated motion
  and ~9.8 fps on the worst-case 10 Hz `/motor` stream, via a flag-gated `torch.compile`
  preview path. `acceptance_voltron.py` measured the deployed path on a real TITAN V at
  **11.9 fps median / 10.1 fps p90 (GO)** — but only with the full software stack (torch 2.6
  + a modern compiler for `torch.compile`); the beamline's default stack falls back to eager
  at 6.3 fps. Settled/offline/`/xray` frames stay bit-exact f64. See RUNBOOK "Deploy on the
  TITAN V" for the exact recipe and DECISIONS.md.
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
4. **Beamline (TITAN V) readiness is now measured** — the 10 fps result reproduces on the
   real card (11.9 fps) *when the software stack is right* (RUNBOOK "Deploy on the TITAN V").
   The remaining work is packaging that stack, not proving the hardware.

The highest-value open engineering items, in rough priority:
- **Package the TITAN V deployment** (the recipe is measured; see RUNBOOK "Deploy on the
  TITAN V"): a torch-2.6 env + a modern compiler for `torch.compile`, plus making the
  silent-fallback-to-eager failure loud so a mis-set stack can't quietly miss 10 fps.
- **Wire `render.py --device cuda` to the resident engine** — it still uses the legacy
  per-object CUDA path; only `camera_server` uses `engine_torch`. Unifying them removes a
  confusing second GPU path.
- **Give `TSurfaceMesh` the AABB cull that `TTube` has** — fixes both the mesh OOM risk and
  its ~1.9 s render (mesh scenes only, e.g. `mitegen_200um`).
- **Click-to-recentre bug** (paused) — lands ~100–200 px off, non-deterministically;
  leading hypothesis is a frame/pose lag race. Full resume plan in `../CLAUDE.md` §"Click-to-
  recentre" and DECISIONS.md.

## Hazards & gotchas

**Deployment reality — the 10 fps target reproduces on the TITAN V (11.9 fps), but only with
the full software stack; the hardware was never the bottleneck, the beamline's default
software stack is.** Measured on a real voltron TITAN V (2026-07-17, `acceptance_voltron.py`,
torch 2.6, devtoolset-7): compiled preview **11.9 fps median / 10.1 fps p90 (GO)**, eager
fallback **6.3 fps**. The three things that decide whether you get 11.9 or 6.3:

- **A — the mesh scene's VRAM is a knife's-edge fit, not a hard OOM.** `TSurfaceMesh` has no
  AABB cull, so `mitegen_200um` (n_cond=1) brute-forces Möller-Trumbore to **~11.1 GB
  reserved**. On the TITAN V's 12 GB that *fits on torch 2.6* (its allocator packs it in),
  but with only ~tens of MB free once the CUDA context is counted — and it **OOMs on torch
  2.0.1** (worse fragmentation). So: fine on a free card with a modern torch, but one
  concurrent allocation or a busier card from the edge. **Fix (verified byte-exact):
  sub-frame tiling** — `render_torch(..., tile_size=32768)` drops mitegen to ~1.2 GB
  (`engine_torch.py:1003` `tile_size = max(tile_size, WH)` is a *perf* clamp, not correctness
  — per-ray results are tile-independent). The durable fix is giving `TSurfaceMesh` the
  `_aabb_survivors` cull `TTube` already has (`engine_torch.py:59`). Tube scenes like hampton
  are unaffected (~250 MB).
- **B — `torch.compile` silently falls back to eager, and on the beamline it WILL fail
  without help.** compiled ≈ 11.9 fps vs eager ≈ 6.3 fps. `camera_server.py`
  `_warmup_compiled_preview` (≈line 714) and the runtime path (≈line 478) catch **any**
  exception and degrade to eager while the server keeps booting and streaming — so a stack
  that can't compile silently misses 10 fps. On voltron, compile fails out of the box two
  ways: the pt env's **torch 2.0.1** has an Inductor `pkg_resources` bug, and even on torch
  2.6 the **system gcc 4.8.5** is too old for Inductor's codegen (`stdatomic.h`). Both are
  cleared by the RUNBOOK recipe (torch 2.6 + devtoolset-7). **Make the fallback loud** (fail
  hard, or a persistent warning) so a mis-set stack can't hide, and **pin torch** — it is not
  listed in `requirements.txt` at all (the CUDA build is site-specific: cu124 on the dev box,
  cu118 on voltron).
- **C — voltron's CPU is a 2× Xeon E5-2650 v4 (2016), and it does NOT tank the compiled
  path.** The eager path is dispatch-bound, so the slow CPU shows there (6.3 fps); but the
  compiled path fuses ~8k kernel launches into ~8 graphs, becoming GPU-bound — so the weak
  CPU stops mattering and voltron's compiled 11.9 fps matches the dev box. (This was an open
  worry that the old CPU would drag the frame to ~5–6 fps; measurement refuted it — but
  *only* because compile works, which loops back to risk B.)

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

- **Push/merge decision for `performance-correctness-optimizations`** — owner: James. Until
  pushed, the branch lives only on this tree + the gateway mirror.
- **Package the TITAN V software stack.** The 11.9 fps result needs torch 2.6 + a modern
  compiler at runtime (RUNBOOK "Deploy on the TITAN V"), assembled by hand in a venv. A
  reproducible env (a pinned recipe, or a launch wrapper that sets `CC`/`CXX`) would make
  deployment turnkey instead of a five-step manual setup.
- **Make the silent compile-fallback loud** (risk B) and **declare torch 2.6** as required —
  a mis-set stack currently misses 10 fps with no signal.
- **The perf-validation tooling in `investigation/` is push-excluded** (`soak_server.py`,
  profiling experiments); `bench_frame.py` and `acceptance_voltron.py` (repo root) ARE
  shipped. `acceptance_voltron.py` is the one-command GO/NO-GO check — run it on the target
  GPU and read the printed verdict + `acceptance_report.json`.

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
- `bench_frame.py` — warm-frame benchmark (`--compiled`, `--fp32`). `acceptance_voltron.py`
  — self-contained TITAN V acceptance test (fps + VRAM + compile check → GO/NO-GO +
  `acceptance_report.json`; auto-picks a free GPU). `run_gpu.slurm` — voltron GPU job (no
  `--time`!). `tests/` — 62 tests (the verify command).
- `README.md` — user guide (repo root). `CLAUDE.md` — deep engineering notes (repo root:
  architecture, precision, concurrency, the recentre bug). `docs/` — the handoff docs
  (this file + `RUNBOOK.md`, `DECISIONS.md`, `DATA.md`). `investigation/` — **not
  shipped**; experiment scratch + perf harnesses.

## Work log (append-only)

- **2026-07-17** — TITAN V acceptance measured. Added `acceptance_voltron.py` (`df33635`), a
  self-contained one-command GO/NO-GO harness (fps + VRAM + compile check), and ran it on a
  real voltron TITAN V. Result: compiled preview **11.9 fps median / 10.1 fps p90 (GO)**,
  eager 6.3 fps; mesh scene fits torch 2.6 at ~11.1 GB (knife's edge), OOMs torch 2.0.1. The
  10 fps goal reproduces on the target **with** the stack in RUNBOOK "Deploy on the TITAN V"
  (torch 2.6 + devtoolset-7); the beamline's default stack (torch 2.0.1, gcc 4.8.5) can't
  compile and silently falls back to 6.3 fps. Resolved the "voltron CPU unknown" question
  (2× Xeon E5-2650 v4) and corrected two earlier predictions (mesh isn't a hard OOM; the old
  CPU doesn't tank the compiled path). **Next:** package the stack + make the silent fallback
  loud; James's call on pushing the branch.
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
