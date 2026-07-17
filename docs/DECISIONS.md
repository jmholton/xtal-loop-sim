# DECISIONS — loop-sim (xtal-loop-sim)

<!-- Append-only, newest first. Never delete an entry; if superseded, add a new one that
     says so. This log exists so the next person understands WHY the code is the way it is
     and doesn't "fix" a deliberate choice, or spend a week re-deriving a measurement that
     already exists. ../CLAUDE.md has the code-level detail; this file is the reasoning and
     the evidence behind it. -->

## Decisions

### 2026-07-06 — 10 fps interactive via a flag-gated `torch.compile` preview path
- **Decision:** motion/preview frames route `next_interface` (and the tube `_kernel`,
  compiled *separately* with a dynamic survivor batch) through `torch.compile`; settled
  frames stay eager. Flags: `--compile-preview` (default on), `--preview-mode` (default on).
- **Why:** the frame is CPU-dispatch-bound (~8k kernel launches/frame); the pure-PyTorch
  dispatch ceiling is ~11 fps. `torch.compile` fuses the launch explosion (~1.7–1.86×),
  which is what clears the 10 fps bar.
- **Design constraints that are load-bearing:** CUDA-only (Inductor's CPU backend
  *miscompiles* the mesh/CSG path — mitegen came out 1734/6912 px wrong); `mode="default"`
  with `dynamic=True`; warm the compiled fn **single-threaded in `start()` before any
  worker thread** (concurrent first-compile crashes dynamo); the compiled/eager choice is
  threaded **explicitly** through the call chain, never shared mutable state (a shared
  `_active_compiled` flag caused a measured `/xray`-vs-bg-render data race); sticky
  eager fallback on any runtime failure.
- **What breaks if you change it:** Inductor reorders float arithmetic at tangent
  boundaries → shifts `best_t` ~1 ULP → flips ≤~16 genuinely-tangent CUDA pixels, so the
  strict `max==0` parity gate is relaxed to a **≤16 px CUDA flip budget** (CPU stays
  exact — compile is off there). Do not tighten that gate back to 0 on the compiled path.

### 2026-07-06 — converge-on-idle preview policy (approximate in motion, exact at rest)
- **Decision:** during motion the server renders `n_cond=1` (+ optionally compiled)
  previews; a full-`n_cond` **bit-exact f64** frame renders automatically once the pose
  settles. Offline renders and `/xray` are always exact. This applies to instant `/motor`
  sets too (the AXIS-consumer path): a stream of `/motor` updates gets fast previews, one
  exact frame on quiet (`--settle-delay`, default 0.5 s).
- **Why:** loop-sim's real purpose is a drop-in AXIS camera producing ~10 image/s; motion
  smoothness matters, per-frame exactness during motion does not. Settled output is never
  degraded, so nothing permanent is sacrificed (test-gated: `test_server_settle_parity`).
- **What breaks if you change it:** making motion frames "exact" collapses fps; the
  invariant is *exact-when-settled*, not exact-always.

### 2026-06-25 — GPU-resident torch engine, byte-identical to the numpy reference (f64)
- **Decision:** `loop_sim/renderer/engine_torch.py` runs the *whole* trace on-device
  (`TorchScene`, `render_torch`); the numpy `microscope.py` path is **FROZEN as ground
  truth** and kept as the CPU fallback. The active ray set uses **compaction**, not masking.
- **Why:** profiling showed ~84% of the render was numpy on the CPU — only the
  intersection touched CUDA, so the GPU sat ~84% idle. Keeping all ray state resident (1
  upload / 1 download per frame, condenser batched) was the real speedup (~6–8×), not
  faster kernels.
- **What breaks if you change it:** do not let the torch engine diverge from the numpy
  reference — it is both the correctness anchor and the no-GPU fallback. Parity is
  enforced by `tests/test_torch_render_parity.py` (incl. full-res CUDA).

### 2026-06-25 — float64 in the GPU intersection quadratic (THE correctness fix)
- **Decision:** `TTube`/`TSurfaceMesh` compute the intersection quadratic in float64
  internally, always — even on CUDA. (`tube.py`, `surface_mesh.py`, `engine_torch.py`.)
- **Why:** float32 **catastrophically cancelled** in `c_ = baba*oaoa - baoa² - r²*baba`,
  amplified by the microscope's 50 mm ray-march origin offset. For a ~7.5 µm fiber 50 mm
  upstream, `oaoa ≈ 2500 mm²` swamps the r² signal (~6e-9, below the float32 noise floor
  ~1.6e-8) → discriminant sign flips → binary hit/miss + a **median ~9 µm t-error** → the
  hit point lands off-surface → poisoned normals (~19–32°) → Snell sends the ray ~20° off →
  it trips the hard binary NA cutoff / TIR (NaN→0) gates → full 0↔255 pixel flips = the
  speckled **"hairy/spikey" fiber**. Decisive isolation: forcing *only* the tube
  intersection to f64 (rest stays CUDA) dropped the hampton roty45 n1 diff **203 px → 0**.
- **What breaks if you change it:** down-casting the intersection geometry or inputs to
  float32 brings the artifact straight back. This is also *why* fp32 preview is rejected
  (below). (`tests/test_gpu_cpu_parity.py` guards it.)

### 2026-06-25 — threaded camera server
- **Decision:** `CameraServer` inherits `ThreadingHTTPServer` (one thread per connection).
- **Why:** a plain single-threaded `HTTPServer` was starved by a browser's idle preconnect
  socket — the MJPEG stream "loaded forever" and `/motor` couldn't be served while
  streaming.

### (inherited from James's model — recorded so they aren't "cleaned up")
- **The twisted-pair stem must be TWO separate tube objects** (`stem_1`, `stem_2`). A
  single helical tube models only one fiber; the crossover shadows disappear and the stem
  looks wrong.
- **Crystal must be listed before droplet** in scene priority. A crystal voxel is also
  inside the droplet; if droplet is first, every crystal voxel is assigned `solvent` and
  the crystal renders invisible.

### 2026-07-14 — deploy target is a TITAN V; GPU speed is NOT the bottleneck (analysis)
- **Decision/finding:** the beamline runs loop-sim on voltron's **TITAN V** (Volta,
  12 GB), but the 10 fps result was only ever measured on an RTX 4080 SUPER. Profiling
  shows the frame is **CPU-dispatch-bound** (73–85% self-CPU, GPU ~29% busy, ~52–65 W of
  320 W), so the TITAN V's 8.5× FP64 advantage is a **red herring** — the dominant ops are
  low-intensity elementwise, not FP64-ALU-bound. The real deployment risks are A (12 GB
  mesh OOM), B (silent `torch.compile` fallback), C (unknown voltron CPU) — see HANDOFF
  "Hazards".
- **Verified measurements (don't re-derive):** hampton n_cond=1 f64, worst-case
  centred-loop `/motor` pose — eager **147.5 ms (6.8 fps)**, compiled **88.7 ms (11.3
  fps)**; per-launch ~**8.25 µs** on WSL2 GPU-PV (native Linux is typically 3–5 µs, so
  voltron *may be faster* CPU-side). **L2 inversion:** the `(N,3)` f64 chain is L2-served
  on the 4080 (64 MB L2) but a 7 MB working set won't fit the TITAN V's **4.5 MB** L2 → it
  falls to HBM2 → real GPU regression ≈**1.5×** (still hidden unless the GPU must be ~2.6×
  slower to bottleneck — margin is thin; this is the #1 thing `ncu` should settle).
- **What breaks / traps:** **byte-exact gates are architecture-blind** — they compare
  against a reference computed on the *same* machine, so a Volta box could pass every gate
  while producing wrong images. Add a numpy-anchored golden-image gate before trusting the
  target. Tooling notes: **CUPTI works under WSL2** (an older note claiming otherwise is
  wrong); `torch.profiler` reports zero GPU time due to a CLOCK_REALTIME vs
  CLOCK_MONOTONIC mismatch (fix: env `CuptiUseRawGpuTimestamps=false`). `bench_results/`
  is gitignored → commit baselines if you want them to travel to voltron.
- **Status:** the profiling/acceptance suite is **PLANNED, zero code written.** Intended
  deliverable: a self-contained `acceptance_voltron.py` the beamline staff run themselves
  (no Nsight needed) → one JSON back, making the perf prediction falsifiable. Cheapest
  first experiment: the **OOM canary** — `torch.cuda.set_per_process_memory_fraction(12/16)`
  caps the 4080 to a Titan-V-sized 12 GB *today*.

## Already Tried

<!-- Evidence, not fences. Each entry is here so nobody spends a week re-deriving a
     number that already exists. -->

### fp32 preview mode
- **What was measured:** on the sync-starved compiled engine, fp32-compiled is **~2×
  SLOWER** than f64-compiled (≈188 ms vs 120–126 ms on worst-case poses), and fp32-eager
  also loses on the worst case. The f32↔f64 cast traffic at the **deliberately-float64
  tube-kernel boundary** outweighs the halved memory bandwidth, and fp32 perturbs the
  material-probe interval enough to change ray-path work profiles. The often-cited June
  "~2× fp32 win" microbench **predates this pipeline** and no longer holds. Reproduce with
  `bench_frame.py --fp32` (the negative result is recorded in commit `fb38fdb`).
- **Would be worth another look if:** the whole engine went fp32-native. Note the
  constraint that shapes this: the intersection quadratic has to stay f64 (see the
  correctness decision above), so any fp32 scheme has to keep that boundary — which is
  exactly where the cast cost showed up.

### `torch.compile(mode="reduce-overhead")` / CUDA graphs in the threaded server
- **What was measured:** `reduce-overhead` silently enables CUDA-graph capture, which is **not
  thread-safe** in the `ThreadingHTTPServer` ("already recording to mempool_id" /
  `CUBLAS_NOT_INITIALIZED`). Whole-frame graph capture is *also* blocked by the AABB cull's
  data-dependent `nonzero` (a graph-break), and dropping the cull to enable capture is a
  **catastrophic 1.95 fps** (a 50× tube blow-up). The fast algorithm (AABB cull) and
  whole-frame capture are mutually exclusive.
- **Would be worth another look if:** a capture strategy that tolerates dynamic shapes.

### custom Triton megakernel (one thread per ray, whole bounce loop in registers)
- **What was measured:** proven **byte-identical** but only **~1.1×** (54 ms vs torch 60 ms) on
  this fp64 thin-fiber workload — it's GPU-compute-bound on fp64 and trades dispatch
  overhead for dense-depth compute, a wash on this geometry. Not worth the complexity; it
  was a throwaway prototype and no code remains.
- **Would be worth another look if:** fp32 (with its cancellation risk) + a single-pass
  tube + 2D ray tiling — uncertain.

### dense alive-mask `trace_rays` (instead of compaction)
- **What was measured:** byte-identical but **regresses** (5.6→6 fps — it runs 12 full-N depths
  vs ~3 with compaction), and CUDA graphs still don't engage (the cull's `nonzero`
  graph-breaks). Compaction is the correct design.

### "fix the hairy fiber" via normals, gate-softening, or more condenser rays
- **What was measured (all four refuted):** the root cause is the float32 intersection **t**, not the
  normals or the gates. `recompute_normals_f64` is already optimal given a correct t (0°
  error); forcing CUDA's `best_k` makes normals *worse*; softening the NA/TIR gates catches
  ~0% of the flips; and **raising n_cond makes the fiber hairier** (the flip count roughly
  doubles), not smoother. Float64 at the quadratic is the only fix.
