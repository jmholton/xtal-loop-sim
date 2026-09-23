# WORK LOG: loop-sim (xtal-loop-sim)

Dated history, newest first. The reasoning behind each change is in DECISIONS.md under the same date; live state is in HANDOFF.md.

## 2026-09-23

- xtalLoopSimDHS exercised against the local dcss rig seeded with `sandbox/LOCAL_loopsim.txt`: dcss registered the five motors and three shutters, and `gtos_start_motor_move gonio_phi 90`, `gtos_start_motor_move sample_x 0.2` and `gtos_start_oscillation gonio_phi video_trigger 30 2` each completed `normal` with the camera server's pose following; the oscillation pushed 59 JPEGs to a receiver on port 9000. BluIce's phi buttons drove the spindle; its click-to-centre failed on the rig's missing `moveSample` operation. New `sandbox/drive_dcss.py` reproduces the check without BluIce.
- The 2026-09-22 work was committed as `c0e6415`, `ece4895`, `342d7c9`, `9c3c4b8`.

## 2026-09-22

- X-ray radiograph library retired: `xray_library/`, `xray_library.py`, the `/xray-stream` push stream, and the Microscope/Radiograph toggle deleted; `GET /xray` renders the live pose on every request instead. New `render.py --xray` writes a radiograph PNG beside a scene, a flag James's master never had. DECISIONS §2026-09-22.
- Frame-library staleness reworked: `render_sha` dropped out of `_BUILD_KEYS`, so a renderer edit no longer grades a library stale on its own; builds are now atomic (`.new`/`.old` swap); the CLI gained `--status`, `--verify`, and `--force`. A bare server launch never rebuilds: it serves a stale library with a warning, or renders live. DECISIONS §2026-09-22.
- `xtalLoopSimDHS/` added: a pydhsfw hardware DHS that drives the goniometer's real motors over dcss and talks to the camera server over localhost HTTP; `camera_server.py` gained `GET /status`, `/move?duration=`, `/video-trigger`, and per-camera zoom-stop serving for it to use. 27 offline tests green; not yet exercised against a running dcss/BluIce. DECISIONS §2026-09-22.
- Repo reorganized: scenes, frame libraries, and reference photos moved under `data/`; benchmark tooling moved under `tools/`; James's seven root scripts stay at the root. Conda retired; `setup_venv.bash` now builds one `.venv/` from `requirements.txt` on any host. DECISIONS §2026-09-22.
- Test suite: 274 tests across 21 files, all green in the root `.venv`.
- Committed the next day as four commits (see 2026-09-23).

## 2026-09-10

- Root scripts (`test_optim`/`test_gpu`/`debug_optim`/`check_diff` .bash, `run_gpu.slurm`, `profile_render`/`profile_gpu` .py/.bash, `make_beam_image.py`) repointed at this checkout instead of the legacy `/home/jamesh` tree. `2af1cef`.
- Dead code deleted: `Scene.material_at` family, `SurfaceMesh` CPU grid, `_mesh_face_count`, `surface_evolver_script`, the elastica ODE helpers. 362 lines removed, manifests re-stamped. `0565816`.
- HANDOFF.md condensed; this file split out of it.

## 2026-08-19

- `/xray-stream` mirrors the optical MJPEG producer/consumer pattern with entirely separate state; `POST /stream-mode?mode=microscope|radiograph` toggles the X-ray producer alone without touching the optical one. `f0bf65d`.
- Both template caches (optical and X-ray) now prewarm at boot and on every scene switch; combined prewarm 8.8 s, streaming throughput ~28 fps while moving.
- All three real 360-frame X-ray libraries built: `mitegen_200um` 1.01 s/frame, `hampton_300um` 2.18 s/frame, `hampton_300um_realistic` 13.58 s/frame mean (81.5 min); 39 MB total. `fa44e35`.
- Viewer gained a Microscope/Radiograph toggle and a `/beam` readout panel; `xray_library.status` added to `/scenes`.
- DECISIONS §2026-08-19.

## 2026-08-18

- `render.py --device cuda` switched to the resident `engine_torch` engine, retiring the separate legacy per-object CUDA path.
- `/beam` and `/xray` held `_scene_lock` across their whole render (21-57 s on the flagship scene), freezing the live camera stream; fixed by snapshotting the pose and scene under the lock, then releasing before rendering.
- X-ray rendering split out of `engine_torch.py` (hashed into `render_sha`) into `renderer/xray_torch.py`, so X-ray-only edits stop invalidating the shipped optical libraries.
- New `loop_sim/library/xray_library.py`: 16-bit greyscale X-ray radiograph frame library mirroring the optical library's geometry.
- Kept illustrative (uncalibrated) `mu_xray` over literature-real absorption coefficients. DECISIONS §2026-08-18. Uncommitted at the time, landed in `fa44e35`.

## 2026-08-14

- Templates now store only their measured content instead of the full centred window; stored resolution dropped sharply (for example `hampton_300um` 5578x2570 to 3709x414). Delivered frame unchanged at 704x480.
- Tracked libraries 74 to 36 MB; decoded sweep 21.9 to 2.5 GB; dev-box slew 87.3 to 23.0 ms.
- Served on three hosts against the 10 fps goal: dataserver3 33.9 fps, voltron 30.0, gateway 27.7.
- Found and fixed a cache-sizing bug: `plan_template_cache` assumed 3 bytes/pixel, PIL stores RGB 4-byte-aligned (~4.22 B); the pre-crop cache was ~20.4 GiB, not the documented 14.4.
- `--mono` now defaults off, since the simulator is a colour instrument. Seven commits, `fb23293..ce853f3`.

## 2026-08-13

- TITAN V renders the flagship scene 4% faster than the dev box (74.55 s/frame mean vs 77.9); the FP64-advantage hypothesis is refuted, since the card is GPU-bound at 100% utilisation.
- CPU serve on voltron: 3.77 fps cold, 14.87 fps warm with `--template-cache auto` (14.4 GiB); the cache flag is required to hit the 10 fps deployment goal there.
- GPU path failed to import at all on the beamline (torch 2.0.1 does not bind `torch._dynamo` until imported); fixed in a new `renderer/torch_compat.py` (`ensure_dynamo()`), kept out of `engine_torch.py` so the fix doesn't mark `render_sha` stale. `51e3334`.
- The droplet scene fits a 12 GB card (preflight 3.30 GB vs 8.80 GB budget); voltron's eight TITAN Vs are held by a training job (~8 GB/card), so the preflight refuses to build there.
- `bench_serve.py` added to benchmark the serve path independent of the GPU. `d4c9f3d`.

## 2026-08-12

- Pin glint no longer guessed the pin's location from the 2D image; `renderer/pin_projection.py` now projects the pin's cylinder from the scene through the pose. `_wide_opaque`, `_fit_one_orientation`, `_pin_axis` and four `STREAK` knobs deleted. `fc84a94`.
- Before the fix, the specular streak rendered entirely on the droplet, not the pin, at zoom 2.5x and above at every angle.
- `hampton_300um_realistic` reached full optical fidelity with a Rayleigh-matched droplet (50,976 faces, 3.28 um facets against the 3.35 um limit) at `--supersample 4`; zoom range goes 1x to 4x. `f57cae5`.
- Build: 360 frames, 7.48 h at 74.8 s/frame, 27.6 MB, peak 7.3 GB.
- A bare launch is now safe on this scene, matching `build_params`' supersample-4 default; `mitegen_200um` still needs `--supersample 1`.

## 2026-08-11

- VRAM enforcement landed: `_mesh_survivor_chunk` had demanded 16.7 GB regardless of the 2 GiB cap beside it (droplet-mesh render goes from never completing to 19 s); `memory_budget()` now derives from free VRAM, with `install_vram_ceiling()`/`check_render_fits()` enforcing it as a hard limit before a build commits. `cfb18bf`, `b906f54`.
- Mesh path gained an AABB cull, the one class that had never had one: `hampton_300um_realistic` 80.6 to 1.86 s/frame (8.06 h to 11.2 min build), `mitegen_200um` 17.0 to 3.00 s. `bef28eb`.
- NA resolves to 0.28 (camera-space crystal/background 0.677 against the photograph's 0.691); not switched, since it would rebuild all three libraries.
- Camera model fixes from driving the viewer: glint tilt corrected (was ±6.4° with phi), grain no longer reads as parallax, background is six octaves of fBm; `hampton_300um_realistic` rebuilt for slice 3. `1e35cf9`, `bb0d36d`. DECISIONS §2026-08-11.

## 2026-08-10

- Renders compared against real photographs for the first time; four of six named fidelity gaps closed.
- Camera emulation added: tone response 1.5% to 100% intermediate, output range 0-255 to 32-181, mean 239.6 to 144.5 against a reference photo's ~150. `4b2a2a4`.
- `render_sha` added, fingerprinting the renderer into the manifest to close the last staleness hole. `557c418`.
- Three plan assumptions were caught wrong by measuring before building: 640 vs 704 px is not a 10% error (pixel aspect cancels it), the streak spec's "peak" figure was a three-pixel frame maximum, and the drop-volume dissent was void on both sides. DECISIONS §2026-08-10.

## 2026-08-08

- `hampton_300um_realistic` frame library shipped: 360 frames, 1396x644 (supersample 1), 2.9 MB, zoom 0.80-1x. `5390132`.
- WSL2 trap: the library CLI's "auto" tile-size ramp spills to host RAM (~45 min/frame); mesh-scene builds there need an explicit `--tile-size` (used 6800, giving 95.2 s/frame, 9.5 h total).
- A second trap: a bare `--scene` launch graded the fresh library stale and deleted its manifest, starting a days-long rebuild; the manifest was reconstructed deterministically. Launch this scene with `--supersample 1`.

## 2026-08-07

- Black-droplet bug root-caused (solvent `color` was secretly an absorption spectrum; a silently-substituted hemisphere refracted past the NA gate) and fixed with a closed-form biconvex spherical-cap lens (`crystal_harvester/droplet.py`), replacing a Bashforth-Adams ODE solver; the droplet is now pinned in the loop aperture. `6f36c2c`.
- `render_torch`'s default ray tile OOM'd on any mesh scene (19.8 GB at 640x480); tile size now calculated from mesh size and free VRAM. `b0b89bf`.
- Stem-to-pin gap fixed in both the generator and the bundled `hampton_300um.yaml`, `hampton_300um` library rebuilt (360 frames, 29.3 MB, `96698ce`/`e73cced`); `mitegen_200um` also rebuilt at 16 MB, smaller than the 28 MB JPEG library it replaced, fixing a grading bug where per-scene supersample was checked against one global default (`3cceb47`).
- All root scripts fixed to import this repo instead of the legacy `/home/jamesh` tree; `investigation/` moved out of the repo, since it had been tracked in git despite a claimed rsync exclude. `ecaf4e1`, `1e90f39`. DECISIONS §2026-08-07.

## 2026-08-06

- Runtime scene switching added (`GET /scenes`, `GET /scene`, `POST /scene?path=&build=`, tab strip); fixed a two-thread deadlock (`_servable` no longer acquires a lock, `_scene_lock` hoisted outside `_gonio_lock`). `13d42b7`.
- Trapezoidal motion profile (ramps to full speed over 0.15 s, speed carried across a preempt) replaced instantaneous jog motion; stage rates halved to match life-size. `471ce20`. Fixed a pre-existing race where a preempted animation could write its pose after a newer command had landed. `ada9e41`.
- Objective PSF added (Gaussian approximation to the Airy disk); surfaced a pre-existing numpy/torch float divergence, now bounded to ±1 grey level. Templates switched from JPEG to lossless PNG (28.7 vs 84.9 MB for the hampton sweep). `87935fc`, `8639e08`.
- Camera driven interactively: `--fps-limit` default raised 5.0 to 30.0 (the MJPEG clamp had capped it, not rendering); translate pad fixed to resolve through `Rᵀ`; out-of-range pose commands now clamped via a new `servable_pose()`.
- `mitegen_200um` served deliberately stale rather than rebuilt; `library_status` now splits current/stale/missing. DECISIONS §2026-08-06.

## 2026-08-03

- `mitegen_200um` library built: 360 frames, 26.8 MB, 103 min at 17.3 s/frame; both scenes now serve without a GPU (19.8 ms/frame, 50 fps).
- Supersample is per-scene, set by each camera's Nyquist limit against the objective's NA (hampton 4x, mitegen 1x), not by preference.
- Fixed: the server rebuilt the library on every default launch, since `--jpeg-quality` (what is served) was forwarded as the library's `quality` (what is stored); split into a separate `--template-quality`.
- `plan_tile_size` rewritten: extrapolating from two small probes under-predicted the mesh path by ~2.7 GB and spilled; now probes reserved memory with a doubling ramp that stops before exceeding budget.
- `manifest.json` now written atomically and fsynced, so a crash mid-build can no longer orphan the library.

## 2026-07-31

- Template pipeline wired end to end: `loop_sim/library/` had existed with zero callers, so every served frame was still a live raytrace; fixed by wiring `TemplateSource`/`ensure_library` into `camera_server`.
- Fixed: template `pixel_size` wasn't scaled with the render margin (960x720 was pure extra field of view); `crop_window` had an inverted sign; `is_current` ignored every build parameter.
- `--supersample` default set to 4, the zoom ceiling set by the NA 0.10 Rayleigh limit; removed a `max(tile_size, W*H)` clamp so peak memory dropped from ~14 GiB (spilled on a 16 GB card) to 4.7 GB at 14.34 Mpx.
- A follow-up review found and fixed further defects: the clamp squeezed the two axes independently (every zoom below ~0.75 served stretched), the crop box rounded to integers (a 0.375 px offset at 4x), and the VRAM-spill warning was baselined against frame 0, so it never fired. Shipped `hampton_300um` library then rebuilt at `--supersample 4`: 360 frames, 84.9 MB, 43 min at 7.2 s/frame, 4.7 GB peak VRAM.

## 2026-07-28

- First scene-fidelity audit: the imaging chain validated (a 700.0 um pin measures 703.0 um at four columns) but the scenes did not (the benchmark scene had a zero-radius droplet, no crystal, and a 69x200 um loop despite its "300um" name; droplets rendered opaque via the NA collection gate rather than absorption).
- Corrected risk A: the documented `tile_size=32768` escape hatch was defeated by a `max(tile_size, W*H)` clamp, so any droplet-bearing scene was unrenderable at full resolution; the `TSurfaceMesh` AABB cull became blocking rather than cosmetic.
- Measured lateral translation as an exact image shift (max pixel difference 0.000000, panning is a crop), which made a pre-computed rotation sweep the chosen architecture over chasing live frame rate.
- Added `loop_sim/library/` plus the tracked `frame_library/` output and a `.gitignore` re-include. `0cf251c`, `b7fc6da`.

## 2026-07-23

- `contacts:` roster removed from HANDOFF front-matter.
- Followed a knowledge-transfer protocol change: a standing roster ages into mis-attribution; people are now named inline where they own a specific artifact. `96356be`.

## 2026-07-17

- TITAN V acceptance measured with a new self-contained harness, `acceptance_voltron.py` (fps, VRAM and compile check). `df33635`.
- Result: compiled preview 11.9 fps median / 10.1 fps p90 (GO), eager 6.3 fps; the mesh scene fits torch 2.6 at ~11.1 GB (knife's edge), OOMs on torch 2.0.1.
- The beamline's default stack (torch 2.0.1, gcc 4.8.5) cannot compile and silently falls back to the 6.3 fps eager path; the 10 fps goal needs torch 2.6 plus devtoolset-7.
- Resolved voltron's CPU spec: 2x Xeon E5-2650 v4.

## 2026-07-16

- RUNBOOK.md written (env setup, run, verify, voltron/SLURM deploy, rollback), completing the handoff doc set alongside HANDOFF.md and DECISIONS.md. `c34b3cf`.
- Verify re-run on a clean checkout (RTX 4080 SUPER).

## 2026-07-15

- HANDOFF.md and DECISIONS.md scaffolded: the first knowledge-transfer flush of the loop-sim memories (bug-hunt root cause, the 10 fps architecture, rejected approaches, TITAN V risks).
- None of this existed in the repo before, living only in the contractor's agent memory and the push-excluded `investigation/` directory.

## 2026-07-06

- 10 fps interactive goal met on an RTX 4080 SUPER: compiled preview path, a single-flight render owner, and a sync-starved trace loop.
- fp32 preview tried and rejected in favor of the existing float64 path.

## 2026-06-25

- float64 GPU correctness fix: the "hairy fiber" artifact gone, GPU output byte-identical to CPU. `56952e4`.
- Added a GPU-resident torch render engine and made the camera server threaded; branch created off `master`. `ea2be75`, `c4f56c9`.
