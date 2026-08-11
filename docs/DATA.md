# DATA — loop-sim (xtal-loop-sim)

<!-- loop-sim is a generator, not a learner: it has no training set and no model weights.
     It does have data in the sense that matters here — the scene and camera/material
     definitions it renders from, and the baselines used to judge correctness and speed.
     Paths are relative to the repo root. -->

**This project has no training data and no model artifacts.** What it does have is scene /
calibration input and performance-and-correctness baselines, below.

## Artifact inventory

| Artifact | Class | Location | Owner | How to rebuild |
|---|---|---|---|---|
| Camera + material properties (`template.yaml`) | reference, **not authoritative** | in-repo, tracked | this repo (originally James Holton) | — cannot be rebuilt. **Reclassified 2026-08-10:** it encodes the hi zoom stop's HORIZONTAL pitch as a square pixel, which makes its vertical field of view 9.91% short, and no shipped scene reads it. The real calibration is `sample_camera_constant` in `wash_pin/claude/BL-831.dat` (1.110 non-square pixels at both stops), and the photographs it describes are in `real_images/`. |
| Bundled scenes (`scene_files/hampton_300um.yaml`, `scene_files/mitegen_200um.yaml`) | **authoritative** | in-repo, tracked | this repo | — complete, hand-built scenes; `hampton` is the tube-based scene, `mitegen` the mesh-based one. |
| Pipeline part-files (`hoop.yaml`, `crystal.yaml`, `droplet.yaml`) | **authoritative** | in-repo, tracked | this repo | Regenerable in principle via `digitize_fiber.py` / `add_*.py` from a real loop photo, but `digitize_fiber.py` needs a human clicking waypoints — treat the committed ones as the record. |
| Assembled scene (`scene.yaml`, `loop.yaml`) | regenerable | **not shipped** — gitignored | — | `generate_scene.py loop.yaml crystal.yaml droplet.yaml --template template.yaml --output scene.yaml` (see ../README.md). |
| Rendered images (`*.png`, `*.jpg`) | regenerable | **not shipped** — gitignored | — | `render.py <scene>.yaml …`. Deterministic given scene + pose + code. |
| **Frame libraries** (`frame_library/<scene>/*.png` + `manifest.json`) | **shipped, tracked** | in-repo, tracked (explicit `.gitignore` re-includes for both `*.png` and `*.jpg`, since both are ignored repo-wide) | this repo | `python -m loop_sim.library --all`. Regenerable but **deliberately committed** — a library is what the camera server replays, so it ships with the code rather than being rebuilt on each host. Each manifest stores a SHA-256 of its scene YAML *and* the build parameters (now including `format` and `psf`); changing either rebuilds on first use. **Stored losslessly as PNG since 2026-08-06** — a real AXIS camera applies exactly one JPEG compression, and storing JPEG templates then re-encoding on the wire applied two. PNG is also *smaller* for these near-binary frames: measured 0.10 MB vs 0.25 MB per hampton frame, **28.7 MB vs 84.9 MB** for the rebuilt hampton sweep, at the cost of a dearer decode (~55 vs 36 ms, which bites only on a spindle slew: 14.7 fps vs 21.6 fps through a spin). Templates also carry the objective PSF baked in; `psf_sigma_px` in the manifest records how much. |
| **Reference photographs** (`real_images/**/*.jpg` + `MANIFEST.tsv`) | **shipped, tracked** | in-repo, tracked (explicit `.gitignore` re-include, `*.jpg` is ignored repo-wide) | this repo, sampled 2026-08-10 | **Not regenerable from anything in this repo** — 44 real BL831 sample-camera frames sampled from six trees on the beamline mirror, 5.6 MB. This is the only ground truth the renderer's *appearance* is judged against, and losing it means losing the ability to falsify a fidelity claim. `MANIFEST.tsv` records each file's source path so the set could be re-sampled; `build_refs.py` was session scratch and is not shipped. **Use it to falsify, not to fit** — `real_images/README.md` has the three limits. |
| Benchmark baselines (`bench_results/`) | regenerable | **not shipped** — gitignored | — | `bench_frame.py` (`--compiled`, `--fp32`). See the gap below. |

## External dependencies & succession

**None.** Nothing this project needs lives in another person's homedir or on external
infrastructure — the authoritative inputs are all tracked in this repo. (Contrast the
sibling CV projects, whose training data lives under `/home/jamesh/projects`.) The only
external dependency is the runtime itself: a torch+CUDA interpreter, unpinned — see the
`requirements.txt` note in `docs/HANDOFF.md` Hazards (B).

## Known gaps

- **Supersampled frame libraries are large, and every scene adds another one.** The 1×
  library was 5.3 MB; `hampton_300um` at the `--supersample 4` default is **28.7 MB**
  (360 frames of 5578×2570). It is tracked in git, so each rebuild writes a fresh copy
  into history and every `push-all` moves it. Options if this becomes a problem: drop to
  `--supersample 2` (4× cheaper, still resolves the fiber), track only the reference
  `hampton_300um` library and let the rest build on first use, or stop tracking them and
  accept a slow first launch per scene. The repo currently ships two, both PNG and both current since the
  `mitegen_200um` rebuild on 2026-08-07 — `hampton_300um` (30 MB at `--supersample 4`)
  and `mitegen_200um` (16 MB at `--supersample 1`), **45 MB together**. Note PNG came
  out smaller than the JPEG it replaced in both cases, so the two rebuilds roughly
  halved this figure rather than growing it. A third, `hampton_300um_realistic`
  (2.9 MB at `--supersample 1`), shipped 2026-08-08 and is **awaiting a rebuild since
  2026-08-10** — its scene changed in slice 3.
- **Benchmark baselines don't travel.** `bench_results/` is gitignored, so the numbers a
  perf claim rests on exist only on the machine that produced them. Comparing this
  machine's results against the beamline's TITAN V (voltron) requires committing a
  baseline set — which needs a `.gitignore` exception, since the current rules ignore
  `bench_results/` outright.
- **There is no golden reference image, and the current gates can't catch an
  architecture-specific error.** The parity tests compare GPU against a CPU reference
  computed *on the same machine*, so they are architecture-blind: a Volta box could pass
  every test while producing wrong images. A numpy-anchored golden image, committed and
  compared against, would close this. That also needs a `.gitignore` exception, because
  `*.png` is ignored repo-wide — `frame_library/**/*.png` is now re-included and sets the
  precedent for how to write one. See `docs/DECISIONS.md` §"TITAN V deployment".
  **A cheaper partial answer now exists:** a *dimensional* assertion needs no committed
  image at all. A feature of known physical size must span `size / pixel_size` pixels; the
  pin measures 703.0 µm against a ground truth of 700.0 µm. That check is
  architecture-independent and would catch a class of error the parity gates cannot. See
  `scene_dimcheck.py` (in the analysis tree outside this repo,
  `/home/jadoughty/projects/loop_sim_MINE/investigation/2026-07_scene_and_perf_harnesses/`)
  and `docs/HANDOFF.md` "Scene fidelity".

- **The scene inputs themselves were never validated until 2026-07-28, and two are wrong.**
  `scene_files/hampton_300um.yaml` carries a zero-radius solvent sphere (no droplet) and a
  loop whose waypoints span 69 × 200 µm despite its `300um` name; `template.yaml` is
  described here as the authoritative calibration but no shipped scene uses its pixel size
  or NA. Treat `crystal_harvester` output as the dimensionally-trustworthy source and the
  hand-built bundled scenes as unverified. See `docs/HANDOFF.md` "Scene fidelity".
  **Partly answered 2026-08-10:** the pixel half of the `template.yaml` puzzle is settled
  (it is the hi stop's horizontal pitch used as a square pixel; the Hampton scenes' square
  640 × 7.4 µm is the mid stop rendered correctly on square pixels, to under 1%). NA is
  still open, and now has measured evidence — see `docs/HANDOFF.md` "Open questions".
- **The gateway mirror carries gitignored scratch.** `TEST_*.png` / `out_*.png` at the repo
  root are untracked scratch renders that happen to sit in the working tree, so the file
  mirror includes them while a `git clone` will not. Don't treat them as references —
  regenerate instead.
