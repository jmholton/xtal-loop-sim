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
| Camera + material properties (`template.yaml`) | **authoritative** | in-repo, tracked | this repo (originally James Holton) | — cannot be rebuilt; it encodes camera geometry, NA, pixel size, and per-material optical/X-ray constants. Losing it loses the calibration. |
| Bundled scenes (`scene_files/hampton_300um.yaml`, `scene_files/mitegen_200um.yaml`) | **authoritative** | in-repo, tracked | this repo | — complete, hand-built scenes; `hampton` is the tube-based scene, `mitegen` the mesh-based one. |
| Pipeline part-files (`hoop.yaml`, `crystal.yaml`, `droplet.yaml`) | **authoritative** | in-repo, tracked | this repo | Regenerable in principle via `digitize_fiber.py` / `add_*.py` from a real loop photo, but `digitize_fiber.py` needs a human clicking waypoints — treat the committed ones as the record. |
| Assembled scene (`scene.yaml`, `loop.yaml`) | regenerable | **not shipped** — gitignored | — | `generate_scene.py loop.yaml crystal.yaml droplet.yaml --template template.yaml --output scene.yaml` (see ../README.md). |
| Rendered images (`*.png`, `*.jpg`) | regenerable | **not shipped** — gitignored | — | `render.py <scene>.yaml …`. Deterministic given scene + pose + code. |
| **Frame libraries** (`frame_library/<scene>/*.jpg` + `manifest.json`) | **shipped, tracked** | in-repo, tracked (explicit `.gitignore` re-include, since `*.jpg` is ignored repo-wide) | this repo | `python -m loop_sim.library --all`. Regenerable but **deliberately committed** — a library is the replay source for the camera, so it ships with the code rather than being rebuilt on each host. Each manifest stores a SHA-256 of its scene YAML; a changed scene rebuilds on first use. ~6 MB per scene at the 1° default. |
| Benchmark baselines (`bench_results/`) | regenerable | **not shipped** — gitignored | — | `bench_frame.py` (`--compiled`, `--fp32`). See the gap below. |

## External dependencies & succession

**None.** Nothing this project needs lives in another person's homedir or on external
infrastructure — the authoritative inputs are all tracked in this repo. (Contrast the
sibling CV projects, whose training data lives under `/home/jamesh/projects`.) The only
external dependency is the runtime itself: a torch+CUDA interpreter, unpinned — see the
`requirements.txt` note in `docs/HANDOFF.md` Hazards (B).

## Known gaps

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
  `*.png` is ignored repo-wide — `frame_library/` now sets the precedent for how to write
  one. See `docs/DECISIONS.md` §"TITAN V deployment".
  **A cheaper partial answer now exists:** a *dimensional* assertion needs no committed
  image at all. A feature of known physical size must span `size / pixel_size` pixels; the
  pin measures 703.0 µm against a ground truth of 700.0 µm. That check is
  architecture-independent and would catch a class of error the parity gates cannot. See
  `investigation/scene_dimcheck.py` and `docs/HANDOFF.md` "Scene fidelity".

- **The scene inputs themselves were never validated until 2026-07-28, and two are wrong.**
  `scene_files/hampton_300um.yaml` carries a zero-radius solvent sphere (no droplet) and a
  loop whose waypoints span 69 × 200 µm despite its `300um` name; `template.yaml` is
  described here as the authoritative calibration but no shipped scene uses its pixel size
  or NA. Treat `crystal_harvester` output as the dimensionally-trustworthy source and the
  hand-built bundled scenes as unverified. See `docs/HANDOFF.md` "Scene fidelity".
- **The gateway mirror carries gitignored scratch.** `TEST_*.png` / `out_*.png` at the repo
  root are untracked scratch renders that happen to sit in the working tree, so the file
  mirror includes them while a `git clone` will not. Don't treat them as references —
  regenerate instead.
