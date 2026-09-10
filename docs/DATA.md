# DATA: loop-sim (xtal-loop-sim)

This project has no training data and no model artifacts. What follows is the scene and
calibration input it renders from, and the performance/correctness baselines it is judged
against. Paths are relative to the repo root.

## Artifact inventory

| Artifact | Tracked | Location | Rebuild command |
|---|---|---|---|
| Camera calibration (`template.yaml`) | yes | repo root | not rebuildable; not authoritative, no shipped scene reads it (see Known gaps) |
| Bundled scenes (`scene_files/hampton_300um.yaml`, `scene_files/mitegen_200um.yaml`) | yes | `scene_files/` | hand-built, frozen; no rebuild command |
| Generated fidelity scene (`scene_files/hampton_300um_realistic.yaml`) | yes | `scene_files/` | `python -m crystal_harvester.cli --loop-type hampton --loop-size 300 --crystal hexagonal -o scene_files/hampton_300um_realistic.yaml` |
| Pipeline part-files (`hoop.yaml`, `crystal.yaml`, `droplet.yaml`) | yes | repo root | `digitize_fiber.py` + `add_*.py` from a loop photo (needs manual waypoints) |
| Assembled scene (`scene.yaml`, `loop.yaml`) | no | not shipped | `generate_scene.py loop.yaml crystal.yaml droplet.yaml --template template.yaml --output scene.yaml` |
| Rendered images (`*.png`, `*.jpg`) | no | not shipped | `render.py <scene>.yaml …` |
| Frame libraries (`frame_library/<scene>/*.png` + `manifest.json`) | yes | in-repo, ~37 MB (`hampton_300um` 15 MB, `hampton_300um_realistic` 14 MB, `mitegen_200um` 9 MB) | `python -m loop_sim.library --all` |
| Reference photographs (`real_images/**/*.jpg` + `MANIFEST.tsv`) | yes | in-repo, ~6 MB, 44 jpg | not regenerable from this repo, see `real_images/README.md` |
| X-ray radiograph library (`xray_library/<scene>/*.png` + `manifest.json`) | yes | in-repo, ~39 MB (`hampton_300um` 12 MB, `hampton_300um_realistic` 10 MB, `mitegen_200um` 18 MB) | `python -m loop_sim.library --modality xray --scene <scene>.yaml` |
| Benchmark baselines (`bench_results/`) | no | not shipped | `bench_frame.py` (`--compiled`, `--fp32`, or `--modality xray`) |

## External dependencies & succession

**None.** Nothing this project needs lives in another person's homedir or on external
infrastructure; the authoritative inputs are all tracked in this repo. The only external
dependency is the runtime itself: a torch+CUDA interpreter, unpinned, see
`docs/HANDOFF.md` Open items (torch 2.6 as a hard requirement).

## Known gaps

- **Frame libraries.** ~37 MB tracked: `hampton_300um` 15 MB, `hampton_300um_realistic` 14
  MB (both `--supersample 4`), `mitegen_200um` 9 MB (`--supersample 1`); 360 PNG frames
  each, all `current`. Each rebuild writes a fresh copy into git history, roughly 15 MB per
  library, so rebuilding is not free. `git checkout -- frame_library/<scene>/` undoes an
  accidental rebuild.
- **Benchmark baselines don't travel.** `bench_results/` is gitignored, so the numbers a
  perf claim rests on exist only on the machine that produced them. Comparing this
  machine's results against the beamline's TITAN V (voltron) requires committing a
  baseline set, which needs a `.gitignore` exception.
- **No golden reference image; parity gates are architecture-blind.** The parity tests
  compare GPU against a CPU reference computed on the same machine, so a different
  architecture, such as the beamline's Volta TITAN V (RUNBOOK "Deploy on the TITAN V"),
  could pass every test while producing wrong images. A numpy-anchored golden image,
  committed and compared against, would close this; it needs the kind of `.gitignore`
  exception `frame_library/**/*.png` already has. A cheaper partial answer needs no
  committed image at all: a feature of known physical size must span `size / pixel_size`
  pixels, and the pin measures 703.0 µm against a ground truth of 700.0 µm, an
  architecture-independent check the parity gates can't do. See `scene_dimcheck.py`
  (`/home/jadoughty/projects/loop_sim_MINE/investigation/2026-07_scene_and_perf_harnesses/`)
  and `docs/HANDOFF.md` Open items.
- **Scene calibration: pixels settled, NA still open.** Pixels: the Hampton scenes' square
  640×7.4 µm rendition matches the real camera to under 1% (`docs/DECISIONS.md`
  §2026-08-10). NA: measured at 0.28 against `real_images/D01` (`docs/DECISIONS.md`
  §2026-08-11 NA fork), but the shipped scenes still carry NA 0.10/0.07; switching rebuilds
  all three frame libraries and is an owner's decision.
- **`hampton_300um` is a bare performance scene.** Its solvent sphere has zero radius (no
  droplet), and its loop waypoints span 69×200 µm despite the `300um` name. See
  `docs/HANDOFF.md` Open items.
- **The gateway mirror carries gitignored scratch.** `TEST_*.png` / `out_*.png` at the repo
  root are untracked scratch renders that happen to sit in the working tree, so the file
  mirror includes them while a `git clone` will not. Regenerate instead of treating them as
  references.
