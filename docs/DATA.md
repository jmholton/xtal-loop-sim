# DATA: loop-sim (xtal-loop-sim)

This project has no training data and no model artifacts. What follows is the scene and
calibration input it renders from, and the performance/correctness baselines it is judged
against. Paths are relative to the repo root.

## Artifact inventory

| Artifact | Tracked | Location | Rebuild command |
|---|---|---|---|
| Camera calibration (`template.yaml`) | yes | repo root | not rebuildable; not authoritative, no shipped scene reads it (see Known gaps) |
| Bundled scenes (`hampton_300um.yaml`, `mitegen_200um.yaml`) | yes | `data/scene_files/` | hand-built, frozen; no rebuild command |
| Generated fidelity scene (`hampton_300um_realistic.yaml`) | yes | `data/scene_files/` | `python -m crystal_harvester.cli --loop-type hampton --loop-size 300 --crystal hexagonal -o data/scene_files/hampton_300um_realistic.yaml` |
| Pipeline example (`hoop.yaml`) | yes | `data/scene_files/examples/` | not rebuildable; the one pipeline sample kept, produced by `digitize_fiber.py` from a loop photo (needs manual waypoints) |
| Assembled scene (`scene.yaml`, `loop.yaml`) | no | not shipped | `generate_scene.py loop.yaml crystal.yaml droplet.yaml --template template.yaml --output scene.yaml` |
| Rendered images (`*.png`, `*.jpg`) | no | not shipped | `render.py <scene>.yaml …` |
| Frame libraries (`<scene>/*.png` + `manifest.json`) | yes | `data/frame_library/`, ~37 MB (`hampton_300um` 15 MB, `hampton_300um_realistic` 14 MB, `mitegen_200um` 9 MB) | `python -m loop_sim.library --all` (builds only what's missing; `--force` to rebuild) |
| Reference photographs (`**/*.jpg` + `MANIFEST.tsv`) | yes | `data/real_images/`, ~6 MB, 44 jpg | not regenerable from this repo, see `data/real_images/README.md` |
| Benchmark baselines | no | `tools/bench_results/` | `tools/bench_frame.py` (`--compiled`, `--fp32`, or `--modality xray`) |
| Python environment (`.venv/`) | no | repo root | `bash setup_venv.bash`; rebuilt per host from `requirements.txt`, never copied between hosts |
| DHS Python environment (`xtalLoopSimDHS/.venv/`) | no | `xtalLoopSimDHS/` | see `xtalLoopSimDHS/README.md` "Create the env"; rebuilt per host, never copied |

## External dependencies & succession

The authoritative inputs are all tracked in this repo. Two runtime dependencies come from
outside it:

- **torch 2.6.0 (cu124 wheel)**, pinned by `requirements.txt` and installed from the
  public wheel index by `setup_venv.bash`. A compiled-vs-eager fallback warns rather than
  refuses, see `docs/HANDOFF.md` Open items.
- **pydhsfw**, the beamline's DHS framework, needed by `xtalLoopSimDHS/` only. It is not
  on PyPI; the DHS venv installs it from a checkout (on the beamline,
  `/home/classen/pydhsfw`; upstream github.com/tetrahedron-technologies/pydhsfw). See
  `xtalLoopSimDHS/README.md` "Create the env".

## Known gaps

- **Frame libraries.** ~37 MB tracked: `hampton_300um` 15 MB, `hampton_300um_realistic` 14
  MB (both `--supersample 4`), `mitegen_200um` 9 MB (`--supersample 1`); 360 PNG frames each.
  `hampton_300um` and `hampton_300um_realistic` are current; `mitegen_200um` is stale (built
  at supersample 1 against the supersample-4 default) and is served as-is. Each rebuild
  writes a fresh copy into git history, roughly 15 MB per library, so rebuilding is not free.
- **Benchmark baselines don't travel.** `tools/bench_results/` is gitignored, so the
  numbers a perf claim rests on exist only on the machine that produced them. Comparing
  this machine's results against the beamline's TITAN V (voltron) requires committing a
  baseline set, which needs a `.gitignore` exception.
- **No golden reference image; parity gates are architecture-blind.** The parity tests
  compare GPU against a CPU reference computed on the same machine, so a different
  architecture, such as the beamline's Volta TITAN V (RUNBOOK §7a),
  could pass every test while producing wrong images. A numpy-anchored golden image,
  committed and compared against, would close this; it needs the kind of `.gitignore`
  exception `data/frame_library/**/*.png` already has. A cheaper partial answer needs no
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
