# CLAUDE.md: loop-sim

Bright-field microscope simulator for protein crystals mounted in nylon cryo-loops.

## Interpreter and verify

Beamline: `/programs/pytorch/envs/pt/bin/python`, never the system `python3`.
Dev box: conda env `loopsim`.
Verify: `python -m pytest tests/ -q` from the repo root. GPU tests skip on CPU-only boxes.

## Invariants

- The intersection quadratic stays float64 on every path, CPU and GPU.
- Never `torch.compile(mode="reduce-overhead")` in the threaded server.
- Crystal must come before droplet in a scene's object list.
- The stem is two tube objects, never one.
- Material `color` is an absorption spectrum, not just display RGB.
- `loop_sim/library/frame_library.py`'s `_RENDER_SOURCES` hashes `renderer/microscope.py`,
  `engine_torch.py`, `optics.py`, `motors/goniometer.py`, and `scene/*.py` into `render_sha`.
  An edit to any of them grades the three optical libraries stale, and a bare server
  launch rebuilds them. The X-ray libraries hash their own list (`xray_torch.py`,
  `engine_torch.py`, `beam.py`, `scene/*.py`, `motors/goniometer.py`). Re-stamp the manifests only after proving pixels unchanged.
- `renderer/` is enumerated file by file and `scene/*.py` is globbed. A new module that
  changes template pixels must be added to `_RENDER_SOURCES` by name or its edits never
  grade a library stale; a serve-time module (`field.py`, `pin_projection.py`,
  `torch_compat.py`) must stay out of the hash, which is why those live in `renderer/`.
- Never set `--time` in `run_gpu.slurm`.
- `n_cond` above 7 buys nothing.
- Only `mitegen_200um` needs `--supersample 1` on a bare launch; the other two libraries
  are safe bare.

## Where things are

- README.md: user guide, quick start, pipeline, server, endpoints.
- docs/HANDOFF.md: start here. State, next actions, hazards, repo map.
- docs/RUNBOOK.md: commands and expected output.
- docs/DECISIONS.md: rationale and dead ends. Add an entry; never correct one in place.
- docs/DATA.md: artifact tables (paths, sizes, provenance).
- docs/WORK_LOG.md: dated history, newest first.

## Docs rules

No em-dashes. When you replace a fact, log the reasoning in docs/DECISIONS.md, not here.
docs/HANDOFF.md live sections stay under 2,500 words.
