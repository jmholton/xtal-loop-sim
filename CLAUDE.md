# CLAUDE.md: loop-sim

Bright-field microscope simulator for protein crystals mounted in nylon cryo-loops.

## Interpreter and verify

`.venv/bin/python`, built by `bash setup_venv.bash` (same recipe on a dev box, voltron,
and dataserver3). Never the system `python3` or a conda python.
Verify: `.venv/bin/python -m pytest tests/ -q` from the repo root. GPU tests skip on
CPU-only boxes. On a 17 GB WSL2 box the full-suite invocation can exhaust RAM; run one
file at a time instead (`for f in tests/test_*.py; do .venv/bin/python -m pytest $f -q; done`),
see docs/RUNBOOK.md "Verify".

## Invariants

- The intersection quadratic stays float64 on every path, CPU and GPU.
- Never `torch.compile(mode="reduce-overhead")` in the threaded server.
- Crystal must come before droplet in a scene's object list.
- The stem is two tube objects, never one.
- Material `color` is an absorption spectrum, not just display RGB.
- `n_cond` above 7 buys nothing.
- Never set `--time` in `tools/run_gpu.slurm`.
- `loop_sim/library/frame_library.py`'s `_RENDER_SOURCES` hashes `renderer/microscope.py`,
  `engine_torch.py`, `optics.py`, `motors/goniometer.py`, and `scene/*.py`: that list is
  still the one whose edits change template pixels. Editing one of them no longer
  rebuilds anything: it makes `tests/test_render_sha_frozen.py` red, and
  `python -m loop_sim.library --verify --scene <s>` is how you find out whether pixels
  actually moved. Nothing rebuilds a library except `--force`; the server never builds.

## Where things are

- README.md: user guide, quick start, pipeline, server, endpoints.
- docs/HANDOFF.md: start here. State, next actions, hazards, repo map.
- docs/RUNBOOK.md: commands and expected output.
- docs/DECISIONS.md: rationale and dead ends. Add an entry; never correct one in place.
- docs/DATA.md: artifact tables (paths, sizes, provenance).
- docs/WORK_LOG.md: dated history, newest first.
- data/: scene_files/, frame_library/, real_images/.
- tools/: benchmarks, profilers, the SLURM job.
- xtalLoopSimDHS/README.md: the DCSS hardware server, its own venv.

## Docs rules

No em-dashes. When you replace a fact, log the reasoning in docs/DECISIONS.md, not here.
docs/HANDOFF.md live sections stay under 2,500 words.
