# CLAUDE.md — loop-sim

Bright-field microscope simulator for protein crystals in cryo-loops.
See README.md for user-facing documentation.

## Python interpreter

Always use `/programs/pytorch/envs/pt/bin/python` — not `python3` or
`/usr/bin/python3`.  The system Python is 3.6, owned by root, and has none of
the required packages.  The PyTorch env has numpy, scipy, PIL, pyyaml, and
PyTorch+CUDA.

## GPU rendering (voltron)

GPU-accelerated rendering requires CUDA — only available on voltron.
Submit via SLURM from the local machine (no SSH needed):

```bash
sbatch run_gpu.slurm          # gpu partition, gres=gpu:1, no --time
squeue --job <jobid>          # check status
```

Do **not** set `--time` in SLURM job scripts — this queue has no time limits
and the flag causes jobs to be cancelled prematurely.

If you need an interactive SSH session (not SLURM): voltron's login shell is
tcsh, which does not parse `&&`.  Always write a bash script and invoke it with
`ssh voltron "cd $PWD ; bash script.bash"`.

`--device cuda` enables the GPU path in `render.py` and `scene.load()`.
`--device cpu` is the reference path (float64, no PyTorch required).

Performance (704×480, TITAN V on voltron):

| n_cond | CPU    | GPU   | speedup |
|--------|--------|-------|---------|
| 1      | ~179 s | ~9 s  | 20×     |
| 7      | ~1230 s| ~23 s | 53×     |

Those are the legacy per-object CUDA path on voltron's TITAN V.  A newer
**GPU-resident engine** (`loop_sim/renderer/engine_torch.py`) runs the whole
trace on-device, is byte-identical to the numpy reference in float64, and is
~6–8× faster again (≈160 ms/frame at 640×480, n_cond=1, on a desktop RTX 4080).
`camera_server` uses it automatically when CUDA is present; `render.py --device
cuda` still uses the legacy per-object path.

## Condenser sampling (n_cond)

`--n-cond 7` (1 centre + 6-point hex ring) gives smooth edge transitions.
`--n-cond 1` is fast preview mode (hard NA step at the objective edge).

Do **not** use n_cond > 7 without a specific reason — it adds render time with
diminishing returns on edge quality.

## Scene assembly order = rendering priority

The object list in `scene.yaml` is priority-ordered: first entry wins at any
point in space.  When assembling scenes with `generate_scene.py`:

```bash
python3 generate_scene.py loop.yaml crystal.yaml droplet.yaml ...
```

- **Crystal must come before droplet.**  A point inside the crystal is also
  inside the droplet.  If droplet is listed first, every crystal voxel is
  assigned `solvent` and the crystal is invisible.

## Stem topology

The twisted-pair stem **must** be two separate tube objects (`stem_1`,
`stem_2`) output by `add_stem.py`.  Do not merge them into one tube.

The crossover points where the two fibers intersect produce characteristic
dark shadows.  A single helical tube models only one fiber: the second is
absent, crossover shadows disappear, and the stem looks wrong.

## Architecture overview

```
render.py                    CLI: loads scene, drives goniometer, calls render()
loop_sim/
  scene/
    scene.py                 YAML loader; Scene.next_interface(); path_lengths();
                             path_segments() (ordered front-to-back, for X-ray attenuation)
    primitives.py            HalfSpace, Sphere, Cylinder, Box, Capsule, Ellipsoid
    tube.py                  Neville-chain tube; CUDA hot path (_intersect_batch_cuda)
    surface_mesh.py          Möller-Trumbore mesh; CUDA hot path (_mt_batch_cuda)
    csg.py                   Intersection / Union / Difference
    materials.py             Material dataclass; AIR, WATER, NYLON constants
  motors/
    goniometer.py            SE(3) from tx/ty/tz/rotx/roty/rotz/zoom
  renderer/
    microscope.py            Snell's law ray tracer; Beer-Lambert; NA cutoff (numpy reference)
    beam.py                  X-ray grid probe → per-material volume + Beer-Lambert
                             attenuation (absorbed_dose, transmitted_frac,
                             beam_transmission); render_xray_numpy (radiograph, CPU ref)
    engine_torch.py          GPU-resident torch engine (TorchScene, render_torch);
                             byte-identical to microscope.py in float64, ~6-8x faster;
                             render_xray_torch (straight-ray transmission map)
  server/
    camera_server.py         AXIS HTTP server; renders via engine_torch on CUDA, else
                             microscope; /beam (JSON) + /xray (radiograph PNG)
```

## Key API: next_interface()

`Scene.next_interface(origins, dirs, t_min=1e-6)` returns:
- `t` — (N,) distance to next interface (inf = exited scene)
- `normals` — (N, 3) outward surface normal at the interface
- `mat_out_oi` — (N,) int: object index of material **after** crossing
  (-1 = background/air; 0..n_obj-1 = `scene.objects[oi]`)

`_trace_rays` in `microscope.py` maintains `cur_mat_oi` (same index
convention) and looks up n, mu, color via pre-built numpy tables
(`mat_n_tab`, `mat_mu_tab`, `mat_col_tab`, index 0 = background,
index k = objects[k-1]).

## GPU intersection precision (float64)

The CUDA intersection path computes the cylinder/triangle quadratics in
**float64** (`tube.py`, `surface_mesh.py`).  It used to use float32, which
catastrophically cancelled in `c_ = baba*oaoa - baoa**2 - r²*baba`: with the ray
launched 50 mm upstream, `oaoa ≈ 2500` swamped the r² signal of a ~7.5 µm fiber,
so float32 produced off-surface hits and wrong normals → the "hairy/spikey"
fiber artifact.  Doing the quadratic in float64 fixes it, and the GPU render is
now **byte-identical** to the float64 CPU reference.  **Do not** down-cast the
intersection inputs or geometry to float32.

### Material-after-interface (mat_out_oi)

`next_interface()` picks the material a ray enters after a crossing with an
interval check on the already-computed `all_te`/`all_tx`:

```python
t_probe = best_t + 1e-4
inside  = (te_m < t_probe) & (t_probe < tx_m)
mat_out_oi = first object whose interval contains t_probe, else -1
```

**Do not** revert to `_obj_index_at_points_batch` probe rays — that was the
original broken path (50% failure from probe-point ambiguity).

## Comparison workflow

After any change to the renderer or scene, submit to SLURM:

```bash
sbatch run_gpu.slurm     # renders CPU+GPU at n_cond=1 and n_cond=7, PNG output
cat slurm_<jobid>.log    # check timing and diff stats
```

Acceptable thresholds (PNG, lossless, 704×480):

| n_cond | pixels>10 | notes |
|--------|-----------|-------|
| 1      | ≤ ~35 K   | baseline; nearly all diff pixels are >60 (TIR/NA flip) |
| 7      | ≤ ~60 K   | higher than n_cond=1 because 7 angles sample more edge cases |

Almost all differing pixels are binary flips (TIR or NA cutoff crossing due to
float32 geometry), not gradual noise.  The n_cond=7 count is ~1.6× the
n_cond=1 count because independent condenser rays can each flip a different set
of edge pixels.
