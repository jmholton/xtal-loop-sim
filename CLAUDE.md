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
    scene.py                 YAML loader; Scene.next_interface(); path_lengths()
    primitives.py            HalfSpace, Sphere, Cylinder, Box, Capsule, Ellipsoid
    tube.py                  Neville-chain tube; CUDA hot path (_intersect_batch_cuda)
    surface_mesh.py          Möller-Trumbore mesh; CUDA hot path (_mt_batch_cuda)
    csg.py                   Intersection / Union / Difference
    materials.py             Material dataclass; AIR, WATER, NYLON constants
  motors/
    goniometer.py            SE(3) from tx/ty/tz/rotx/roty/rotz/zoom
  renderer/
    microscope.py            Snell's law ray tracer; Beer-Lambert; NA cutoff
    beam.py                  X-ray grid probe → {material: volume_mm3}
  server/
    camera_server.py         AXIS-compatible HTTP: MJPEG, snapshot, /motor, /beam
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

## Float32 precision at surfaces (GPU path)

The GPU path uses float32.  At t ≈ 50 mm, float32 ULP ≈ 6 nm.

### Material-after-interface (mat_out_oi)

`next_interface()` determines which material a ray enters after crossing a
surface using an **interval check** on the already-computed `all_te`/`all_tx`
arrays:

```python
t_probe = best_t + 1e-4   # 100 nm >> 6 nm ULP → lands on correct side
inside  = (te_m < t_probe) & (t_probe < tx_m)
mat_out_oi = inside.argmax(axis=1) if inside.any(axis=1) else -1
```

The 100 nm offset is >> float32 ULP at t ≈ 50 mm, so it reliably lands past
the interface in both entry and exit cases.  No extra CUDA round-trips needed.

**Do not** revert to `_obj_index_at_points_batch` probe rays — that approach
was the original broken path (fires a +Z probe from the float32 hit point;
50% failure rate due to ULP ambiguity).

### Tube normals

Tube (`_intersect_batch_cuda`) returns float32 normals (~0.1% error at tube
radius 10 µm).  `next_interface()` recomputes them in float64 via
`Tube.recompute_normals_f64()` using the float64 origins/dirs from the ray
tracer and the float64 `_curve_pts` stored in each Tube.  Residual error:
~0.03% (limited by float32 t precision, 6 nm ULP).

`SurfaceMesh` normals are already float64 (face normals looked up by face
index after CUDA intersection).

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
