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
Run GPU scripts via SSH:

```bash
ssh voltron "cd $PWD ; bash debug_optim.bash"
```

**Never** use `&&` in SSH commands to voltron: its login shell is tcsh, which
does not parse `&&`.  Always write a bash script and invoke it with
`ssh voltron "cd $PWD ; bash script.bash"`.

`--device cuda` enables the GPU path in `render.py` and `scene.load()`.
`--device cpu` is the reference path (float64, no PyTorch required).

Performance: CPU ~230 s/frame, GPU (TITAN V) ~3 s/frame at 704×480, n_cond=1.

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

**Do not** probe material-after-interface using `te ≤ best_t < tx` from the
original ray's intersection values — nearby surfaces within float32 precision
of each other will corrupt the active-object set and produce wrong materials.

**Do** use `_obj_index_at_points_batch(probe_pts)` (fires a fresh +Z probe
ray from the hit point).  The +Z direction is independent of the original ray,
so containment is re-evaluated in a fresh coordinate frame and is robust to
float32 surface proximity.

## Comparison workflow

After any change to the renderer or scene on voltron:

```bash
ssh voltron "cd $PWD ; bash debug_optim.bash"   # renders scene_ref.jpg + scene_optim.jpg
ssh voltron "cd $PWD ; bash check_diff.bash"     # 3-way: CPU vs GPU-original vs GPU-new
```

Acceptable: `pixels>10` count for GPU-new vs CPU-ref should be ≤ GPU-original
vs CPU-ref (~36-40 K pixels for a 704×480 scene with float32 GPU).
