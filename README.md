# loop-sim

Bright-field microscope simulator for protein crystals mounted in nylon cryo-loops.
Useful for beamline alignment algorithm development, AI training data generation,
and dose estimation.

**Physics model:** Snell's law refraction at every material interface + Beer-Lambert
absorption + Köhler condenser illumination (soft NA edges via multi-ray sampling).
X-ray beam volumes are computed separately by grid ray-casting.

**Output:** JPEG images or an AXIS-compatible MJPEG HTTP stream that looks like a
real beamline camera.

---

## Quick start (pre-built scene)

```bash
# Render the default scene at rest position
python3 render.py scene.yaml --n-cond 7

# Rotate the loop 45° around the phi axis and render
python3 render.py scene.yaml --roty 45 --n-cond 7

# Translate the loop so the crystal is off-center
python3 render.py scene.yaml --tx 0.05 --ty -0.02 --n-cond 7
```

`--n-cond 7` uses one centre + six-point hex ring of condenser rays per pixel,
giving smooth edge transitions.  Use `--n-cond 1` for a fast binary-NA preview.

Output is written to `scene.jpg` (or `--output myfile.jpg`).

---

## Full pipeline from a real loop image

### 1. Digitize the fiber

```bash
python3 digitize_fiber.py real_loop.jpg \
    --pixel-size 0.8233 \       # µm per pixel for your camera
    --diameter   0.020  \       # fiber diameter in mm
    --output hoop.yaml
```

An interactive window opens.  Click to place waypoints around the loop fiber,
starting at the crossover (stem attachment point).  Press **Enter** when done.

### 2. Add the twisted-pair stem

```bash
python3 add_stem.py hoop.yaml \
    --stem-length 0.7 \         # mm of stem visible in the field of view
    --output loop.yaml
```

Outputs three tube objects: `hoop`, `stem_1`, `stem_2`.

### 3. Add a solvent droplet

```bash
python3 add_droplet.py hoop.yaml \
    --volume 0.001 \            # mm³ (≈ 1 nL)
    --output droplet.yaml
```

Produces a biconvex lens mesh whose rim follows the smooth fiber path, in
contact with the hoop fiber all the way around the loop.

### 4. Add a crystal (optional)

```bash
python3 add_crystal.py hoop.yaml \
    --preset plate \            # cube | plate | needle | hexagonal
    --dim 0.04 0.01 \           # half-widths in mm (plate: ab_half c_half)
    --output crystal.yaml
```

Crystal orientation defaults to identity; supply `--a-axis`, `--b-axis`,
`--c-axis` (in Å, XDS convention) to rotate the crystal habit to match
your data collection geometry.

### 5. Assemble the scene

```bash
python3 generate_scene.py loop.yaml crystal.yaml droplet.yaml \
    --template template.yaml \
    --output scene.yaml
```

Object order sets scene priority (first = highest).  Crystal before droplet
is required so the crystal is not masked by the solvent mesh.

### 6. Render

```bash
python3 render.py scene.yaml --n-cond 7
```

---

## Live MJPEG server

The server exposes an AXIS-compatible HTTP interface so it can replace a real
beamline camera in any software that speaks AXIS (MxCuBE, EPICS areaDetector,
browser, VLC, etc.).

```python
from loop_sim.scene.scene       import load
from loop_sim.server.camera_server import CameraServer

scene  = load("scene.yaml")
server = CameraServer(scene, host="0.0.0.0", port=8080, n_cond=7)
server.start()   # blocks; Ctrl-C to stop
```

### HTTP endpoints

| Endpoint | Description |
|---|---|
| `GET /axis-cgi/mjpg/video.cgi` | MJPEG stream |
| `GET /axis-cgi/jpg/image.cgi` | Single JPEG snapshot |
| `GET /motor?tx=0.05&roty=45` | Move motors, returns JSON state |
| `GET /beam` | X-ray illuminated volumes (JSON) |

**Motor parameters:** `tx`, `ty`, `tz` (mm), `rotx`, `roty`, `rotz` (degrees),
`zoom` (dimensionless; `zoom=2` halves pixel size).

**Beam response example:**
```json
{
  "crystal": {"volume_mm3": 0.00042, "weighted_volume": 0.00038},
  "solvent":  {"volume_mm3": 0.00180, "weighted_volume": 0.00165},
  "nylon":    {"volume_mm3": 0.00008, "weighted_volume": 0.00007}
}
```

---

## Scene template

`template.yaml` controls camera geometry, NA, pixel size, beam profile, and
material optical/X-ray properties.  Key fields:

```yaml
camera:
  width:        704
  height:       480
  pixel_size:   0.0008233   # mm at zoom=1
  na_objective: 0.28
  na_condenser: 0.17

materials:
  nylon:   {n: 1.53, mu_optical: 0.1,  mu_xray: 0.1,  color: [0.9, 0.8, 0.6]}
  solvent: {n: 1.333,mu_optical: 0.0,  mu_xray: 0.03, color: [0.97, 0.98, 1.0]}
  crystal: {n: 1.52, mu_optical: 0.02, mu_xray: 2.1,  color: [1.0, 0.0, 0.0]}
```

`color` values are per-channel (R, G, B) Beer-Lambert scale factors; set the
crystal colour to distinguish it visually in the rendered image.

---

## Python interpreter

This project requires:
```
/programs/pytorch/bin/python3
```
