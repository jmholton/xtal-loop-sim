# loop-sim

Bright-field microscope simulator for protein crystals mounted in nylon cryo-loops.
Useful for beamline alignment algorithm development, AI training data generation,
and dose estimation.

Maintainers: project status, open issues, and design rationale are in
[`docs/HANDOFF.md`](docs/HANDOFF.md).

**Output:** JPEG images or an AXIS-compatible MJPEG HTTP stream that looks like a
real beamline camera.

---

## Contents

- [The physics](#the-physics)
  - [What is modelled](#what-is-modelled)
  - [What is not modelled](#what-is-not-modelled)
- [Quick start (pre-built scene)](#quick-start-pre-built-scene)
- [Full pipeline from a real loop image](#full-pipeline-from-a-real-loop-image)
- [Live MJPEG server](#live-mjpeg-server)
  - [Pre-computed templates (the default)](#pre-computed-templates-the-default)
  - [Interactive control page](#interactive-control-page)
  - [HTTP endpoints](#http-endpoints)
- [Scene template](#scene-template)
- [Python interpreter](#python-interpreter)

Every knob — CLI flags, scene keys, environment — is tabulated in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md#every-lever).

---

## The physics

### What is modelled

**Optical (bright-field) path.** Rays are launched from the condenser and traced
through the scene:

- **Snell refraction at every material interface.** Each object carries a refractive
  index; the ray bends at every crossing, and total internal reflection is handled.
  The intersection quadratic runs in float64 — float32 catastrophically cancels here
  and produces a "hairy fiber" artefact (`docs/DECISIONS.md`).
- **Beer-Lambert absorption.** Intensity decays as `exp(-mu * path_length)` through
  each material, with `mu_optical` per material.
- **Köhler illumination**, sampled at `n_cond` condenser angles (1 = a hard NA step,
  7 = 1 centre + a 6-point hex ring giving soft edges). More angles cost time
  linearly and buy little past 7.
- **Numerical-aperture collection gate.** A ray reaching the objective is collected
  only if its angle falls inside `na_objective`; the condenser's `na_condenser` sets
  the illumination cone.
- **Diffraction blur (point-spread function).** The traced image is convolved with a
  Gaussian approximating the objective's Airy PSF, `sigma = 0.21 * lambda / NA` at
  `lambda = 550 nm` — 1.155 um for NA 0.10. Ray tracing alone is geometric optics and
  produces edges sharper than any real objective can form; without this the picture
  is visibly blocky as soon as you magnify. See `loop_sim/renderer/optics.py`.

**X-ray path** (`/beam`, `/xray`) is a separate modality, not the same rays: the beam
is grid-cast through the scene as straight lines, and per-material path lengths give
illuminated volume, absorbed dose and a transmission radiograph via Beer-Lambert. The
objective PSF does **not** apply to it — there is no objective.

**Template replay.** When serving from a pre-computed sweep, depth translation is
approximated as a Gaussian defocus whose width grows with distance from the focal
plane. Everything else (rotation, translation, zoom) is exact: the camera is
orthographic, so those are image-space transforms of the rendered master.

### What is not modelled

Worth knowing before treating a rendered image as ground truth:

- **Partial coherence.** `na_condenser / na_objective = 0.70` (< 1) means the imaging
  is partially coherent, so real edges overshoot and ring slightly. The Gaussian PSF
  is the incoherent approximation and will not reproduce that.
- **Interference and phase contrast** — amplitude only; no propagation of phase.
- **Polarisation** — rays carry no polarisation state.
- **Chromatic effects** — a single 550 nm wavelength, so no dispersion and no colour
  fringing.
- **Scattering** — absorption and refraction only; no diffuse or Mie scattering.

Rationale and the measurements behind each choice are in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Quick start (pre-built scene)

```bash
# Render the default scene at rest position
python3 render.py scene.yaml --n-cond 7

# Rotate the sample 45° about the goniometer spindle and render
python3 render.py scene.yaml --rotx 45 --n-cond 7

# Translate the loop so the crystal is off-center
python3 render.py scene.yaml --tx 0.05 --ty -0.02 --n-cond 7
```

`--n-cond 7` uses one centre + six-point hex ring of condenser rays per pixel,
giving smooth edge transitions.  Use `--n-cond 1` for a fast binary-NA preview.
Add `--device cuda` to render on a CUDA GPU (the CPU path is the default
reference; both produce the same image).

Which motor is the spindle (φ) depends on the scene — it is whichever of
`rotx`/`roty`/`rotz` matches the scene's rotation-axis config and the pin's
mounting direction.  For the bundled `scene_files/` scenes the spindle is
`rotx`; `roty`/`rotz` tilt the sample out of that plane.

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

```bash
python -m loop_sim.server.camera_server --scene scene.yaml --port 8080
```

Useful flags: `--templates {on,off}` (on = serve from a pre-computed frame
library, building it first if absent; off = raytrace every frame live),
`--supersample` (template sampling factor when a build is needed),
`--n-cond` (condenser rays for settled frames, default 7),
`--fps-limit` (MJPEG stream cap, default 30), `--engine {auto,torch,numpy}`,
`--preview-mode {on,off}` (on = fast approximate frames while a move animates,
refining to the exact frame on settle; off = every frame exact full quality),
`--compile-preview {on,off}` (on = preview frames render through a
`torch.compile`d trace for ~10+ fps motion; compiles once at startup, ~20 s),
and `--settle-delay` (seconds of pose quiet after an instant `/motor` set
before the exact full-quality frame renders, default 0.5).  Instant `/motor`
sets — how AXIS-style consumers such as MxCuBE/EPICS drive the goniometer —
count as motion too: streams of `/motor` updates get fast preview frames, and
one exact frame renders automatically once the pose settles.

Or from Python:

```python
from loop_sim.scene.scene       import load
from loop_sim.server.camera_server import CameraServer

scene  = load("scene.yaml")
server = CameraServer(scene, host="0.0.0.0", port=8080, n_cond=7)
server.start()   # blocks; Ctrl-C to stop
```

When a CUDA GPU is present the server renders through the GPU-resident engine
automatically (`engine="auto"`; pass `engine="numpy"` to force the CPU reference
renderer).  Settled frames are byte-identical to the CPU reference; during
animated moves the server streams fast preview frames (n_cond=1, optionally
`torch.compile`d) and renders one exact full-quality frame when the move
settles.  Frame production is single-flight: one background thread renders,
MJPEG clients consume the newest frame (idle streams re-send the last frame
about once a second as a keepalive).

### Pre-computed templates (the default)

The camera is orthographic, so the spindle is the only motor that genuinely
changes image content — everything else is an image-space transform.  The server
therefore renders one 360° sweep per scene, stores it in `frame_library/`, and
serves every frame by cropping, scaling and blurring a template.  On startup it
checks for a current library and builds one if it is missing or stale, so the
first launch for a new scene is slow and every launch after it is instant.

Frames are served in **single-digit milliseconds** and **no GPU is needed at
runtime** — a GPU only accelerates building the library.  Pass `--templates off`
to raytrace every frame live instead.  See `docs/RUNBOOK.md` "Frame libraries"
for the build flags.

### Interactive control page

Open **`http://<host>:<port>/`** in a browser for a live control panel: the
MJPEG view with a centre crosshair, a jog pad, zoom buttons, a speed dial,
**click-in-image-to-recentre**, and a **goniometer target** panel — type X, Y, Z
(mm) and φ (degrees), press **GO**, and the stage slews there.  Moves are
**animated** — the sample interpolates linearly to the target instead of
teleporting (≈4 s to cross the screen, 30 rpm for the spindle, scaled by the
speed dial).  Motion follows a trapezoidal velocity profile — it accelerates
to full speed over ~0.15 s, holds, and decelerates into the target — so it
moves like a real stage rather than snapping between poses.

### HTTP endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Interactive control page (HTML) |
| `GET /axis-cgi/mjpg/video.cgi` | MJPEG stream |
| `GET /axis-cgi/jpg/image.cgi` | Single JPEG snapshot |
| `GET /motor?tx=0.05&roty=45` | Set motors instantly, returns JSON state |
| `GET /move?drotx=90&speed=2` | **Animated** move; returns target JSON state |
| `GET /recenter?px=400&py=300` | Animated move bringing a pixel to the centre |
| `GET /beam` | X-ray illuminated volumes + Beer-Lambert attenuation (JSON) |
| `GET /xray` | X-ray transmission map / radiograph (grayscale PNG) |

**Motor parameters:** `tx`, `ty`, `tz` (mm), `rotx`, `roty`, `rotz` (degrees),
`zoom` (dimensionless; `zoom=2` halves pixel size).

**`/move` parameters:** any absolute motor key, relative deltas (`dtx`, `drotx`,
`dzoom`, …), screen-fraction pan (`panx`, `pany`; ±1 = one field of view), and
`speed` (`>1` faster, `<1` slow-motion).  Unlike `/motor`, `/move` animates the
transition; `/motor` stays instant for AXIS back-compatibility.

**Beam response example:**
```json
{
  "crystal": {"volume_mm3": 0.00042, "weighted_volume": 0.00038,
              "absorbed_dose": 0.41, "transmitted_frac": 0.33},
  "solvent": {"volume_mm3": 0.00180, "weighted_volume": 0.00165,
              "absorbed_dose": 0.05, "transmitted_frac": 0.95},
  "nylon":   {"volume_mm3": 0.00008, "weighted_volume": 0.00007,
              "absorbed_dose": 0.01, "transmitted_frac": 0.99},
  "beam_transmission": 0.27
}
```

`absorbed_dose` is the incident-weighted X-ray flux absorbed in each material
(shadowing-aware: a material behind an absorber sees already-attenuated flux),
`transmitted_frac` is the mean fraction of flux that survives passing through
it, and the top-level `beam_transmission` is the fraction of the whole incident
beam that exits the sample.  By energy book-keeping, Σ `absorbed_dose` +
`beam_transmission` ≈ 1.  These are relative units — `mu_xray` values are
illustrative, not absolute dosimetry.

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
/programs/pytorch/envs/pt/bin/python
```
