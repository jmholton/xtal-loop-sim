# loop-sim

Bright-field microscope simulator for protein crystals mounted in nylon cryo-loops.
Useful for beamline alignment algorithm development, AI training data generation,
and dose estimation.

Maintainers: project status, open issues, and design rationale are in
[`docs/HANDOFF.md`](docs/HANDOFF.md). Setting up an environment from nothing:
[`docs/RUNBOOK.md`](docs/RUNBOOK.md#1-environment).

**Output:** JPEG images or an AXIS-compatible MJPEG HTTP stream that looks like a
real beamline camera.

---

## Contents

- [1. The physics](#1-the-physics)
  - [1a. What is modelled](#1a-what-is-modelled)
  - [1b. What is not modelled](#1b-what-is-not-modelled)
- [2. Quick start (pre-built scene)](#2-quick-start-pre-built-scene)
- [3. Full pipeline from a real loop image](#3-full-pipeline-from-a-real-loop-image)
  - [3a. Digitize the fiber](#3a-digitize-the-fiber)
  - [3b. Add the twisted-pair stem](#3b-add-the-twisted-pair-stem)
  - [3c. Add a solvent droplet](#3c-add-a-solvent-droplet)
  - [3d. Add a crystal (optional)](#3d-add-a-crystal-optional)
  - [3e. Assemble the scene](#3e-assemble-the-scene)
  - [3f. Render](#3f-render)
- [4. Live MJPEG server](#4-live-mjpeg-server)
  - [4a. Pre-computed templates (the default)](#4a-pre-computed-templates-the-default)
  - [4b. Interactive control page](#4b-interactive-control-page)
  - [4c. HTTP endpoints](#4c-http-endpoints)
  - [4d. Switching scenes without restarting](#4d-switching-scenes-without-restarting)
- [5. Driving it from dcss and BluIce](#5-driving-it-from-dcss-and-bluice)
- [6. Scene template](#6-scene-template)
- [7. Environment](#7-environment)

Every camera-server and library-builder flag is tabulated in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md#5-flags); `--help` on any command lists the rest.

---

## 1. The physics

### 1a. What is modelled

**Optical (bright-field) path.** Rays are launched from the condenser and traced
through the scene:

- **Snell refraction at every material interface.** Each object carries a refractive
  index; the ray bends at every crossing, and total internal reflection is handled.
  The intersection quadratic runs in float64: float32 catastrophically cancels here
  and produces a "hairy fiber" artefact (see `docs/DECISIONS.md`).
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
  `lambda = 550 nm`, 1.155 um for NA 0.10. Ray tracing alone is geometric optics and
  produces edges sharper than any real objective can form; without this the picture
  is visibly blocky at higher zoom. See `loop_sim/renderer/optics.py`.

**X-ray path** (`/beam`, `/xray`, `render.py --xray`) is a separate modality: the beam is grid-cast
through the scene as straight lines, and per-material path lengths give illuminated
volume, absorbed dose and a transmission radiograph via Beer-Lambert. The objective
PSF does not apply; there is no objective.

**Template replay.** When serving from a pre-computed sweep, depth translation is
approximated as a Gaussian defocus whose width grows with distance from the focal
plane. Everything else (rotation, translation, zoom) is exact: the camera is
orthographic, so those are image-space transforms of the rendered master.

### 1b. What is not modelled

- **Partial coherence.** `na_condenser / na_objective = 0.70` (< 1) means the imaging
  is partially coherent, so real edges overshoot and ring slightly. The Gaussian PSF
  is the incoherent approximation and will not reproduce that.
- **Interference and phase contrast.** Amplitude only; no propagation of phase.
- **Polarisation.** Rays carry no polarisation state.
- **Chromatic effects.** A single 550 nm wavelength, so no dispersion and no colour
  fringing.
- **Scattering.** Absorption and refraction only; no diffuse or Mie scattering.

Rationale and the measurements behind each choice are in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## 2. Quick start (pre-built scene)

Build the env: `bash setup_venv.bash` (see [Environment](#7-environment)).

```bash
# Render the default scene at rest position
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --n-cond 7

# Rotate the sample 45° about the goniometer spindle and render
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --rotx 45 --n-cond 7

# Translate the loop so the crystal is off-center
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --tx 0.05 --ty -0.02 --n-cond 7

# Single-shot X-ray radiograph at the same pose
.venv/bin/python render.py data/scene_files/hampton_300um.yaml --xray
```

`--n-cond 7` uses one centre + six-point hex ring of condenser rays per pixel,
giving smooth edge transitions. Use `--n-cond 1` for a fast binary-NA preview.
Add `--device cuda` to render through the GPU-resident engine; CPU is the default
reference and both produce the same image.

Which motor is the spindle (φ) depends on the scene: it is whichever of
`rotx`/`roty`/`rotz` matches the scene's rotation-axis config and the pin's
mounting direction. For the bundled `data/scene_files/` scenes the spindle is
`rotx`; `roty`/`rotz` tilt the sample out of that plane.

Output is written to `data/scene_files/hampton_300um.jpg` (or `--output myfile.jpg`).
`--xray` writes `data/scene_files/hampton_300um_xray.png` instead. `render.py` has no
`--zoom` or `--tz`: set them in the scene's `motor:` block, or use the server, whose
`/motor` endpoint takes all seven axes.

---

## 3. Full pipeline from a real loop image

Each step writes its output beside itself at the repo root (`hoop.yaml`,
`loop.yaml`, `droplet.yaml`, `crystal.yaml`, `scene.yaml`); those outputs are
gitignored. No real loop photo to digitize? `data/scene_files/examples/hoop.yaml`
is a sample digitized hoop: pass it to step 3b in place of `hoop.yaml` and skip
step 3a.

### 3a. Digitize the fiber

```bash
.venv/bin/python digitize_fiber.py real_loop.jpg \
    --pixel-size 0.8233 \       # µm per pixel for your camera
    --diameter   0.020  \       # fiber diameter in mm
    --output hoop.yaml
```

An interactive window opens.  Click to place waypoints around the loop fiber,
starting at the crossover (stem attachment point).  Press **Enter** when done.

### 3b. Add the twisted-pair stem

```bash
.venv/bin/python add_stem.py hoop.yaml \
    --stem-length 0.7 \         # mm of stem visible in the field of view
    --output loop.yaml
```

Outputs three tube objects: `hoop`, `stem_1`, `stem_2`.

### 3c. Add a solvent droplet

```bash
.venv/bin/python add_droplet.py hoop.yaml \
    --volume 0.001 \            # mm³ (≈ 1 nL)
    --output droplet.yaml
```

Produces a biconvex lens mesh in contact with the hoop fiber around the loop.

### 3d. Add a crystal (optional)

```bash
.venv/bin/python add_crystal.py hoop.yaml \
    --preset plate \            # cube | plate | needle | hexagonal
    --dim 0.04 0.01 \           # half-widths in mm (plate: ab_half c_half)
    --output crystal.yaml
```

Crystal orientation defaults to identity; supply `--a-axis`/`--b-axis`/`--c-axis`
(Å, XDS convention) to match your data collection geometry.

### 3e. Assemble the scene

```bash
.venv/bin/python generate_scene.py loop.yaml crystal.yaml droplet.yaml \
    --template template.yaml \
    --output scene.yaml
```

Object order sets scene priority (first = highest).  Crystal before droplet
is required so the crystal is not masked by the solvent mesh.

### 3f. Render

```bash
.venv/bin/python render.py scene.yaml --n-cond 7
```

---

## 4. Live MJPEG server

The server exposes an AXIS-compatible HTTP interface so it can replace a real
beamline camera in any software that speaks AXIS (MxCuBE, EPICS areaDetector,
browser, VLC, etc.).

```bash
.venv/bin/python -m loop_sim.server.camera_server --scene data/scene_files/hampton_300um.yaml --port 8081
```

A new server needs five flags:

- `--templates {on,off}` (default on): serve from a pre-computed frame library,
  building it first if absent. `off` raytraces every frame live.
- `--supersample`: template sampling factor when a build is needed (builder
  default 4).
- `--n-cond` (default 7): condenser rays for settled frames.
- `--fps-limit` (default 30): MJPEG stream rate cap.
- `--engine {auto,torch,numpy}` (default auto): render backend; `auto` uses the
  GPU-resident engine when CUDA is present.

The rest, including the preview-animation and runtime-scene-switching flags, are
in [`docs/RUNBOOK.md`](docs/RUNBOOK.md#5-flags) "Flags".

Instant `/motor` sets, how AXIS-style consumers such as MxCuBE and EPICS drive
the goniometer, count as motion too: a stream of `/motor` updates gets fast
preview frames, and one exact frame renders once the pose settles.

Or from Python:

```python
from loop_sim.scene.scene       import load
from loop_sim.server.camera_server import CameraServer

scene  = load("data/scene_files/hampton_300um.yaml")
server = CameraServer(scene, host="0.0.0.0", port=8081, n_cond=7)
server.start()   # blocks; Ctrl-C to stop
```

### 4a. Pre-computed templates (the default)

The camera is orthographic, so the spindle is the only motor that changes
image content: everything else is an image-space transform. The server
therefore renders one 360° sweep per scene ahead of time, stores it in
`data/frame_library/`, and serves every frame by cropping, scaling and
blurring a template.

The server itself never builds a library. `--templates on` (the default)
serves whatever is on disk, current or stale, warning once when it is stale;
a scene with no library at all renders live and the server prints the exact
build command to run. `--templates off` raytraces every frame live regardless.
Building is `.venv/bin/python -m loop_sim.library --scene <scene.yaml>`; see
`docs/RUNBOOK.md` "Frame libraries" for the build flags.

Frames are served in **single-digit milliseconds** and **no GPU is needed at
runtime**: a GPU only accelerates building a library.

### 4b. Interactive control page

Open **`http://<host>:<port>/`** in a browser for a live control panel: the
MJPEG view with a centre crosshair, a jog pad, zoom buttons, a speed dial,
click-in-image-to-recentre, and a goniometer target panel (type X, Y, Z in mm
and φ in degrees, press **GO**). A tab strip switches scenes live (see
"Switching scenes" below). Moves are animated: the sample interpolates to the target on a
trapezoidal velocity profile (accelerates over ~0.15 s, holds, decelerates),
about 4 s to cross the screen and 30 rpm for the spindle, scaled by the speed
dial.

### 4c. HTTP endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Interactive control page (HTML) |
| `GET /axis-cgi/mjpg/video.cgi` | MJPEG stream; `camera=N` serves that request at a zoom stop from `--camera-zoom` (default `1:1.0,2:0.5,3:0.25`) |
| `GET /axis-cgi/jpg/image.cgi` | Single JPEG snapshot; same `camera=N` zoom-stop override |
| `GET/POST /motor?tx=0.05&roty=45` | Set motors instantly, returns JSON state |
| `GET/POST /move?drotx=90&speed=2` | **Animated** move; returns target JSON state (see below) |
| `GET /recenter?px=400&py=300` | Animated move bringing a pixel to the centre |
| `GET /status` | `{"positions": {...}, "target": {...}, "moving": bool}`: live pose, last commanded target, whether an animated move is running |
| `GET/POST /video-trigger?state=open\|closed` | While open, every newly published frame is POSTed to `--jpeg-receiver`; `GET` returns `{"state": ...}` |
| `GET /beam` | X-ray illuminated volumes + Beer-Lambert attenuation (JSON) |
| `GET /xray` | X-ray transmission radiograph, single-shot, rendered live (grayscale PNG) |
| `GET /scenes` | Switchable scenes and the state of each one's frame library (JSON) |
| `GET /scene` | The scene being served, plus progress/errors of any switch in flight |
| `POST /scene?path=<scene>&build=preview\|full` | Switch scenes at runtime (see below) |

**Motor parameters:** `tx`, `ty`, `tz` (mm), `rotx`, `roty`, `rotz` (degrees),
`zoom` (dimensionless; `zoom=2` halves pixel size).

**`/move` parameters:** any absolute motor key, relative deltas (`dtx`, `drotx`,
`dzoom`, …), screen-fraction pan (`panx`, `pany`; ±1 = one field of view),
`speed` (`>1` faster, `<1` slow-motion), and `duration=<s>` to fix the move's
wall time instead of `speed` (`duration=0` commits the target at once, like
`/motor`).

### 4d. Switching scenes without restarting

The control page has a **tab per scene** in `data/scene_files/`; clicking one swaps
the served sample live without restarting or dropping the stream. The pose
resets to home on a switch: a millimetre does not mean the same thing in two
scenes with different pixel sizes.

Each tab is badged with the state of that scene's frame library:

| badge | meaning | clicking the tab |
|---|---|---|
| *(none)* | library matches the current build settings | switches immediately |
| `stale` | complete and servable, but built with different settings | switches immediately, never rebuilt automatically, and says what differs |
| `preview` | only a coarse on-demand library exists | switches immediately, at 5° steps and 1× zoom |
| `no library` | nothing to serve | offers a preview (~72 frames, minutes) or full (360 frames) build |

Building a fresh library needs a GPU (~179 s/frame on CPU, so hours even for a
preview). Build commands and what a switch costs while in flight are in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md#4-switching-scenes-on-a-running-server).

`POST /scene` returns **202** and does the work on a background thread (a build
outlasts any browser timeout); **409** if a switch is already running or the
scene has no library and you did not pick a build; **503** if a build was asked
for with no CUDA; **400** for an unknown scene. Only scenes the server itself
enumerates can be selected, so the `path` you send is a lookup key and never
touches the filesystem. A switch that fails leaves the running scene untouched.

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

`absorbed_dose` is the incident-weighted X-ray flux each material absorbs
(shadowing-aware: a material behind an absorber sees already-attenuated flux).
`transmitted_frac` is the mean surviving fraction, and `beam_transmission` is
the fraction of the whole beam that exits the sample; Σ `absorbed_dose` +
`beam_transmission` ≈ 1 by energy bookkeeping. These are relative units:
`mu_xray` values are illustrative, not absolute dosimetry.

---

## 5. Driving it from dcss and BluIce

`xtalLoopSimDHS/` is a DCSS hardware server that owns the goniometer motors in
a dcss database and drives this camera server over HTTP, so BluIce's motor
widgets and centering scripts move the simulated sample the way they move the
real goniometer. It connects to dcss under its own name, `xtalLoopSimDHS`, and
takes over the goniometer devices only from a database edited to hand them over.
It has run against a local, offline dcss only. See
[`xtalLoopSimDHS/README.md`](xtalLoopSimDHS/README.md) for the wire contract,
the database rows dcss needs, and what remains before it can run against SIM831
or the beamline.

---

## 6. Scene template

`template.yaml` controls camera geometry, NA, pixel size, beam profile, and
material optical/X-ray properties.  Key fields:

> **`template.yaml` is a reference file, not the calibration, and no shipped scene
> reads it.** It uses the camera's *hi* zoom stop and takes that stop's horizontal
> pitch (0.8233 µm) as a square pixel, but the BL831 sample camera's pixels are
> 1.110 non-square, so its vertical field of view comes out 9.91% short. The scenes
> that ship use 640 × 7.4 µm square, which is the *mid* stop rendered correctly on
> square pixels (same field of view to under 1%). See `data/real_images/README.md`.

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

`color` is a per-channel absorption factor, not a display tint: the renderer adds
`30 * (1 - color)` per mm to `mu_optical` in each channel, so a coloured material is
an absorbing one. Keep water-like solvents near-white.

---

## 7. Environment

Build the environment once:

```bash
bash setup_venv.bash
```

This creates `.venv/` from `requirements.txt` and runs the test suite;
`--force` rebuilds it. Every command in this README is `.venv/bin/python ...`
from the repo root. Details, including the beamline base interpreter and the
DHS's own separate `.venv`, are in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md#1-environment).
