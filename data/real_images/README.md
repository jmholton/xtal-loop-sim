# real_images: real BL831 sample-camera frames

Not inputs to the simulator: the ground truth its output is judged against.

All 704x480, from the BL831 sample microscope, at its two usable zoom stops
(`camera_zoom` 0.5 and >0.51; third stop a 64 mm hutch view):

| stop | um/px horizontal | um/px vertical | field of view |
|---|---|---|---|
| `mid` | 6.7324 | 7.4729 | 4739.6 x 3587.0 um |
| `hi`  | 0.8233 | 0.9139 | 579.6 x 438.7 um |

Source: `sample_camera_constant` in `home/jamesh/projects/wash_pin/claude/BL-831.dat`,
selected by `dcss/scripts/operations/gridGroupConfig.tcl:2651`, quoted per DCSS's
352x240 raster, halved above because these files are 704x480.

Pixels are not square (aspect 1.11): the simulator's one square `pixel_size` renders a
circle as a 1.11 ellipse on the real camera. Account for that before calling a shape
mismatch a geometry bug.

Two frames are 684x460, not 704x480 (`D08`, `D09`): `snap.com -trim 10` crops, a 10 px
border trimmed on every side, FOV comment recomputed to match
(`0.579607 * 684/704 = 0.563141`). Same um/px; don't divide by 704.

The Hampton scenes' square 7.4 um pixel at width 640 is this camera's square-pixel
rendition, not a discrepancy (checked 2026-08-10): `640 x 7.4 = 4736.0 um` against the
real `704 x 6.7324 = 4739.6`, and `480 x 7.4 = 3552.0` against `480 x 7.4729 = 3587.0`:
0.08% horizontally, 0.98% vertically. The `704/640 = 1.100` shape difference is the
`1.110` pixel aspect; they cancel. Rendering 704 wide at 7.4 um would over-cover the
field by +9.9%. 640 does not reproduce the frame *shape*, so the server resamples to
704x480 at delivery (`loop_sim/renderer/field.py` `to_sensor`, `--sensor-pitch`). On a
served frame the pin stays 95 px tall (703.0 um) and horizontal pitch is 6.7273 um/px
against the real 6.7324.

`template.yaml` (`pixel_size: 0.0008233`, 704x480) is the `hi` stop's HORIZONTAL pitch
at the real width, 9.91% short vertically for the same reason: the one real 10% scale
error in the set, and nothing uses the file.

## Before calibrating against these frames

COMPARE IN CAMERA SPACE. A raw render is not in the same units as these files;
treating it as one has produced a wrong conclusion. The tracer emits TRANSMITTANCE; a
photograph is what the camera RECORDED. `field.apply_camera` maps between them,
affine (`out = (e - B)*t + B`, black floor B = 0.1765): it does not preserve ratios;
the error is largest on dark bodies. A crystal reading 0.186x background in
transmittance reads 0.416x in camera space. Run the render through `to_sensor` then
`apply_camera` before comparing any number to one from this directory.

Divide the render by its clear-field level. `field.py`'s vignette is a 41.6%
peak-to-trough bowl fitted to the 2020 session; its docstring flags it as 5-7x stronger
than every other epoch. Most frames here are flatter: `D01`'s five sky boxes span 2.8%
of level, centre-vs-corner -0.5%, so a centre-body/corner-background ratio in the
render carries a vignette the photograph lacks. Left in, it shifted the NA answer by a
whole stop. Worked examples of both steps: `scratch/na_fork.py`, `scratch/d01_measure.py`.

Use them to falsify, not to fit. Three limits, all measured 2026-08-10:

- The set reproduces the open three-calibration problem rather than resolving it. The
  `B` frames carry `fov 0.585999 x -0.399545`, whose aspect (0.6818 = 480/704) implies
  square pixels; `D` frames carry `0.579607 x -0.438657`, implying the 1.11 non-square
  pitch quoted above. Same instrument, two mutually exclusive calibrations, both inside
  this directory. (`mag` is derived from each file's `fov` comment, not hand-typed.)
- The A and C sets carry no `fov` comment, so their scale is inferred, not
  self-consistent: `C06`/`C07`/`C09` measure a pin at 34-37 px while `C01`/`A01`/`A05`
  measure 56-64 px on the same day, and a 0.7 mm pin at the `mid` pitch should be
  93.7 px. A comment-carrying frame, `D07`, measures 709.9 um against a 700 um nominal:
  1.4% off. Use comment-carrying frames for anything dimensional; A/C only for tone,
  specular and background, which are scale-free.
- No single "real background" exists. Level spans 117-254 across the set, and the
  mottle pattern is session-specific: the 2020 field correlates r = 0.99-1.00 within its
  own session but only 0.11 / -0.31 / -0.18 / +0.35 with 2005 / 2021 / 2025 / 2026, with
  5-7x smaller amplitude in every other epoch. `loop_sim/renderer/field.py` ships the
  2020 shape because the work was judged against it: plausible, not universal.

## How to use these

`MANIFEST.tsv` gives, per file: magnification, subject, why it was kept, and the exact
source path. Filenames are `<group><n>_<subject>_<mag>.jpg`.

Frames tagged `NOT MODELLED` (ice, devitrification) are kept deliberately, marking
where a mismatch against the simulator is expected, not to be chased.

## Provenance

Sampled 44 frames on 2026-08-10 from six trees, stratified against what
`crystal_harvester` can build (Hampton loops 100-1000 um, MiTeGen kapton micromounts, a
0.7 mm bevelled pin, a twisted-pair stem, a spherical-cap solvent lens, a faceted crystal)
and against the open fidelity questions in `docs/HANDOFF.md`. Rebuild with
`build_refs.py` (kept in the session scratch, not shipped).

Two sources were considered and rejected:

- `/home/hollatz/projects/deep_learning/datasets/try1_original_rgb/`: 224x224 8-bit
  grayscale `fake_*.png`. Synthetic or heavily-processed CNN crops, not camera frames.
- `/home/jamesh/projects/training_data/overhead*` (40,809 frames) and
  `cryo_bubble/drop*/ice_*.jpg`: the overhead goniometer camera, not the microscope.
  They carry a plausible `beam ... fov ...` JPEG comment anyway: `snap.com` stamps the
  sample camera's calibration onto whatever camera it grabbed. Judge by eye, never by
  the comment.

Coverage caveat: `loop_center_AI` holds 27,000 frames but only one mount type (pin
position varies), so it contributes 6 frames here, not a proportional share.
