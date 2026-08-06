"""Diffraction point-spread function for the bright-field imaging path.

The ray tracer is geometric optics with a binary NA collection gate: a ray is
either collected by the objective or it is not.  That produces edges sharper
than any real objective can form.  Measured on the shipped hampton scene before
this module existed: 97.7% of a rendered frame was pure 0 or pure 255 and a
silhouette edge resolved in ~1 template pixel, where an NA 0.10 objective at
550 nm has a Rayleigh resolution of 3.35 um and cannot do better than ~1.8
template pixels.  Convolving with the objective's point-spread function is what
turns a geometric render into what the optics would actually form; without it
the picture is visibly blocky as soon as you magnify past the sampling budget.

ONE IMPLEMENTATION, CALLED BY BOTH RENDERERS.  `microscope.render` (numpy
reference) and `render_torch` (GPU engine) are asserted byte-identical after
uint8 quantisation by tests/test_torch_render_parity.py.  Two blur
implementations -- say scipy here and a torch conv2d there -- would differ in
kernel truncation, normalisation, border handling and summation order, and a
single ULP is enough to cross a rounding boundary and break that guarantee.  So
this module is numpy-only and the torch engine round-trips through it.  Do not
"optimise" it into a device-side convolution without also deciding what happens
to the byte-identity guarantee.

`microscope.py` deliberately does NOT import torch (the CPU reference is
documented as working without PyTorch), which is the other reason the shared
implementation lives here in numpy rather than in the torch engine.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# Illumination wavelength, mm.  550 nm -- green, the eye's peak sensitivity and
# the conventional stand-in for white-light illumination.  A single wavelength
# means no chromatic effects are modelled; see README "What is not modelled".
WAVELENGTH_MM = 0.00055

# Below this the kernel is indistinguishable from the identity and the filter is
# skipped.  Matches the threshold the template defocus blur already uses in
# camera_server.TemplateSource.render, so the two blurs behave consistently.
MIN_SIGMA_PX = 0.05


def psf_sigma_px(camera_cfg, eff_px):
    """Gaussian sigma, in pixels, approximating the objective's Airy PSF.

    sigma = 0.21 * lambda / NA is the standard Gaussian fit to the widefield
    Airy intensity PSF; the Airy first zero for comparison is 0.61 * lambda / NA.

    `eff_px` is mm per pixel AT THE SAMPLE (pixel_size / zoom), so the width
    scales correctly with both zoom and template supersampling: the blur is a
    fixed size in object space, and the same physical PSF is 0.156 px on a
    7.4 um camera pixel but 0.624 px in a 4x supersampled template of the same
    scene.  That is exactly why the softening only becomes visible when you
    magnify -- which is where the geometric sharpness became visible too.
    """
    na_obj = float(camera_cfg.get("na_objective", 0.10))
    if na_obj <= 0.0 or eff_px <= 0.0:
        return 0.0
    return 0.21 * WAVELENGTH_MM / na_obj / float(eff_px)


def apply_psf(img, sigma_px):
    """Convolve an (H, W, 3) image with the PSF.  Returns a new array.

    `mode="nearest"` replicates the border rather than reflecting it: a
    microscope field of view is a window onto a larger scene, so the sample
    continues past the edge -- reflecting would fabricate a mirrored copy of it.
    The sigma is applied per spatial axis only, never across colour.
    """
    if sigma_px is None or sigma_px < MIN_SIGMA_PX:
        return img
    return gaussian_filter(img, sigma=(float(sigma_px), float(sigma_px), 0.0),
                           mode="nearest")


def apply_psf_for(img, camera_cfg, eff_px, enabled=True):
    """Convenience wrapper: resolve sigma from the camera, then convolve."""
    if not enabled:
        return img
    return apply_psf(img, psf_sigma_px(camera_cfg, eff_px))
