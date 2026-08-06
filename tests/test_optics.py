"""
Objective point-spread function (loop_sim/renderer/optics.py).

The PSF is what stops the render being sharper than the optics it models. These
tests pin the width to the physics rather than to a remembered number, and pin
the two behaviours that would make it silently useless: becoming a no-op, and
being applied across the colour axis.

Run:  pytest tests/test_optics.py -v
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.renderer.optics import (WAVELENGTH_MM, MIN_SIGMA_PX, apply_psf,
                                      apply_psf_for, psf_sigma_px)

HAMPTON_CAM = {"pixel_size": 0.0074, "na_objective": 0.10, "na_condenser": 0.07}
MITEGEN_CAM = {"pixel_size": 0.0010, "na_objective": 0.10, "na_condenser": 0.07}


def test_sigma_follows_the_rayleigh_limit():
    """sigma = 0.21 lambda / NA, expressed in whatever pixel the render uses."""
    sigma_mm = 0.21 * WAVELENGTH_MM / 0.10
    assert psf_sigma_px(HAMPTON_CAM, 0.0074) == pytest.approx(sigma_mm / 0.0074)
    # ~1.155 um of blur: 0.156 px on a 7.4 um camera pixel ...
    assert psf_sigma_px(HAMPTON_CAM, 0.0074) == pytest.approx(0.156, abs=0.005)
    # ... and 0.624 px in a 4x supersampled template of the same scene, which is
    # why the softening only shows up once you magnify.
    assert psf_sigma_px(HAMPTON_CAM, 0.0074 / 4) == pytest.approx(0.624, abs=0.005)
    # mitegen samples 7.4x finer, so the same physical PSF is well resolved
    assert psf_sigma_px(MITEGEN_CAM, 0.0010) == pytest.approx(1.155, abs=0.005)


def test_sigma_scales_with_zoom():
    """eff_px is pixel_size/zoom, so zooming in widens the blur in pixels."""
    s1 = psf_sigma_px(HAMPTON_CAM, 0.0074)
    s4 = psf_sigma_px(HAMPTON_CAM, 0.0074 / 4)
    assert s4 == pytest.approx(4 * s1)


def test_a_higher_na_resolves_finer():
    assert psf_sigma_px({"na_objective": 0.50}, 0.001) < \
           psf_sigma_px({"na_objective": 0.10}, 0.001)


def test_degenerate_camera_is_not_a_crash():
    assert psf_sigma_px({"na_objective": 0.0}, 0.001) == 0.0
    assert psf_sigma_px(HAMPTON_CAM, 0.0) == 0.0


def _step_image():
    img = np.zeros((16, 16, 3), dtype=np.float64)
    img[:, 8:, :] = 1.0
    return img


def test_below_threshold_is_exactly_a_no_op():
    """A sub-threshold sigma must return the input untouched, not almost.

    Coarse renders (a 7.4 um camera pixel at zoom 1) sit near this boundary, and
    an 'almost' no-op there would perturb the CPU/GPU comparison for nothing.
    """
    img = _step_image()
    out = apply_psf(img, MIN_SIGMA_PX / 2)
    assert out is img
    assert np.array_equal(apply_psf_for(img, HAMPTON_CAM, 0.0074, enabled=False), img)


def test_psf_softens_a_step_edge():
    img = _step_image()
    out = apply_psf(img, 1.155)
    row = out[8, :, 0]
    assert row[7] > 0.0 and row[8] < 1.0, "the step did not spread at all"
    # No ringing: a Gaussian is strictly positive, so it cannot overshoot the
    # input range beyond float rounding in the normalised kernel sum.
    assert row.min() >= -1e-12 and row.max() <= 1.0 + 1e-12
    # monotonic across the edge -- a Gaussian cannot ring
    band = row[5:12]
    assert np.all(np.diff(band) >= -1e-12)


def test_psf_does_not_blur_across_colour():
    """sigma is applied per spatial axis only; a colour fringe must not bleed."""
    img = np.zeros((16, 16, 3), dtype=np.float64)
    img[:, :, 1] = 1.0                     # pure green field
    out = apply_psf(img, 2.0)
    assert out[:, :, 0].max() == pytest.approx(0.0)
    assert out[:, :, 2].max() == pytest.approx(0.0)
    assert out[:, :, 1].min() == pytest.approx(1.0)


def test_psf_preserves_total_intensity():
    """A normalised kernel conserves flux away from the borders."""
    rng = np.random.default_rng(0)
    img = rng.random((64, 64, 3))
    out = apply_psf(img, 1.5)
    inner = (slice(8, -8), slice(8, -8))
    assert out[inner].mean() == pytest.approx(img[inner].mean(), rel=2e-2)
