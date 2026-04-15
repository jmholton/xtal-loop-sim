#!/usr/bin/env python3
"""
Generate a PNG beam profile image for loop-sim.

Each pixel represents one X-ray mini-beam sampling point; its 16-bit value
is proportional to the relative beam intensity at that position.  The physical
pixel size is embedded in the PNG as a text chunk (pixel_size_mm) so beam.py
reads it back without extra configuration.

The image is the central beam entity: provide your own (e.g. from a real YAG
scintillator measurement) or generate one here from analytic parameters.

Usage
-----
    python3 make_beam_image.py [options]

    --fwhm-h MM      Horizontal FWHM in mm  (default 0.040 = 40 µm)
    --fwhm-v MM      Vertical   FWHM in mm  (default 0.070 = 70 µm)
    --pinhole MM     Round pinhole diameter in mm (default 0.100, 0 = no mask)
    --pixel-size MM  Physical size of one pixel in mm (default 0.001 = 1 µm)
    --output PATH    Output PNG path (default: beam.png)

Reference this image from template.yaml:

    beam:
      image: beam.png
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image, PngImagePlugin


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--fwhm-h',    type=float, default=0.040,
                   metavar='MM',
                   help='Horizontal FWHM in mm (default 0.040 = 40 µm)')
    p.add_argument('--fwhm-v',    type=float, default=0.070,
                   metavar='MM',
                   help='Vertical FWHM in mm (default 0.070 = 70 µm)')
    p.add_argument('--pinhole',   type=float, default=0.0,
                   metavar='MM',
                   help='Pinhole diameter in mm baked into the image (default 0 = off).\n'
                        'Prefer setting pinhole_diameter in template.yaml instead so the\n'
                        'aperture can be changed without regenerating the image.')
    p.add_argument('--pixel-size', type=float, default=0.001,
                   metavar='MM',
                   help='mm per pixel (default 0.001 = 1 µm)')
    p.add_argument('--output',    default='beam.png',
                   help='Output PNG path (default: beam.png)')
    return p.parse_args()


def make_beam_image(fwhm_h, fwhm_v, pinhole_mm, pixel_size_mm):
    """
    Build a 2-D Gaussian beam profile masked by a circular pinhole.

    Parameters
    ----------
    fwhm_h, fwhm_v : float — horizontal / vertical FWHM in mm
    pinhole_mm      : float — pinhole diameter in mm (0 = no mask)
    pixel_size_mm   : float — mm per pixel

    Returns
    -------
    arr16 : (H, W) uint16 ndarray, values 0–65535
    """
    px        = pixel_size_mm
    pinhole_r = pinhole_mm / 2.0

    # Image extent: cover the pinhole (or 3-sigma Gaussian extent if larger)
    # plus a 10-pixel margin on each side so the mask edge is not clipped.
    sigma_max  = max(fwhm_h, fwhm_v) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half_mm    = max(pinhole_r, 3.0 * sigma_max) + 10 * px
    n          = int(np.ceil(2 * half_mm / px))
    if n % 2 == 0:
        n += 1          # odd → centre pixel is exactly on-axis

    cx = cy = (n - 1) / 2.0
    iy_g, ix_g = np.mgrid[0:n, 0:n]
    x = (ix_g - cx) * px   # mm, horizontal (camera_fast)
    y = (iy_g - cy) * px   # mm, vertical   (camera_slow)

    # 2-D elliptical Gaussian
    sx = fwhm_h / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sy = fwhm_v / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    profile = np.exp(-x**2 / (2.0 * sx**2) - y**2 / (2.0 * sy**2))

    # Circular pinhole aperture
    if pinhole_r > 0:
        profile[x**2 + y**2 > pinhole_r**2] = 0.0

    return (profile * 65535).clip(0, 65535).astype(np.uint16)


def main():
    args = parse_args()

    arr16 = make_beam_image(args.fwhm_h, args.fwhm_v,
                            args.pinhole, args.pixel_size)

    pil  = Image.fromarray(arr16)
    meta = PngImagePlugin.PngInfo()
    meta.add_text("pixel_size_mm", f"{args.pixel_size:.8g}")
    pil.save(args.output, pnginfo=meta)

    n         = arr16.shape[0]
    n_nonzero = int((arr16 > 0).sum())
    print(f"Beam image: {n}×{n} px,  pixel = {args.pixel_size * 1e3:.2f} µm",
          file=sys.stderr)
    print(f"  Gaussian  FWHM {args.fwhm_h * 1e3:.1f} µm (H) "
          f"× {args.fwhm_v * 1e3:.1f} µm (V)",
          file=sys.stderr)
    if args.pinhole > 0:
        print(f"  Pinhole   ⌀{args.pinhole * 1e3:.0f} µm", file=sys.stderr)
    print(f"  Nonzero pixels: {n_nonzero}  (= mini-beam count)", file=sys.stderr)
    print(f"  → {args.output}", file=sys.stderr)
    print(f"\nAdd to template.yaml:\n  beam:\n    image: {args.output}",
          file=sys.stderr)


if __name__ == '__main__':
    main()
