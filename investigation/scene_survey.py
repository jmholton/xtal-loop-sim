"""Render a spread of loop/pin scenes at several spindle angles and assemble
a labelled contact sheet, so the geometry can be eyeballed against reality.

Usage:  python survey.py <out_dir> <scene.yaml> [scene.yaml ...]
"""
import sys, os, time
import numpy as np
import torch
from PIL import Image, ImageDraw

from loop_sim.scene.scene import load
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer.engine_torch import TorchScene, render_torch

ANGLES = [0.0, 45.0, 90.0]
N_COND = 7


def shape_kinds(scene):
    """Count shape class names across the whole scene tree."""
    def walk(s):
        yield s
        for attr in ("children", "A", "B"):
            c = getattr(s, attr, None)
            if c is None:
                continue
            for x in (c if isinstance(c, (list, tuple)) else [c]):
                yield from walk(x)
    kinds = {}
    for o in scene.objects:
        for s in walk(o.shape):
            kinds[type(s).__name__] = kinds.get(type(s).__name__, 0) + 1
    return kinds


def render(tscene, scene, rotx, n_cond=N_COND):
    """Render, backing off resolution on OOM (the mesh path has no AABB cull,
    so its (rays x faces) working set can exceed VRAM at full resolution)."""
    cam = scene.camera_cfg
    W0, H0 = cam["width"], cam["height"]
    gono = Goniometer(scene.geometry)
    gono.set(rotx=rotx)
    for div in (1, 2, 4, 8):
        cam["width"], cam["height"] = W0 // div, H0 // div
        try:
            img = render_torch(tscene, gono, n_cond=n_cond)
            a = (img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            im = Image.fromarray(a, mode="RGB")
            if div > 1:
                im = im.resize((W0, H0), Image.NEAREST)
            return im, div
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
        finally:
            cam["width"], cam["height"] = W0, H0
    raise RuntimeError("OOM even at 1/8 resolution")


def main():
    out_dir, scene_paths = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for path in scene_paths:
        tag = os.path.splitext(os.path.basename(path))[0]
        scene = load(path, device="cpu")
        cam = scene.camera_cfg
        px = cam["pixel_size"]
        print(f"[{tag}] objects={[o.name for o in scene.objects]}")
        print(f"[{tag}] shapes={shape_kinds(scene)}")
        print(f"[{tag}] camera {cam['width']}x{cam['height']} px={px*1000:.2f}um "
              f"FOV={cam['width']*px:.3f}x{cam['height']*px:.3f}mm "
              f"NA_obj={cam['na_objective']} NA_cond={cam['na_condenser']}")
        tscene = TorchScene(scene, dev, torch.float64)
        imgs = []
        for ang in ANGLES:
            t0 = time.time()
            im, div = render(tscene, scene, ang)
            dt = time.time() - t0
            fn = os.path.join(out_dir, f"{tag}_rotx{int(ang):03d}.png")
            im.save(fn)
            note = "" if div == 1 else f"  [OOM backoff: rendered at 1/{div} res]"
            print(f"    rotx={ang:5.1f}  {dt:6.2f}s  -> {os.path.basename(fn)}{note}")
            imgs.append(im)
        rows.append((tag, imgs))
        del tscene
        torch.cuda.empty_cache()

    # contact sheet
    w, h = rows[0][1][0].size
    pad, label_h, lead = 8, 22, 190
    sheet = Image.new("RGB", (lead + len(ANGLES) * (w + pad) + pad,
                             label_h + len(rows) * (h + pad) + pad), "white")
    d = ImageDraw.Draw(sheet)
    for j, ang in enumerate(ANGLES):
        d.text((lead + j * (w + pad) + w // 2 - 30, 6), f"rotx = {ang:g}°", fill="black")
    for i, (tag, imgs) in enumerate(rows):
        y = label_h + i * (h + pad) + pad
        d.text((6, y + h // 2), tag, fill="black")
        for j, im in enumerate(imgs):
            sheet.paste(im, (lead + j * (w + pad) + pad, y))
    sheet_path = os.path.join(out_dir, "contact_sheet.png")
    sheet.save(sheet_path)
    print("contact sheet ->", sheet_path)


if __name__ == "__main__":
    main()
