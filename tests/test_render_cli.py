"""render.py, run as the command a user types.

  * `--xray` writes an 8-bit greyscale PNG of the camera's size, at the
    default `<scene_stem>_xray.png` beside the scene and at `--output`
    -> test_xray_writes_a_png_of_the_camera_size
  * the optical CLI still parses -> test_help_lists_the_optical_and_xray_flags

The scene is a tmp copy of mitegen_200um.yaml shrunk to 40x30, so the CPU
radiograph takes seconds.
"""
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "data", "scene_files", "mitegen_200um.yaml")
RES = (40, 30)


def _run(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "render.py"), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def _small_scene(tmp_path):
    with open(SCENE) as fh:
        doc = yaml.safe_load(fh)
    doc["camera"]["width"], doc["camera"]["height"] = RES
    path = tmp_path / "mitegen_small.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(doc, fh)
    return path


def test_xray_writes_a_png_of_the_camera_size(tmp_path):
    from PIL import Image
    scene = _small_scene(tmp_path)

    r = _run(str(scene), "--xray")
    assert r.returncode == 0, r.stderr
    default_out = tmp_path / "mitegen_small_xray.png"
    img = Image.open(default_out)
    assert img.format == "PNG" and img.mode == "L" and img.size == RES

    explicit = tmp_path / "x.png"
    r = _run(str(scene), "--xray", "--rotx", "90", "--output", str(explicit))
    assert r.returncode == 0, r.stderr
    assert Image.open(explicit).size == RES


def test_help_lists_the_optical_and_xray_flags():
    r = _run("--help")
    assert r.returncode == 0, r.stderr
    for flag in ("--tx", "--rotx", "--n-cond", "--device", "--output", "--xray"):
        assert flag in r.stdout
