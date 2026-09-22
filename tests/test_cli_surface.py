"""The seven root scripts keep every flag James's version had.

The option strings below were captured from `git show master:<script>` (the
state James handed over). HEAD may add flags; it may not remove or rename one,
and the default output file names may not change. Read from source, not by
running the scripts: digitize_fiber.py imports matplotlib at the top, which a
test box need not have.
"""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MASTER_FLAGS = {
    "render.py": {"scene", "--tx", "--ty", "--rotx", "--roty", "--rotz",
                  "--n-cond", "--device", "--output"},
    "make_beam_image.py": {"--fwhm-h", "--fwhm-v", "--pinhole", "--pixel-size",
                           "--output"},
    "digitize_fiber.py": {"image", "--pixel-size", "--output", "--diameter"},
    "add_stem.py": {"hoop", "--output", "--stem-length", "--pitch-ratio",
                    "--n-samples-per-mm"},
    "add_droplet.py": {"hoop", "--output", "--volume", "--n-z", "--n-phi"},
    "add_crystal.py": {"hoop", "--output", "--preset", "--dim", "--a-axis",
                       "--b-axis", "--c-axis", "--offset"},
    "generate_scene.py": {"components", "--template", "--output"},
}

# Default output names as at master: the pipeline scripts write beside
# themselves, make_beam_image.py into the cwd.
MASTER_DEFAULT_OUTPUTS = {
    "add_stem.py": "loop.yaml",
    "add_droplet.py": "droplet.yaml",
    "add_crystal.py": "crystal.yaml",
    "generate_scene.py": "scene.yaml",
    "make_beam_image.py": "beam.png",
}

_ARG = re.compile(r"add_argument\(\s*['\"]([^'\"]+)['\"]")


def _declared_flags(script):
    with open(os.path.join(ROOT, script)) as fh:
        return set(_ARG.findall(fh.read()))


@pytest.mark.parametrize("script", sorted(MASTER_FLAGS))
def test_master_flags_survive(script):
    missing = MASTER_FLAGS[script] - _declared_flags(script)
    assert not missing, f"{script} lost flags James's version had: {sorted(missing)}"


@pytest.mark.parametrize("script", sorted(MASTER_DEFAULT_OUTPUTS))
def test_master_default_output_name_survives(script):
    with open(os.path.join(ROOT, script)) as fh:
        src = fh.read()
    assert f"'{MASTER_DEFAULT_OUTPUTS[script]}'" in src \
        or f'"{MASTER_DEFAULT_OUTPUTS[script]}"' in src
