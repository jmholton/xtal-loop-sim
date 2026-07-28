"""Pre-computed frame libraries (rotation sweeps) for replay without rendering."""
from .frame_library import (          # noqa: F401
    DEFAULT_ROOT,
    build_library,
    crop_window,
    ensure_library,
    frame_for_angle,
    is_current,
    library_dir,
    load_manifest,
)
