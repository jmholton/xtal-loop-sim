"""Tripwire: the renderer-source digest the shipped frame libraries were built with.

A red run means an edit landed in one of the files `render_sha` hashes (see
`_RENDER_SOURCES` in loop_sim/library/frame_library.py). That does not rebuild
anything any more, but it does mean the three tracked libraries may no longer
match a live render: run `python -m loop_sim.library --verify --scene <scene>`
and, if the pixels moved, rebuild. If the pixels did not move, update the
literal below.
"""
from loop_sim.library.frame_library import render_sha

SHIPPED_RENDER_SHA = "db3a7ad5732274364b2c48e7ad09e632027bcbb00294a6f82f7d454b693335d1"


def test_render_sha_matches_the_shipped_libraries():
    assert render_sha() == SHIPPED_RENDER_SHA
