"""Shared pytest setup: the ``slow`` marker and headless rendering.

Tests whose name contains ``render`` need an OpenGL context. On Linux without a display, MuJoCo is pointed at EGL
(unless ``MUJOCO_GL`` is already set); when no context can be created, those tests are skipped instead of failing.
"""

import functools
import os
import sys

import pytest

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    # Must happen before mujoco is imported by any test module.
    os.environ.setdefault("MUJOCO_GL", "egl")


@functools.cache
def _gl_unavailable_reason() -> str | None:
    import mujoco

    try:
        context = mujoco.GLContext(16, 16)
        context.make_current()
        context.free()
    except Exception as error:  # MuJoCo raises different errors per GL backend.
        return f"No OpenGL context available ({type(error).__name__}: {error})."
    return None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: takes longer than about a minute (deselect with -m 'not slow')",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    render_items = [
        item for item in items if "render" in getattr(item, "originalname", item.name)
    ]
    if render_items and (reason := _gl_unavailable_reason()) is not None:
        for item in render_items:
            item.add_marker(pytest.mark.skip(reason=reason))
