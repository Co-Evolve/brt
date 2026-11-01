"""Brittle star colour palette as RGBA arrays in ``[0, 1]`` (the convention of MuJoCo's ``rgba`` attributes)."""

import numpy as np


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> np.ndarray:
    """Convert ``"#rrggbb"`` to an RGBA array in ``[0, 1]``."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected a '#rrggbb' colour, got '#{hex_color}'.")
    rgb = [int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    return np.array([*rgb, alpha])


rgba_green = hex_to_rgba("#7db5a8")
rgba_red = hex_to_rgba("#b75659")
rgba_gray = hex_to_rgba("#595959")

rgba_tendon_relaxed = np.array([214, 174, 114, 190]) / 255.0

rgba_sand = np.array([225, 191, 146, 255]) / 255.0
