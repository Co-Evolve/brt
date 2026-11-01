"""RGBA colours of the seahorse tail (those of the original model)."""

import numpy as np

from biorobot.utils.colors import hex_to_rgba

PLATE = hex_to_rgba("#7db5a8")
CONNECTOR = hex_to_rgba("#595959")
VERTEBRAE = np.array([158, 38, 212, 255]) / 255
SITE = hex_to_rgba("#b75659")
STRUT = hex_to_rgba("#595959")
HM_BEAM = hex_to_rgba("#73c2fb")
MVM_BEAM = hex_to_rgba("#b75659")
