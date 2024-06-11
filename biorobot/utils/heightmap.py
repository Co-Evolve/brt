from importlib.resources import Package
from typing import Sequence, Union

import chex
import numpy as np

from biorobot.utils.noise import generate_perlin_noise_2d


def generate_hfield(shape: Sequence[int], rng: Union[np.random.RandomState, chex.PRNGKey],
                    noise_scale: int, npi: Package) -> chex.Array:
    hfield = generate_perlin_noise_2d(shape=shape, res=(noise_scale, noise_scale), rng=rng,
                                      npi=npi)
    hfield = (hfield - hfield.min()) / (hfield.max() - hfield.min())
    return hfield


def generate_radial_matrix(shape: Sequence[int], inner_radius: float, outer_radius: float, npi: Package) -> np.ndarray:
    # Create a grid of coordinates
    r, c = shape
    y, x = npi.ogrid[-r / 2:r / 2, -c / 2:c / 2]

    # Compute the distance from the center
    distance = npi.sqrt(x ** 2 + y ** 2)

    # Normalize the distance based on the given radius
    radial_matrix = distance / outer_radius

    # Clip the values to be in the range [0, 1]
    radial_matrix = npi.clip(radial_matrix, 0, 1)

    radial_matrix[distance < inner_radius] = 0

    return radial_matrix
