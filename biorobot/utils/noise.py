"""2D Perlin noise for numpy and jax.numpy.

Adapted from https://github.com/pvigier/perlin-numpy (MIT): one implementation for both array libraries, typed and
seeded. The JAX variant is traceable, so it can be generated inside a jitted/vmapped ``reset``.
"""

from types import ModuleType
from typing import Any, Tuple

import numpy as np


def _interpolant(t: Any) -> Any:
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    rng_key: Any = None,
    npi: ModuleType = np,
) -> Any:
    """Generate a 2D array of Perlin noise, normalised to ``[0, 1]``.

    Args:
        shape: Shape of the generated array; each side must be a multiple of the matching ``res``.
        res: Number of noise periods along each axis.
        rng_key: Seed for numpy's RNG (``npi=numpy``) or a JAX PRNG key (``npi=jax.numpy``).
        npi: The array library to use (``numpy`` or ``jax.numpy``).
    """
    if npi is np:
        random_values = np.random.RandomState(seed=rng_key).rand(res[0] + 1, res[1] + 1)
    else:
        import jax

        random_values = jax.random.uniform(rng_key, shape=(res[0] + 1, res[1] + 1))

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = (
        npi.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    )
    # Gradients
    angles = 2 * npi.pi * random_values
    gradients = npi.dstack((npi.cos(angles), npi.sin(angles)))
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[: -d[0], : -d[1]]
    g10 = gradients[d[0] :, : -d[1]]
    g01 = gradients[: -d[0], d[1] :]
    g11 = gradients[d[0] :, d[1] :]
    # Ramps
    n00 = npi.sum(npi.dstack((grid[:, :, 0], grid[:, :, 1])) * g00, 2)
    n10 = npi.sum(npi.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = npi.sum(npi.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = npi.sum(npi.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = _interpolant(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    noise = npi.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
    return (noise - npi.min(noise)) / (npi.max(noise) - npi.min(noise))
