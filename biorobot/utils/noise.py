"""
Code mainly copied from: https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py

Changes:
    - added Jax variant
    - merged jax and numpy implementation
    - added typing
    - added seed
"""

from typing import Optional, Tuple, Union

import jax
import numpy as np

ArrayType = Union[np.ndarray, jax.numpy.ndarray]


def interpolant(t: ArrayType) -> ArrayType:
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
    shape: Tuple[int, int],
    res: Tuple[int, int],
    rng_key: Optional[Union[int, jax.random.PRNGKey]] = None,
    npi: Union[np, jax.numpy] = np,
) -> ArrayType:
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note: shape must be a multiple of
            res.
        rng_key: The seed used for numpy's RNG or the key for jax's RNG.
        npi: The numpy implementation to use (numpy or jax.numpy).
    Returns:
        A numpy or jax array of the given shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    if npi == jax.numpy:
        random_values = jax.random.uniform(
            rng_key, shape=(res[0] + 1, res[1] + 1), minval=0, maxval=1
        )
    else:
        rs = np.random.RandomState(seed=rng_key)
        random_values = rs.rand(res[0] + 1, res[1] + 1)

    def _generate(random_values: ArrayType) -> ArrayType:
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = (
            npi.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0)
            % 1
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
        t = interpolant(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        noise = npi.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

        return (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    if npi == jax.numpy:
        return jax.jit(_generate)(random_values)
    else:
        return _generate(random_values)
