"""Simulation backends.

Every BRT environment implements its task logic once, against the small interface defined here:

* ``mjc``: regular MuJoCo (``mujoco.MjData``, numpy, CPU).
* ``mjx``: MuJoCo XLA with the JAX implementation (``mjx.Data``, jit-able and vmap-able, CPU/GPU/TPU).
* ``warp``: MuJoCo XLA with the MuJoCo Warp implementation (same JAX interface as ``mjx``, NVIDIA GPUs).

Task code therefore uses ``backend.xp`` (``numpy`` or ``jax.numpy``) for array math and the helpers below for the few
operations that differ between the two (functional updates, random sampling, stepping).
"""

from __future__ import annotations

import abc
import warnings
from typing import Any, List, Sequence, Tuple, Union

import mujoco
import numpy as np

Array = Any  # np.ndarray or jax.Array
RNG = Any  # np.random.Generator or jax PRNG key


class Backend(abc.ABC):
    name: str
    xp: Any

    def __init__(self, mj_model: mujoco.MjModel) -> None:
        self.mj_model = mj_model

    @abc.abstractmethod
    def make_data(self) -> Any:
        """Fresh simulation data in the model's default (qpos0) configuration."""

    @abc.abstractmethod
    def replace(self, data: Any, **fields: Array) -> Any:
        """Return ``data`` with ``fields`` replaced (in place for ``mjc``, functionally for ``mjx``)."""

    @abc.abstractmethod
    def forward(self, data: Any) -> Any: ...

    @abc.abstractmethod
    def step(self, data: Any, ctrl: Array, num_steps: int) -> Any: ...

    @abc.abstractmethod
    def at_set(self, array: Array, index: Any, value: Array) -> Array:
        """``array[index] = value`` as a functional update."""

    @abc.abstractmethod
    def make_rng(self, seed: Union[int, RNG]) -> RNG: ...

    @abc.abstractmethod
    def uniform(
        self, rng: RNG, low: Array = 0.0, high: Array = 1.0, shape: Sequence[int] = ()
    ) -> Tuple[Array, RNG]:
        """Sample uniformly; returns the sample and the RNG to use next."""

    @abc.abstractmethod
    def to_mj_data(self, data: Any, index: int | None = None) -> mujoco.MjData:
        """A CPU ``MjData`` copy of (one environment in) ``data``, used for rendering."""


class MJCBackend(Backend):
    name = "mjc"
    xp = np

    def make_data(self) -> mujoco.MjData:
        return mujoco.MjData(self.mj_model)

    def replace(self, data: mujoco.MjData, **fields: Array) -> mujoco.MjData:
        for field, value in fields.items():
            getattr(data, field)[:] = value
        return data

    def forward(self, data: mujoco.MjData) -> mujoco.MjData:
        mujoco.mj_forward(self.mj_model, data)
        return data

    def step(self, data: mujoco.MjData, ctrl: Array, num_steps: int) -> mujoco.MjData:
        data.ctrl[:] = ctrl
        mujoco.mj_step(self.mj_model, data, nstep=num_steps)
        return data

    def at_set(self, array: Array, index: Any, value: Array) -> Array:
        array = np.array(array, copy=True)
        array[index] = value
        return array

    def make_rng(self, seed: Union[int, RNG]) -> np.random.Generator:
        if isinstance(seed, np.random.Generator):
            return seed
        return np.random.default_rng(seed)

    def uniform(self, rng, low=0.0, high=1.0, shape=()):
        return rng.uniform(low=low, high=high, size=shape), rng

    def to_mj_data(
        self, data: mujoco.MjData, index: int | None = None
    ) -> mujoco.MjData:
        return data


class MJXBackend(Backend):
    """JAX backend shared by MJX's ``jax`` and ``warp`` implementations.

    Args:
        num_envs: Upper bound on the number of environments that are vmapped together. Only the ``warp``
            implementation uses it: MuJoCo Warp allocates one contact buffer that is shared by *all* vmapped worlds, so
            it must be sized for the whole batch up front.
    """

    def __init__(
        self, mj_model: mujoco.MjModel, impl: str = "jax", num_envs: int = 1
    ) -> None:
        super().__init__(mj_model)
        check_backend_support(mj_model, "mjx" if impl == "jax" else impl)
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        self._jax = jax
        self._mjx = mjx
        self.xp = jnp
        self.name = "mjx" if impl == "jax" else impl
        self.mjx_model = mjx.put_model(mj_model, impl=impl)
        self._impl = impl
        self.num_envs = num_envs
        if impl == "warp" and num_envs > 1 and mj_model.nconmax <= 0:
            warnings.warn(
                "The model does not set <size nconmax> (MjSpec.nconmax, contacts per world): MuJoCo Warp's default "
                f"contact buffer is sized for a single world and will overflow with {num_envs} environments.",
                stacklevel=2,
            )

    def make_data(self):
        # MuJoCo Warp builds its data from the CPU model; the JAX implementation accepts either.
        sizes = {}
        if self._impl == "warp":
            # MuJoCo Warp preallocates contacts and constraints with small defaults; models with many contacts set
            # <size nconmax> (MjSpec.nconmax, contacts per world) and <size njmax> (MjSpec.njmax, constraint rows per
            # world) to raise them. The contact buffer (naconmax) is shared by all vmapped worlds.
            if self.mj_model.nconmax > 0:
                sizes["naconmax"] = self.mj_model.nconmax * self.num_envs
            if self.mj_model.njmax > 0:
                sizes["njmax"] = self.mj_model.njmax
        return self._mjx.make_data(self.mj_model, impl=self._impl, **sizes)

    def replace(self, data, **fields):
        return data.replace(**{k: self.xp.asarray(v) for k, v in fields.items()})

    def forward(self, data):
        return self._mjx.forward(self.mjx_model, data)

    def step(self, data, ctrl, num_steps: int):
        data = data.replace(ctrl=ctrl)
        return self._jax.lax.fori_loop(
            0, num_steps, lambda _, d: self._mjx.step(self.mjx_model, d), data
        )

    def at_set(self, array, index, value):
        return self.xp.asarray(array).at[index].set(value)

    def make_rng(self, seed):
        if isinstance(seed, (int, np.integer)):
            return self._jax.random.PRNGKey(int(seed))
        return seed

    def uniform(self, rng, low=0.0, high=1.0, shape=()):
        rng, sub_rng = self._jax.random.split(rng)
        sample = self._jax.random.uniform(
            sub_rng, shape=tuple(shape), minval=low, maxval=high
        )
        return sample, rng

    def to_mj_data(self, data, index: int | None = None) -> mujoco.MjData:
        if index is not None:
            # MuJoCo Warp keeps some buffers (e.g. contacts) shared across vmapped worlds: only take batched leaves.
            batch_size = data.time.shape[0]
            data = self._jax.tree_util.tree_map(
                lambda x: x[index] if x.ndim and x.shape[0] == batch_size else x, data
            )
        mj_data = mujoco.MjData(self.mj_model)
        self._mjx.get_data_into(mj_data, self.mj_model, data)
        return mj_data


BACKENDS = ("mjc", "mjx", "warp")


def unsupported_features(mj_model: mujoco.MjModel, backend: str) -> List[str]:
    """Descriptions of the model features that ``backend`` cannot simulate correctly (empty if it supports them all).

    Only features that a backend gets *silently* wrong are listed here; MJX and MuJoCo Warp already raise for the
    features they reject themselves.
    """
    features = []
    if backend == "mjx" and np.any(mj_model.geom_fluid[:, 0] > 0):
        # MJX's JAX implementation ignores fluidshape="ellipsoid" and applies the inertia-based fluid model to those
        # geoms instead (tested with MuJoCo 3.17: the fluid forces differ by ~13% for a capsule chain).
        features.append(
            "the ellipsoid fluid model (geoms with fluidshape='ellipsoid'), which MJX's JAX implementation replaces by "
            "the inertia-based fluid model without notice"
        )
    return features


def check_backend_support(mj_model: mujoco.MjModel, backend: str) -> None:
    """Raise ``NotImplementedError`` if ``backend`` cannot simulate ``mj_model`` correctly (see
    :func:`unsupported_features`)."""
    features = unsupported_features(mj_model, backend)
    if features:
        supported = [b for b in BACKENDS if not unsupported_features(mj_model, b)]
        raise NotImplementedError(
            f"backend='{backend}' does not support {'; '.join(features)}. Use backend={' or '.join(map(repr, supported))}"
            f", or change the model."
        )


def make_backend(name: str, mj_model: mujoco.MjModel, num_envs: int = 1) -> Backend:
    """``"mjc"``: MuJoCo (C, CPU). ``"mjx"``: MJX with the JAX implementation. ``"warp"``: MJX with the MuJoCo Warp
    implementation (NVIDIA GPUs; install with ``pip install biorobot[warp]``). ``num_envs`` is the maximum number of
    vmapped environments (see :class:`MJXBackend`)."""
    name = name.lower()
    if name == "mjc":
        return MJCBackend(mj_model)
    if name == "mjx":
        return MJXBackend(mj_model, impl="jax", num_envs=num_envs)
    if name == "warp":
        return MJXBackend(mj_model, impl="warp", num_envs=num_envs)
    raise ValueError(f"Unknown backend '{name}'. Choose one of {BACKENDS}.")
