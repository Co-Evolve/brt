"""Helpers to turn MuJoCo sensors into :class:`~biorobot.core.env.Observable` objects."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from biorobot.core.env import Observable


def find_sensors(
    mj_model: mujoco.MjModel,
    sensor_type: mujoco.mjtSensor,
    name_contains: Optional[str | Sequence[str]] = None,
) -> List[int]:
    """Ids of all sensors of ``sensor_type`` whose name contains (all of) ``name_contains``, in id order."""
    if isinstance(name_contains, str):
        name_contains = [name_contains]
    ids = []
    for sensor_id in range(mj_model.nsensor):
        sensor = mj_model.sensor(sensor_id)
        if sensor.type[0] != sensor_type:
            continue
        if name_contains and not all(s in sensor.name for s in name_contains):
            continue
        ids.append(sensor_id)
    return ids


def sensordata_indices(
    mj_model: mujoco.MjModel, sensor_ids: Sequence[int]
) -> np.ndarray:
    """Indices into ``data.sensordata`` covering the given sensors (concatenated)."""
    indices = [
        np.arange(
            mj_model.sensor_adr[i], mj_model.sensor_adr[i] + mj_model.sensor_dim[i]
        )
        for i in sensor_ids
    ]
    return np.concatenate(indices).astype(int) if indices else np.zeros(0, dtype=int)


def float32_bounds(low: np.ndarray, high: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The tightest float32 values within ``[low, high]``, to clip observations to.

    Clipping a float32 observation (MJX/Warp) to ``low``/``high`` themselves rounds them to the nearest float32, which
    can lie just outside the (float64) observation space.
    """
    low32, high32 = np.float32(low), np.float32(high)
    low32 = np.where(low32 < low, np.nextafter(low32, np.float32(np.inf)), low32)
    high32 = np.where(high32 > high, np.nextafter(high32, np.float32(-np.inf)), high32)
    return low32.astype(np.float32), high32.astype(np.float32)


def sensor_observable(
    name: str,
    mj_model: mujoco.MjModel,
    sensor_type: mujoco.mjtSensor,
    name_contains: Optional[str | Sequence[str]] = None,
    low: Optional[np.ndarray | float] = None,
    high: Optional[np.ndarray | float] = None,
    transform: Optional[Callable] = None,
) -> Observable:
    """Observable that reads (and optionally transforms) the matching sensors from ``data.sensordata``.

    ``low``/``high`` default to ``-inf``/``inf``; scalars are broadcast to the observable's size.
    """
    indices = sensordata_indices(
        mj_model, find_sensors(mj_model, sensor_type, name_contains)
    )
    size = len(indices)
    low = -np.inf if low is None else low
    high = np.inf if high is None else high
    low = np.broadcast_to(np.asarray(low, dtype=float), (size,)).copy()
    high = np.broadcast_to(np.asarray(high, dtype=float), (size,)).copy()

    if transform is None:
        fn = lambda data, info: data.sensordata[indices]
    else:
        fn = lambda data, info: transform(data.sensordata[indices])
    return Observable(name=name, low=low, high=high, fn=fn)


def concatenate_observables(name: str, observables: Sequence[Observable]) -> Observable:
    """Merge several observables into one, in order."""
    return Observable(
        name=name,
        low=np.concatenate([o.low for o in observables]),
        high=np.concatenate([o.high for o in observables]),
        fn=lambda data, info: _concat([o.fn(data, info) for o in observables]),
    )


def _concat(arrays):
    import numpy as _np

    if any(type(a).__module__.startswith("jax") for a in arrays):
        import jax.numpy as jnp

        return jnp.concatenate([jnp.reshape(a, (-1,)) for a in arrays])
    return _np.concatenate([_np.reshape(a, (-1,)) for a in arrays])
