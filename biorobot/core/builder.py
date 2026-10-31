"""Thin helpers for building morphologies and arenas with ``mujoco.MjSpec``.

A :class:`Morphology` is the robot (built from a specification dataclass), an :class:`Arena` is the world it is placed
in. :func:`compose` attaches a morphology to an arena and returns a single ``MjSpec`` that an environment compiles.
Both expose the raw ``MjSpec`` as ``.spec`` so users can freely modify the model before an environment is created.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Generic, Optional, Sequence, TypeVar

import mujoco
import numpy as np

SpecificationT = TypeVar("SpecificationT")


class _SpecComponent(abc.ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.spec = mujoco.MjSpec()
        self.spec.modelname = name
        self.spec.compiler.degree = False

    def compile(self) -> mujoco.MjModel:
        return self.spec.compile()

    def to_xml(self) -> str:
        self.spec.compile()
        return self.spec.to_xml()

    def export(self, path: str | Path) -> None:
        """Write the component as an MJCF XML file (handy to inspect it in ``python -m mujoco.viewer``)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_xml())


class Morphology(_SpecComponent, Generic[SpecificationT], abc.ABC):
    """Base class for a parameterised robot morphology.

    Subclasses build their bodies under ``self.root`` (a body directly below the worldbody) in :meth:`_build`. Element
    names are given without a prefix; once attached to an arena, every name is prefixed with ``f"{name}/"``.
    """

    def __init__(self, specification: SpecificationT, name: str) -> None:
        super().__init__(name=name)
        self.specification = specification
        self._build()

    @abc.abstractmethod
    def _build(self) -> None: ...


class Arena(_SpecComponent, abc.ABC):
    """Base class for an arena: ground, lights, cameras, and task-specific objects.

    Physics options that should apply to the whole simulation (timestep, integrator, medium density, ...) belong on
    the arena's ``spec.option``, since options of an attached morphology are discarded by MuJoCo.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self._build()

    @abc.abstractmethod
    def _build(self) -> None: ...


def compose(
    arena: Arena,
    morphology: Morphology,
    position: Sequence[float] = (0.0, 0.0, 0.0),
    euler: Sequence[float] = (0.0, 0.0, 0.0),
    free_joint: bool = True,
    root_body: Optional[str] = None,
) -> mujoco.MjSpec:
    """Attach (a copy of) ``morphology`` to (a copy of) ``arena`` and return the combined spec.

    Args:
        position: Position of the attachment frame in the world.
        euler: Orientation (xyz Euler angles, radians) of the attachment frame.
        free_joint: Add a free joint named ``f"{morphology.name}/freejoint"`` to the morphology's root body.
        root_body: Unprefixed name of the morphology's root body (defaults to the first body under the worldbody).
    """
    world = arena.spec.copy()
    robot = morphology.spec.copy()
    if root_body is None:
        root_body = robot.worldbody.first_body().name

    frame = world.worldbody.add_frame(
        pos=np.asarray(position, dtype=float),
        quat=euler_to_quat(np.asarray(euler, dtype=float)),
    )
    prefix = f"{morphology.name}/"
    world.attach(robot, prefix=prefix, frame=frame)
    if free_joint:
        world.body(prefix + root_body).add_freejoint(name=f"{prefix}freejoint")
    return world


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Extrinsic xyz Euler angles to a MuJoCo (w, x, y, z) quaternion."""
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray(euler, dtype=float), "xyz")
    return quat
