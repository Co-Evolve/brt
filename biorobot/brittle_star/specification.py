"""Brittle star morphology specification: plain dataclasses and the default (parameterised) brittle star.

Units are SI (metres, kilograms per cubic metre, radians). The defaults are based on the brittle star *Ophiomastix
wendtii*; densities come from https://doi.org/10.1088/1748-3190/adbecb.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Union

import numpy as np

START_SEGMENT_RADIUS = 0.025
STOP_SEGMENT_RADIUS = 0.0125
START_SEGMENT_LENGTH = 0.075
STOP_SEGMENT_LENGTH = 0.025
DISK_DIAMETER = 0.25
DISK_HEIGHT = 0.025

ARM_DENSITY = 1530
DISK_DENSITY = 1270


@dataclass
class BrittleStarJointSpecification:
    """A hinge joint between two arm segments.

    Args:
        range: Symmetric joint limit; the joint moves within ``[-range, range]`` radians.
        stiffness: Joint stiffness (N m / rad) pulling the joint back to zero.
        damping: Joint damping (N m s / rad).
        armature: Rotor inertia added to the joint (kg m^2); stabilises the light, stiff arm chains.
    """

    range: float
    stiffness: float
    damping: float
    armature: float


@dataclass
class BrittleStarArmSegmentSpecification:
    """One arm segment: a capsule of ``length`` (cylinder part) and ``radius``, with an in-plane (yaw) and an
    out-of-plane (pitch) joint at its proximal end."""

    radius: float
    length: float
    density: float
    in_plane_joint_specification: BrittleStarJointSpecification
    out_of_plane_joint_specification: BrittleStarJointSpecification


@dataclass
class BrittleStarArmSpecification:
    segment_specifications: List[BrittleStarArmSegmentSpecification] = field(
        default_factory=list
    )

    @property
    def number_of_segments(self) -> int:
        return len(self.segment_specifications)


@dataclass
class BrittleStarDiskSpecification:
    """The central disk; collisions use a cylinder of ``radius`` and half-height ``height``."""

    radius: float
    height: float
    density: float


@dataclass
class BrittleStarActuationSpecification:
    """How the arm joints are actuated.

    Exactly one of ``use_p_control`` (position servos) and ``use_torque_control`` (motors) must be set. With
    ``use_tendons``, every segment is moved by four contracting tendons instead of joint motors (torque control only).
    The actuator strength of a segment is ``radius_to_strength_factor * segment radius``.
    """

    use_tendons: bool
    use_p_control: bool
    use_torque_control: bool
    radius_to_strength_factor: float

    def __post_init__(self) -> None:
        if self.use_p_control + self.use_torque_control != 1:
            raise ValueError(
                "Use exactly one of 'use_p_control' and 'use_torque_control'."
            )
        if self.use_tendons and not self.use_torque_control:
            raise ValueError("Only torque control is supported with tendons.")


@dataclass
class BrittleStarSensorSpecification:
    """``num_contact_sensors_per_segment``: 1 touch sensor covering the whole capsule, or N sensors placed around its
    circumference."""

    num_contact_sensors_per_segment: int = 1


@dataclass
class BrittleStarMorphologySpecification:
    disk_specification: BrittleStarDiskSpecification
    arm_specifications: List[BrittleStarArmSpecification]
    actuation_specification: BrittleStarActuationSpecification
    sensor_specification: BrittleStarSensorSpecification

    @property
    def number_of_arms(self) -> int:
        return len(self.arm_specifications)

    @property
    def number_of_non_empty_arms(self) -> int:
        return int(np.count_nonzero(self.number_of_segments_per_arm))

    @property
    def number_of_segments_per_arm(self) -> np.ndarray:
        return np.array([arm.number_of_segments for arm in self.arm_specifications])

    @property
    def total_number_of_segments(self) -> int:
        return int(np.sum(self.number_of_segments_per_arm))


def linear_interpolation(alpha: float, start: float, stop: float) -> float:
    return start + alpha * (stop - start)


def default_joint_specification(range: float) -> BrittleStarJointSpecification:
    return BrittleStarJointSpecification(
        range=range, stiffness=0.005, damping=1.0, armature=0.2
    )


def default_arm_segment_specification(
    alpha: float,
) -> BrittleStarArmSegmentSpecification:
    """Segment at relative position ``alpha`` in ``[0, 1)`` along the arm; arms taper linearly towards the tip."""
    return BrittleStarArmSegmentSpecification(
        radius=linear_interpolation(alpha, START_SEGMENT_RADIUS, STOP_SEGMENT_RADIUS),
        length=linear_interpolation(alpha, START_SEGMENT_LENGTH, STOP_SEGMENT_LENGTH),
        density=ARM_DENSITY,
        in_plane_joint_specification=default_joint_specification(
            range=45 / 180 * np.pi
        ),
        out_of_plane_joint_specification=default_joint_specification(
            range=30 / 180 * np.pi
        ),
    )


def default_arm_specification(num_segments_per_arm: int) -> BrittleStarArmSpecification:
    return BrittleStarArmSpecification(
        segment_specifications=[
            default_arm_segment_specification(alpha=index / num_segments_per_arm)
            for index in range(num_segments_per_arm)
        ]
    )


def default_brittle_star_morphology_specification(
    num_arms: int = 5,
    num_segments_per_arm: Union[int, Sequence[int]] = 5,
    use_tendons: bool = False,
    use_p_control: bool = False,
    use_torque_control: bool = False,
    radius_to_strength_factor: float = 150,
    num_contact_sensors_per_segment: int = 1,
) -> BrittleStarMorphologySpecification:
    """The default brittle star.

    Args:
        num_arms: Number of arms (at most 5; arms are placed at multiples of 72 degrees).
        num_segments_per_arm: Segments per arm, either the same for every arm or one value per arm (0 removes an arm).
        use_tendons: Actuate segments with tendons (requires ``use_torque_control``).
        use_p_control: Position control: actions are target joint angles.
        use_torque_control: Torque control: actions are joint torques (or tendon forces with ``use_tendons``).
        radius_to_strength_factor: Actuator strength per metre of segment radius.
        num_contact_sensors_per_segment: Number of touch sensors per segment.
    """
    if num_arms > 5:
        raise ValueError("The brittle star has at most 5 arms.")
    if isinstance(num_segments_per_arm, int):
        num_segments_per_arm = [num_segments_per_arm] * num_arms
    elif len(num_segments_per_arm) != num_arms:
        raise ValueError(
            "The length of 'num_segments_per_arm' must be equal to 'num_arms'."
        )

    return BrittleStarMorphologySpecification(
        disk_specification=BrittleStarDiskSpecification(
            radius=DISK_DIAMETER / 2, height=DISK_HEIGHT, density=DISK_DENSITY
        ),
        arm_specifications=[
            default_arm_specification(num_segments_per_arm=num_segments)
            for num_segments in num_segments_per_arm
        ],
        actuation_specification=BrittleStarActuationSpecification(
            use_tendons=use_tendons,
            use_p_control=use_p_control,
            use_torque_control=use_torque_control,
            radius_to_strength_factor=radius_to_strength_factor,
        ),
        sensor_specification=BrittleStarSensorSpecification(
            num_contact_sensors_per_segment=num_contact_sensors_per_segment
        ),
    )
