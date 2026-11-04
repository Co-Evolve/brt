"""Swimmer morphology specification: plain dataclasses and the default (parameterised) swimmer.

The swimmer is an abstract anguilliform body: a chain of capsules in the horizontal plane, connected by hinges with
passive torsional springs and driven by position servos.

Units are SI (metres, kilograms per cubic metre, radians, newton metres).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

HEAD_SEGMENT_LENGTH = 0.1
SEGMENT_RADIUS = 0.015
SEGMENT_DENSITY = 600
JOINT_STIFFNESS = 0.01
JOINT_DAMPING = 0.005
JOINT_RANGE = 1.5
P_CONTROL_KP = 2.0
#: Wide enough for the servo targets of Van der Pol CPGs (up to ~4.9 rad, see ``examples/van_der_pol_cpg.py``), which go
#: far beyond the joint limits: with kp = 2 N m/rad, the servos then act as soft torque sources and must not be clipped.
P_CONTROL_RANGE = 2 * np.pi

FLUID_MODELS = ("ellipsoid", "inertia")


@dataclass
class SwimmerSegmentSpecification:
    """One body segment: a capsule of ``radius`` whose axis runs ``length`` metres from this segment's joint to the
    next one (the hemispherical caps of neighbouring capsules overlap)."""

    length: float
    radius: float
    density: float


@dataclass
class SwimmerJointSpecification:
    """A hinge (around the vertical axis) between two neighbouring segments.

    Args:
        stiffness: Passive torsional spring (N m/rad) pulling the joint back to straight.
        damping: Joint damping (N m s/rad).
        range: Symmetric joint limit; the joint moves within ``[-range, range]`` radians.
        armature: Rotor inertia added to the joint (kg m^2).
    """

    stiffness: float
    damping: float
    range: float
    armature: float = 0.0


@dataclass
class SwimmerActuationSpecification:
    """A position servo on every joint: torque = ``kp`` * (target - angle), with targets in
    ``[-control_range, control_range]`` radians."""

    kp: float = P_CONTROL_KP
    control_range: float = P_CONTROL_RANGE


@dataclass
class SwimmerMorphologySpecification:
    """The swimmer, from head (segment 0) to tail.

    Args:
        segment_specifications: One per segment, head first.
        joint_specifications: One per joint; joint ``i`` connects segment ``i`` to segment ``i + 1``.
        actuation_specification: The joint servos.
        fluid_model: MuJoCo's fluid model for the segments: ``"ellipsoid"`` (``fluidshape="ellipsoid"``: drag,
            added mass and lift per capsule) or ``"inertia"`` (MuJoCo's default model, based on
            each body's equivalent inertia box; the only one MJX's JAX implementation supports).
    """

    segment_specifications: List[SwimmerSegmentSpecification]
    joint_specifications: List[SwimmerJointSpecification]
    actuation_specification: SwimmerActuationSpecification = field(
        default_factory=SwimmerActuationSpecification
    )
    fluid_model: str = "ellipsoid"

    def __post_init__(self) -> None:
        if self.number_of_segments < 2:
            raise ValueError("The swimmer needs at least 2 segments.")
        if len(self.joint_specifications) != self.number_of_segments - 1:
            raise ValueError(
                f"Expected {self.number_of_segments - 1} joint specifications (one between every two segments), "
                f"got {len(self.joint_specifications)}."
            )
        if self.fluid_model not in FLUID_MODELS:
            raise ValueError(
                f"Unknown fluid model '{self.fluid_model}'. Choose one of {FLUID_MODELS}."
            )

    @property
    def number_of_segments(self) -> int:
        return len(self.segment_specifications)

    @property
    def body_length(self) -> float:
        """Distance (m) from the head's joint end to the tail's end, without the capsules' end caps."""
        return float(sum(segment.length for segment in self.segment_specifications))


def segment_lengths(
    num_segments: int, taper: float, head_segment_length: float = HEAD_SEGMENT_LENGTH
) -> List[float]:
    """Segment lengths (m), head first, changing linearly from ``head_segment_length`` to ``taper`` times that at the
    tail."""
    return [
        head_segment_length * (1 + (taper - 1) * i / (num_segments - 1))
        for i in range(num_segments)
    ]


def default_swimmer_morphology_specification(
    num_segments: int = 8,
    taper: float = 1.0,
    stiffness: float = JOINT_STIFFNESS,
    head_segment_length: float = HEAD_SEGMENT_LENGTH,
    radius: float = SEGMENT_RADIUS,
    density: float = SEGMENT_DENSITY,
    damping: float = JOINT_DAMPING,
    joint_range: float = JOINT_RANGE,
    kp: float = P_CONTROL_KP,
    control_range: float = P_CONTROL_RANGE,
    fluid_model: str = "ellipsoid",
) -> SwimmerMorphologySpecification:
    """The default swimmer.

    Args:
        num_segments: Number of capsules. Since the head segment's
            length is fixed, this also sets the body length.
        taper: Length of the tail segment relative to the head segment;
            segment lengths change linearly in between.
        stiffness: Passive torsional stiffness (N m/rad) of every joint.
        head_segment_length: Length (m) of the head segment.
        radius: Capsule radius (m), the same for all segments.
        density: Capsule density (kg/m^3), the same for all segments.
        damping: Joint damping (N m s/rad).
        joint_range: Symmetric joint limit (rad).
        kp: Servo gain (N m/rad).
        control_range: Symmetric range of the servo targets (rad).
        fluid_model: ``"ellipsoid"`` or ``"inertia"``, see :class:`SwimmerMorphologySpecification`.
    """
    return SwimmerMorphologySpecification(
        segment_specifications=[
            SwimmerSegmentSpecification(length=length, radius=radius, density=density)
            for length in segment_lengths(num_segments, taper, head_segment_length)
        ],
        joint_specifications=[
            SwimmerJointSpecification(
                stiffness=stiffness, damping=damping, range=joint_range
            )
            for _ in range(num_segments - 1)
        ],
        actuation_specification=SwimmerActuationSpecification(
            kp=kp, control_range=control_range
        ),
        fluid_model=fluid_model,
    )
