"""Parameters of the in-silico jumping spider (Salticidae).

Units and scale: SI units (metres, kilograms, seconds, radians, newton metres) for a **1:10 scale model** of a large
salticid such as *Phidippus regius* (body length ~15 mm in nature, ~13 cm here). Gravity is not scaled, so the model
behaves like a 13 cm legged robot shaped like a jumping spider rather than a dynamically similar miniature; tiny
millimetre-scale bodies cannot be simulated stably at practical timesteps.

Mass comes from a single uniform tissue density (``density``, 1000 kg/m^3, i.e. water) applied to the body geometry,
which gives a cephalothorax of 94 g, an abdomen of 155 g and legs of 8.9 g each, 320 g in total (see the README).

Joint conventions (identical for all legs on both sides of the body):

* in-plane joints rotate around the segment's dorsal (z) axis; positive values swing the leg backwards (towards the
  abdomen) on both sides;
* out-of-plane joints rotate around the segment's lateral axis; positive values flex the leg ventrally (downwards).

Joint ranges are always given as ``(lower, upper)`` in radians in this convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

#: Leg segments from the body outwards.
LEG_SEGMENT_NAMES = (
    "coxa",
    "trochanter",
    "femur",
    "patella",
    "tibia",
    "metatarsus",
    "tarsus",
)
#: Joint-to-joint segment lengths (m).
LEG_SEGMENT_LENGTHS = (0.0116, 0.0068, 0.0246, 0.0127, 0.0176, 0.0143, 0.0118)
#: Segment radii (m).
LEG_SEGMENT_RADII = (0.00585, 0.00415, 0.00513, 0.00425, 0.0047, 0.00275, 0.00285)
#: Resting angle of each segment relative to its parent, out of plane (positive is downwards).
LEG_SEGMENT_ATTACHMENT_ANGLES = tuple(np.deg2rad([-21, 0, -11, 40, 56, 23, -53]))
#: Out-of-plane range of motion (lower, upper) of each segment's joint.
LEG_SEGMENT_OOP_RANGES = tuple(
    zip(
        np.deg2rad([-13, 0, -46, -40, -20, -96, -31]),
        np.deg2rad([39, 60, 56, 107, 0, 27, 91]),
    )
)
#: Total in-plane range of motion of each segment's joint (centred around 0; 0 means no in-plane joint).
LEG_SEGMENT_IP_ROMS = tuple(np.deg2rad([35, 70, 0, 0, 70, 15, 65]))
#: Maximum actuator torque per segment (N m), proportional to the squared segment radius (muscle cross-section).
#: 0.8 N m at the coxa lets the 0.32 kg model stand comfortably and jump ~3 body lengths with a scripted open-loop
#: controller (see ``examples/scripted_jump.py``).
LEG_SEGMENT_TORQUE_LIMITS = tuple(
    0.8 * (np.asarray(LEG_SEGMENT_RADII) / LEG_SEGMENT_RADII[0]) ** 2
)
#: Angle between the body's x-axis and each leg pair's base, in the horizontal plane.
LEG_PAIR_IN_PLANE_ANGLES = tuple(np.deg2rad([30, 63.3, 96.7, 130]))
#: Elevation of the leg bases on the cephalothorax surface (negative is below the equator).
LEG_OUT_OF_PLANE_ANGLE = float(np.deg2rad(-40))


@dataclass
class JointSpecification:
    """A hinge joint.

    Args:
        range: ``(lower, upper)`` joint limits (rad).
        stiffness: Passive spring towards the resting pose (N m / rad).
        damping: Passive viscous damping (N m s / rad).
        armature: Rotor inertia added to the joint (kg m^2). Leg segments weigh only a few grams; armature keeps the
            joint dynamics resolvable at a 1 ms timestep (in a robot it would be the reflected motor inertia).
    """

    range: Tuple[float, float]
    stiffness: float = 0.005
    damping: float = 0.002
    armature: float = 2e-5


@dataclass
class LegSegmentSpecification:
    """One leg segment: a capsule from this segment's joint to the next one.

    Args:
        name: Segment name (e.g. ``"femur"``).
        length: Joint-to-joint length (m).
        radius: Capsule radius (m).
        attachment_angle: Resting out-of-plane angle relative to the parent segment (rad, positive is downwards).
        oop_joint: Out-of-plane joint.
        ip_joint: In-plane joint, or ``None`` for segments that only flex out of plane (femur, patella).
        torque_limit: Maximum actuator torque of this segment's joints (N m).
    """

    name: str
    length: float
    radius: float
    attachment_angle: float
    oop_joint: JointSpecification
    ip_joint: Optional[JointSpecification]
    torque_limit: float


@dataclass
class LegSpecification:
    """One leg pair (mirrored onto the left and right side).

    Args:
        in_plane_angle: Angle between the body's forward axis and the leg base, in the horizontal plane (rad).
        out_of_plane_angle: Elevation of the leg base on the cephalothorax surface (rad, negative is below).
        segments: Segments from coxa to tarsus.
    """

    in_plane_angle: float
    out_of_plane_angle: float
    segments: List[LegSegmentSpecification]


@dataclass
class CephalothoraxSpecification:
    """Ellipsoidal cephalothorax (prosoma) with half-sizes ``size`` (m) along x (forward), y (left) and z (up)."""

    size: Tuple[float, float, float] = (0.03, 0.03, 0.025)


@dataclass
class AbdomenSpecification:
    """Ellipsoidal abdomen (opisthosoma), attached behind the cephalothorax by a 2-DoF joint (the pedicel).

    Args:
        size: Half-sizes (m).
        inset_factor: Fraction of the abdomen's length that overlaps with the pedicel joint.
        ip_joint: Yaw joint (positive turns the abdomen's tip to the right).
        oop_joint: Pitch joint (positive raises the abdomen's tip).
        torque_limit: Maximum actuator torque (N m).
    """

    size: Tuple[float, float, float] = (0.04, 0.033, 0.028)
    inset_factor: float = 0.2
    ip_joint: JointSpecification = field(
        default_factory=lambda: JointSpecification(range=(-np.pi / 4, np.pi / 4))
    )
    oop_joint: JointSpecification = field(
        default_factory=lambda: JointSpecification(range=(-np.pi / 4, np.pi / 4))
    )
    torque_limit: float = 0.2


@dataclass
class DraglineSpecification:
    """Optional dragline (safety line): a spatial tendon from the spinnerets to an anchor in the arena.

    Spiders pay out silk while jumping and can brake its release to control their attitude. The silk release brake is
    modelled as a damper actuator (``force = -brake_damping * ctrl * velocity``, ``ctrl`` in [0, 1]) that can only pull.

    Args:
        enabled: Add the dragline.
        stiffness: Passive tendon stiffness (N / m) around its initial length.
        damping: Passive tendon damping (N s / m).
        brake_damping: Maximum damping of the brake actuator (N s / m).
        max_brake_force: Maximum pulling force of the brake (N).
    """

    enabled: bool = False
    stiffness: float = 0.0
    damping: float = 0.01
    brake_damping: float = 5.0
    max_brake_force: float = 5.0


@dataclass
class ActuationSpecification:
    """Joint actuation.

    Args:
        position_control: Position servos (actions are joint angle targets in rad) instead of torque motors (actions
            are torques in N m).
        saturation_error: Position control only: the joint angle error (rad) at which a servo reaches its torque
            limit, i.e. ``kp = torque_limit / saturation_error``.
        damping_ratio: Position control only: damping ratio of each servo (``kv``) given the joint's inertia. Kept low
            because servo damping brakes the fast leg extensions that jumping requires.
        activation_time_constant: Time constant (s) of the first-order low-pass filter between the action and the
            actuator (cf. muscle activation dynamics). Without it (and the 1 ms physics timestep of the environments),
            saturated actions that switch at the control rate whip the gram-scale leg segments to hundreds of rad/s and
            the airborne body spins fast enough to make the simulation diverge.
    """

    position_control: bool = True
    saturation_error: float = 0.25
    damping_ratio: float = 0.3
    activation_time_constant: float = 0.01


@dataclass
class JumpingSpiderMorphologySpecification:
    """The complete morphology. ``legs`` lists the four leg pairs I-IV from front to back."""

    cephalothorax: CephalothoraxSpecification
    abdomen: AbdomenSpecification
    legs: List[LegSpecification]
    dragline: DraglineSpecification
    actuation: ActuationSpecification
    density: float = 1000.0
    leg_friction: float = 1.0
    body_friction: float = 0.5


def default_leg_segment_specifications() -> List[LegSegmentSpecification]:
    segments = []
    for name, length, radius, angle, oop_range, ip_rom, torque_limit in zip(
        LEG_SEGMENT_NAMES,
        LEG_SEGMENT_LENGTHS,
        LEG_SEGMENT_RADII,
        LEG_SEGMENT_ATTACHMENT_ANGLES,
        LEG_SEGMENT_OOP_RANGES,
        LEG_SEGMENT_IP_ROMS,
        LEG_SEGMENT_TORQUE_LIMITS,
    ):
        ip_joint = None
        if ip_rom > 0:
            ip_joint = JointSpecification(range=(-ip_rom / 2, ip_rom / 2))
        segments.append(
            LegSegmentSpecification(
                name=name,
                length=length,
                radius=radius,
                attachment_angle=float(angle),
                oop_joint=JointSpecification(range=tuple(map(float, oop_range))),
                ip_joint=ip_joint,
                torque_limit=float(torque_limit),
            )
        )
    return segments


def default_jumping_spider_morphology_specification(
    dragline: bool = False, position_control: bool = True
) -> JumpingSpiderMorphologySpecification:
    """The default 1:10 scale jumping spider (4 leg pairs with 7 segments and 12 actuated joints each)."""
    legs = [
        LegSpecification(
            in_plane_angle=float(angle),
            out_of_plane_angle=LEG_OUT_OF_PLANE_ANGLE,
            segments=default_leg_segment_specifications(),
        )
        for angle in LEG_PAIR_IN_PLANE_ANGLES
    ]
    return JumpingSpiderMorphologySpecification(
        cephalothorax=CephalothoraxSpecification(),
        abdomen=AbdomenSpecification(),
        legs=legs,
        dragline=DraglineSpecification(enabled=dragline),
        actuation=ActuationSpecification(position_control=position_control),
    )
