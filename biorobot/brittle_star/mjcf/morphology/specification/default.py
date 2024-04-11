from typing import List, Union

import numpy as np

from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarActuationSpecification,
    BrittleStarArmSegmentSpecification,
    BrittleStarArmSpecification,
    BrittleStarDiskSpecification,
    BrittleStarJointSpecification,
    BrittleStarMorphologySpecification,
)

START_SEGMENT_RADIUS = 0.025
STOP_SEGMENT_RADIUS = 0.0125
START_SEGMENT_LENGTH = 0.075
STOP_SEGMENT_LENGTH = 0.025
DISK_DIAMETER = 0.25
DISK_HEIGHT = 0.025


def linear_interpolation(alpha: float, start: float, stop: float) -> float:
    return start + alpha * (stop - start)


def default_joint_specification(range: float) -> BrittleStarJointSpecification:
    joint_specification = BrittleStarJointSpecification(
        range=range, stiffness=0.1, damping=0.5, armature=0.02
    )

    return joint_specification


def default_arm_segment_specification(
    alpha: float,
) -> BrittleStarArmSegmentSpecification:
    in_plane_joint_specification = default_joint_specification(
        range=30 / 180 * np.pi
    )  # 30
    out_of_plane_joint_specification = default_joint_specification(
        range=30 / 180 * np.pi
    )  # 5

    radius = linear_interpolation(
        alpha=alpha, start=START_SEGMENT_RADIUS, stop=STOP_SEGMENT_RADIUS
    )
    length = linear_interpolation(
        alpha=alpha, start=START_SEGMENT_LENGTH, stop=STOP_SEGMENT_LENGTH
    )

    segment_specification = BrittleStarArmSegmentSpecification(
        radius=radius,
        length=length,
        in_plane_joint_specification=in_plane_joint_specification,
        out_of_plane_joint_specification=out_of_plane_joint_specification,
    )
    return segment_specification


def default_arm_specification(num_segments_per_arm: int) -> BrittleStarArmSpecification:
    segment_specifications = list()
    for segment_index in range(num_segments_per_arm):
        segment_specification = default_arm_segment_specification(
            alpha=segment_index / num_segments_per_arm
        )
        segment_specifications.append(segment_specification)

    arm_specification = BrittleStarArmSpecification(
        segment_specifications=segment_specifications
    )
    return arm_specification


def default_brittle_star_morphology_specification(
    num_arms: int = 5,
    num_segments_per_arm: Union[int, List[int]] = 5,
    use_p_control: bool = False,
    use_torque_control: bool = False,
    radius_to_strength_factor: float = 200,
) -> BrittleStarMorphologySpecification:
    disk_specification = BrittleStarDiskSpecification(
        diameter=DISK_DIAMETER, height=DISK_HEIGHT
    )

    if isinstance(num_segments_per_arm, int):
        num_segments_per_arm = [num_segments_per_arm] * num_arms
    else:
        assert len(num_segments_per_arm) == num_arms, (
            f"Length of the 'num_segments_per_arm' input must be"
            f"equal to the 'num_arms' input."
        )

    arm_specifications = list()
    for num_segments in num_segments_per_arm:
        arm_specification = default_arm_specification(num_segments_per_arm=num_segments)
        arm_specifications.append(arm_specification)

    actuation_specification = BrittleStarActuationSpecification(
        use_p_control=use_p_control,
        use_torque_control=use_torque_control,
        radius_to_strength_factor=radius_to_strength_factor,
    )
    specification = BrittleStarMorphologySpecification(
        disk_specification=disk_specification,
        arm_specifications=arm_specifications,
        actuation_specification=actuation_specification,
    )

    return specification
