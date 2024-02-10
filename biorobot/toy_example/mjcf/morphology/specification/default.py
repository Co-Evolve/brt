import numpy as np

from biorobot.toy_example.mjcf.morphology.specification.specification import (
    ToyExampleActuationSpecification,
    ToyExampleArmSegmentSpecification,
    ToyExampleArmSpecification,
    ToyExampleJointSpecification,
    ToyExampleMorphologySpecification,
    ToyExampleTorsoSpecification,
)


def default_joint_specification() -> ToyExampleJointSpecification:
    joint_specification = ToyExampleJointSpecification(
        range=20 / 180 * np.pi, stiffness=0.1, damping=0.1
    )
    return joint_specification


def default_actuation_specification() -> ToyExampleActuationSpecification:
    actuation_specification = ToyExampleActuationSpecification(kp=10)
    return actuation_specification


def default_arm_segment_specification() -> ToyExampleArmSegmentSpecification:
    joint_specification = default_joint_specification()
    segment_specification = ToyExampleArmSegmentSpecification(
        radius=0.025, length=0.1, joint_specification=joint_specification
    )
    return segment_specification


def default_arm_specification(num_segments: int) -> ToyExampleArmSpecification:
    segment_specifications = []
    for _ in range(num_segments):
        segment_specification = default_arm_segment_specification()
        segment_specifications.append(segment_specification)

    arm_specification = ToyExampleArmSpecification(
        segment_specifications=segment_specifications
    )
    return arm_specification


def default_torso_specification() -> ToyExampleTorsoSpecification:
    torso_specification = ToyExampleTorsoSpecification(radius=0.1)
    return torso_specification


def default_toy_example_morphology_specification(
    num_arms: int, num_segments_per_arm: int
) -> ToyExampleMorphologySpecification:

    torso_specification = default_torso_specification()

    arm_specifications = []
    for _ in range(num_arms):
        arm_specification = default_arm_specification(num_segments=num_segments_per_arm)
        arm_specifications.append(arm_specification)

    actuation_specification = default_actuation_specification()

    morphology_specification = ToyExampleMorphologySpecification(
        torso_specification=torso_specification,
        arm_specifications=arm_specifications,
        actuation_specification=actuation_specification,
    )

    return morphology_specification
