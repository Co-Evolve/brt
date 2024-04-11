from typing import List

import numpy as np
from fprs.parameters import FixedParameter
from fprs.specification import MorphologySpecification, Specification


class BrittleStarJointSpecification(Specification):
    def __init__(
        self,
        range: float,
        stiffness: float,
        damping: float,
        armature: float,
    ) -> None:
        super().__init__()
        self.stiffness = FixedParameter(value=stiffness)
        self.range = FixedParameter(value=range)
        self.damping = FixedParameter(value=damping)
        self.armature = FixedParameter(value=armature)


class BrittleStarArmSegmentSpecification(Specification):
    def __init__(
        self,
        radius: float,
        length: float,
        in_plane_joint_specification: BrittleStarJointSpecification,
        out_of_plane_joint_specification: BrittleStarJointSpecification,
    ) -> None:
        super().__init__()
        self.radius = FixedParameter(radius)
        self.length = FixedParameter(length)
        self.in_plane_joint_specification = in_plane_joint_specification
        self.out_of_plane_joint_specification = out_of_plane_joint_specification


class BrittleStarArmSpecification(Specification):
    def __init__(
        self, segment_specifications: List[BrittleStarArmSegmentSpecification]
    ) -> None:
        super().__init__()
        self.segment_specifications = segment_specifications

    @property
    def number_of_segments(self) -> int:
        return len(self.segment_specifications)


class BrittleStarDiskSpecification(Specification):
    def __init__(
        self,
        diameter: float,
        height: float,
    ) -> None:
        super().__init__()
        self.radius = FixedParameter(diameter / 2)
        self.height = FixedParameter(height)


class BrittleStarActuationSpecification(Specification):
    def __init__(
        self,
        use_p_control: bool,
        use_torque_control: bool,
        radius_to_strength_factor: float,
    ) -> None:
        super().__init__()
        assert (
            use_p_control + use_torque_control == 1
        ), "Only one actuation method can be used."

        self.use_p_control = FixedParameter(use_p_control)
        self.use_torque_control = FixedParameter(use_torque_control)
        self.radius_to_strength_factor = FixedParameter(radius_to_strength_factor)


class BrittleStarMorphologySpecification(MorphologySpecification):
    def __init__(
        self,
        disk_specification: BrittleStarDiskSpecification,
        arm_specifications: List[BrittleStarArmSpecification],
        actuation_specification: BrittleStarActuationSpecification,
    ) -> None:
        super(BrittleStarMorphologySpecification, self).__init__()
        self.disk_specification = disk_specification
        self.arm_specifications = arm_specifications
        self.actuation_specification = actuation_specification

    @property
    def number_of_arms(self) -> int:
        return len(self.arm_specifications)

    @property
    def number_of_non_empty_arms(self) -> int:
        return len(
            [
                number_of_segments
                for number_of_segments in self.number_of_segments_per_arm
                if number_of_segments > 0
            ]
        )

    @property
    def number_of_segments_per_arm(self) -> np.ndarray:
        return np.array(
            [
                arm_specification.number_of_segments
                for arm_specification in self.arm_specifications
            ]
        )

    @property
    def total_number_of_segments(self) -> int:
        return np.sum(self.number_of_segments_per_arm)
