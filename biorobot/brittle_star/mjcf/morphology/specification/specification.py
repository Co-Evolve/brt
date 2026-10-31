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
        density: float,
        in_plane_joint_specification: BrittleStarJointSpecification,
        out_of_plane_joint_specification: BrittleStarJointSpecification,
    ) -> None:
        super().__init__()
        self.radius = FixedParameter(radius)
        self.length = FixedParameter(length)
        self.density = FixedParameter(density)
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
    def __init__(self, diameter: float, height: float, density: float) -> None:
        super().__init__()
        self.radius = FixedParameter(diameter / 2)
        self.height = FixedParameter(height)
        self.density = FixedParameter(density)


class BrittleStarActuationSpecification(Specification):
    def __init__(
        self,
        use_tendons: bool,
        use_p_control: bool,
        use_torque_control: bool,
        radius_to_strength_factor: float,
    ) -> None:
        super().__init__()
        assert (
            use_p_control + use_torque_control == 1
        ), "Only one actuation method can be used."
        assert (
            not use_tendons or use_torque_control
        ), "Only torque control is supported with tendons."
        self.use_tendons = FixedParameter(use_tendons)
        self.use_p_control = FixedParameter(use_p_control)
        self.use_torque_control = FixedParameter(use_torque_control)
        self.radius_to_strength_factor = FixedParameter(radius_to_strength_factor)


class BrittleStarSensorSpecification(Specification):
    def __init__(self, num_contact_sensors_per_segment: int) -> int:
        super().__init__()
        self.num_contact_sensors_per_segment = FixedParameter(
            num_contact_sensors_per_segment
        )


class BrittleStarMorphologySpecification(MorphologySpecification):
    def __init__(
        self,
        disk_specification: BrittleStarDiskSpecification,
        arm_specifications: List[BrittleStarArmSpecification],
        actuation_specification: BrittleStarActuationSpecification,
        sensor_specification: BrittleStarSensorSpecification,
    ) -> None:
        super(BrittleStarMorphologySpecification, self).__init__()
        self.disk_specification = disk_specification
        self.arm_specifications = arm_specifications
        self.actuation_specification = actuation_specification
        self.sensor_specification = sensor_specification

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
