from typing import List

from fprs.parameters import FixedParameter
from fprs.specification import MorphologySpecification, Specification


class JumpingSpiderJointSpecification(Specification):
    def __init__(
            self,
            lower_range: float,
            upper_range: float,
            stiffness: float,
            damping: float,
            armature: float
    ) -> None:
        super().__init__()
        self.lower_range = FixedParameter(value=lower_range)
        self.upper_range = FixedParameter(value=upper_range)
        self.stiffness = FixedParameter(value=stiffness)
        self.damping = FixedParameter(value=damping)
        self.armature = FixedParameter(value=armature)


class JumpingSpiderLegSegmentSpecification(Specification):
    def __init__(
            self,
            ip_joint_specification: JumpingSpiderJointSpecification,
            oop_joint_specification: JumpingSpiderJointSpecification,
            torque_limit: float,
            length: float,
            radius: float,
            oop_attachment_angle: float
    ) -> None:
        super().__init__()
        self.ip_joint_specification = ip_joint_specification
        self.oop_joint_specification = oop_joint_specification
        self.torque_limit = FixedParameter(torque_limit)
        self.length = FixedParameter(length)
        self.radius = FixedParameter(radius)
        self.oop_attachment_angle = FixedParameter(oop_attachment_angle)


class JumpingSpiderLegSpecification(Specification):
    def __init__(
            self,
            in_plane_connection_angle: float,
            out_of_plane_connection_angle: float,
            segment_specifications: List[JumpingSpiderLegSegmentSpecification]
    ) -> None:
        super().__init__()
        self.in_plane_connection_angle = FixedParameter(in_plane_connection_angle)
        self.out_of_plane_connection_angle = FixedParameter(out_of_plane_connection_angle)
        self.segment_specifications = segment_specifications

    @property
    def number_of_segments(
            self
    ) -> int:
        return len(self.segment_specifications)


class JumpingSpiderAbdomenSpecification(Specification):
    def __init__(
            self,
            size_x: float,
            size_y: float,
            size_z: float,
            inset_factor: float,
            joint_specification: JumpingSpiderJointSpecification,
            torque_limit: float
    ) -> None:
        super().__init__()
        self.size_x = FixedParameter(size_x)
        self.size_y = FixedParameter(size_y)
        self.size_z = FixedParameter(size_z)
        self.inset_factor = FixedParameter(inset_factor)
        self.joint_specification = joint_specification
        self.torque_limit = FixedParameter(torque_limit)


class JumpingSpiderCephalothoraxSpecification(Specification):
    def __init__(
            self,
            size_x: float,
            size_y: float,
            size_z: float,
            leg_connector_radius: float
    ) -> None:
        super().__init__()
        self.size_x = FixedParameter(size_x)
        self.size_y = FixedParameter(size_y)
        self.size_z = FixedParameter(size_z)
        self.leg_connector_radius = FixedParameter(leg_connector_radius)


class JumpingSpiderDraglineSpecification(Specification):
    def __init__(self, stiffness: float, damping: float, enabled: bool) -> None:
        super().__init__()
        self.stiffness = FixedParameter(stiffness)
        self.damping = FixedParameter(damping)
        self.enabled = FixedParameter(enabled)


class JumpingSpiderActuationSpecification(Specification):
    def __init__(self, position_control: bool, kp: float | None = None) -> None:
        super().__init__()
        assert not (position_control and not kp), "Position control requires a gain value (kp)."
        self.position_control = FixedParameter(position_control)
        self.kp = FixedParameter(kp)


class JumpingSpiderMorphologySpecification(MorphologySpecification):
    def __init__(self, cephalothorax_specification: JumpingSpiderCephalothoraxSpecification,
                 abdomen_specification: JumpingSpiderAbdomenSpecification,
                 leg_specifications: List[JumpingSpiderLegSpecification],
                 dragline_specification: JumpingSpiderDraglineSpecification,
                 actuation_specification: JumpingSpiderActuationSpecification) -> None:
        super().__init__()
        self.cephalothorax_specification = cephalothorax_specification
        self.abdomen_specification = abdomen_specification
        self.leg_specifications = leg_specifications
        self.dragline_specification = dragline_specification
        self.actuation_specification = actuation_specification
