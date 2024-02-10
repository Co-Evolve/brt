from typing import List

from fprs.parameters import FixedParameter
from fprs.specification import MorphologySpecification, Specification


class ToyExampleJointSpecification(Specification):
    def __init__(self, range: float, stiffness: float, damping: float) -> None:
        super().__init__()
        self.stiffness = FixedParameter(value=stiffness)
        self.range = FixedParameter(value=range)
        self.damping = FixedParameter(value=damping)


class ToyExampleArmSegmentSpecification(Specification):
    def __init__(
        self,
        radius: float,
        length: float,
        joint_specification: ToyExampleJointSpecification,
    ) -> None:
        super().__init__()
        self.radius = FixedParameter(radius)
        self.length = FixedParameter(length)
        self.joint_specification = joint_specification


class ToyExampleArmSpecification(Specification):
    def __init__(
        self, segment_specifications: List[ToyExampleArmSegmentSpecification]
    ) -> None:
        super().__init__()
        self.segment_specifications = segment_specifications


class ToyExampleTorsoSpecification(Specification):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = FixedParameter(radius)


class ToyExampleActuationSpecification(Specification):
    def __init__(self, kp: float) -> None:
        super().__init__()
        self.kp = FixedParameter(kp)


class ToyExampleMorphologySpecification(MorphologySpecification):
    def __init__(
        self,
        torso_specification: ToyExampleTorsoSpecification,
        arm_specifications: List[ToyExampleArmSpecification],
        actuation_specification: ToyExampleActuationSpecification,
    ) -> None:
        super().__init__()
        self.torso_specification = torso_specification
        self.arm_specifications = arm_specifications
        self.actuation_specification = actuation_specification
