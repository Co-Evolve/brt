from typing import Union

import numpy as np
from dm_control import mjcf
from moojoco.mjcf.morphology import MJCFMorphology, MJCFMorphologyPart

from biorobot.toy_example.mjcf.morphology.specification.specification import (
    ToyExampleJointSpecification,
    ToyExampleMorphologySpecification,
)
from biorobot.utils.colors import rgba_green


class MJCFToyExampleArmSegment(MJCFMorphologyPart):
    def __init__(
        self,
        parent: Union[MJCFMorphology, MJCFMorphologyPart],
        name: str,
        pos: np.array,
        euler: np.array,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> ToyExampleMorphologySpecification:
        return super().morphology_specification

    def _build(self, arm_index: int, segment_index: int) -> None:
        self._arm_index = arm_index
        self._segment_index = segment_index
        self._arm_specification = self.morphology_specification.arm_specifications[
            self._arm_index
        ]
        self._segment_specification = self._arm_specification.segment_specifications[
            segment_index
        ]
        self._actuator_specification = (
            self.morphology_specification.actuation_specification
        )

        self._build_capsule()
        self._configure_joints()
        self._configure_actuators()
        self._configure_sensors()

    @property
    def center_of_capsule(self) -> np.ndarray:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value
        x_offset = radius + length / 2
        return np.array([x_offset, 0, 0])

    def _build_capsule(self) -> None:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value

        self.capsule = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_capsule",
            type="capsule",
            pos=self.center_of_capsule,
            euler=[0, np.pi / 2, 0],
            size=[radius, length / 2],
            rgba=rgba_green,
        )

    def _configure_joint(
        self,
        name: str,
        axis: np.ndarray,
        joint_specification: ToyExampleJointSpecification,
    ) -> mjcf.Element:
        joint = self.mjcf_body.add(
            "joint",
            name=name,
            type="hinge",
            limited=True,
            range=[-joint_specification.range.value, joint_specification.range.value],
            axis=axis,
            stiffness=joint_specification.stiffness.value,
            damping=joint_specification.damping.value,
        )
        return joint

    def _configure_joints(self) -> None:
        self.in_plane_joint = self._configure_joint(
            name=f"{self.base_name}_in_plane_joint",
            axis=[0, 0, 1],
            joint_specification=self._segment_specification.joint_specification,
        )
        self.out_of_plane_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_joint",
            axis=[0, 1, 0],
            joint_specification=self._segment_specification.joint_specification,
        )

    def _configure_actuator(self, joint: mjcf.Element) -> None:
        self.mjcf_model.actuator.add(
            "position",
            name=f"{joint.name}_p_control",
            joint=joint,
            kp=self._actuator_specification.kp.value,
            ctrllimited=True,
            ctrlrange=joint.range,
        )

    def _configure_actuators(self) -> None:
        self._configure_actuator(joint=self.in_plane_joint)
        self._configure_actuator(joint=self.out_of_plane_joint)

    def _configure_joint_sensor(self, joint: mjcf.Element) -> None:
        self.mjcf_model.sensor.add(
            "jointpos", joint=joint, name=f"{joint.name}" f"_jointpos_sensor"
        )
        self.mjcf_model.sensor.add(
            "jointvel", joint=joint, name=f"{joint.name}" f"_jointvel_sensor"
        )

    def _configure_sensors(self) -> None:
        self._configure_joint_sensor(self.in_plane_joint)
        self._configure_joint_sensor(self.out_of_plane_joint)
