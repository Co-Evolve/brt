from typing import Union

import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphology, MJCFMorphologyPart

from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarJointSpecification,
    BrittleStarMorphologySpecification,
)
from biorobot.utils import colors


class MJCFBrittleStarArmSegment(MJCFMorphologyPart):
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
    def morphology_specification(self) -> BrittleStarMorphologySpecification:
        return super().morphology_specification

    def _build(self, arm_index: int, segment_index: int) -> None:
        self._arm_index = arm_index
        self._segment_index = segment_index

        self._arm_specification = self.morphology_specification.arm_specifications[
            self._arm_index
        ]
        self._segment_specification = self._arm_specification.segment_specifications[
            self._segment_index
        ]

        self._build_capsule()
        self._build_connector()
        self._configure_joints()
        self._configure_actuators()
        self._configure_sensors()

    def _build_capsule(self) -> None:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value

        self._capsule = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_capsule",
            type="capsule",
            pos=self.center_of_capsule,
            euler=[0, np.pi / 2, 0],
            size=[radius, length / 2],
            rgba=colors.rgba_green,
        )

    def _build_connector(self) -> None:
        radius = self._segment_specification.radius.value
        self._connector = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_connector",
            type="sphere",
            pos=np.zeros(3),
            size=[0.5 * radius],
            rgba=colors.rgba_gray,
            contype=0,
            conaffinity=0,
        )

    @property
    def center_of_capsule(self) -> np.ndarray:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value
        x_offset = radius + length / 2
        return np.array([x_offset, 0, 0])

    def _configure_joint(
        self,
        name: str,
        axis: np.ndarray,
        joint_specification: BrittleStarJointSpecification,
    ) -> _ElementImpl:
        joint = self.mjcf_body.add(
            "joint",
            name=name,
            type="hinge",
            limited=True,
            range=[-joint_specification.range.value, joint_specification.range.value],
            axis=axis,
            stiffness=joint_specification.stiffness.value,
            damping=joint_specification.damping.value,
            armature=joint_specification.armature.value,
        )
        return joint

    def _configure_joints(self) -> None:
        self._in_plane_joint = self._configure_joint(
            name=f"{self.base_name}_in_plane_joint",
            axis=[0, 0, 1],
            joint_specification=self._segment_specification.in_plane_joint_specification,
        )
        self._out_of_plane_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_joint",
            axis=[0, -1, 0],
            joint_specification=self._segment_specification.out_of_plane_joint_specification,
        )

    def _is_first_segment(self) -> bool:
        return self._segment_index == 0

    def _is_last_segment(self) -> bool:
        number_of_segments = len(self._arm_specification.segment_specifications)
        return self._segment_index == number_of_segments - 1

    def _get_strength(self, joint: _ElementImpl) -> float:
        strength = (
            self._segment_specification.radius.value
            * self.morphology_specification.actuation_specification.radius_to_strength_factor.value
        )
        return strength

    def _configure_p_control_actuator(self, joint: _ElementImpl) -> _ElementImpl:
        return self.mjcf_model.actuator.add(
            "position",
            name=f"{joint.name}_p_control",
            joint=joint,
            kp=50,
            ctrllimited=True,
            ctrlrange=joint.range,
            forcelimited=True,
            forcerange=[-self._get_strength(joint), self._get_strength(joint)],
        )

    def _configure_p_control_actuators(self) -> None:
        if self.morphology_specification.actuation_specification.use_p_control.value:
            self._in_plane_actuator = self._configure_p_control_actuator(
                self._in_plane_joint
            )
            self._out_of_plane_actuator = self._configure_p_control_actuator(
                self._out_of_plane_joint
            )

    def _configure_torque_control_actuator(self, joint: _ElementImpl) -> _ElementImpl:
        return self.mjcf_model.actuator.add(
            "motor",
            name=f"{joint.name}_torque_control",
            joint=joint,
            ctrllimited=True,
            ctrlrange=[-self._get_strength(joint), self._get_strength(joint)],
            forcelimited=True,
            forcerange=[-self._get_strength(joint), self._get_strength(joint)],
        )

    def _configure_torque_control_actuators(self) -> None:
        if (
            self.morphology_specification.actuation_specification.use_torque_control.value
        ):
            self._in_plane_actuator = self._configure_torque_control_actuator(
                self._in_plane_joint
            )
            self._out_of_plane_actuator = self._configure_torque_control_actuator(
                self._out_of_plane_joint
            )

    def _configure_actuators(self) -> None:
        self._configure_p_control_actuators()
        self._configure_torque_control_actuators()

    def _configure_position_sensor(self) -> None:
        self.mjcf_model.sensor.add(
            "framepos",
            name=f"{self.base_name}_position_sensor",
            objtype="geom",
            objname=self._capsule.name,
        )

    def _configure_joint_sensors(self, joint: _ElementImpl) -> None:
        self.mjcf_model.sensor.add(
            "jointpos", joint=joint, name=f"{joint.name}_jointpos_sensor"
        )
        self.mjcf_model.sensor.add(
            "jointvel", joint=joint, name=f"{joint.name}_jointvel_sensor"
        )
        self.mjcf_model.sensor.add(
            "jointactuatorfrc", joint=joint, name=f"{joint.name}_actuatorfrc_sensor"
        )

    def _configure_joints_sensors(self) -> None:
        self._configure_joint_sensors(joint=self._in_plane_joint)
        self._configure_joint_sensors(joint=self._out_of_plane_joint)

    def _configure_actuator_sensors(self) -> None:
        self.mjcf_model.sensor.add(
            "actuatorfrc",
            actuator=self._in_plane_actuator,
            name=f"{self._in_plane_actuator.name}_actuatorfrc_sensor",
        )
        self.mjcf_model.sensor.add(
            "actuatorfrc",
            actuator=self._out_of_plane_actuator,
            name=f"{self._out_of_plane_actuator.name}_actuatorfrc_sensor",
        )

    def _configure_sensors(self) -> None:
        self._configure_position_sensor()
        self._configure_joints_sensors()
        self._configure_actuator_sensors()
