from typing import Union

import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from biorobot.jumping_spider.mjcf.morphology.specification.specification import JumpingSpiderMorphologySpecification, \
    JumpingSpiderJointSpecification
from biorobot.utils import colors


class MJCFJumpingSpiderLegSegment(MJCFMorphologyPart):
    def __init__(
            self,
            parent: Union[MJCFMorphology, MJCFMorphologyPart],
            name: str,
            pos: np.array,
            euler: np.array,
            *args,
            **kwargs
    ) -> None:
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(
            self
    ) -> JumpingSpiderMorphologySpecification:
        return super().morphology_specification

    def _build(
            self,
            leg_index: int,
            segment_index: int
    ) -> None:
        self._leg_index = leg_index
        self._segment_index = segment_index

        self._leg_specification = self.morphology_specification.leg_specifications[self._leg_index]
        self._segment_specification = self._leg_specification.segment_specifications[self._segment_index]

        self._build_capsule()
        self._configure_joints()
        self._configure_actuators()
        self._configure_sensors()

    def _build_capsule(
            self
    ) -> None:

        self._capsule = self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_capsule",
            type="capsule",
            pos=self.center_of_capsule,
            euler=[0, np.pi / 2, 0],
            size=[self._segment_specification.radius.value, self._segment_specification.length.value / 2],
            friction=[2, 0.1, 0.1],
            rgba=colors.rgba_green,
        )

    @property
    def center_of_capsule(
            self
    ) -> np.ndarray:
        radius = self._segment_specification.radius.value
        length = self._segment_specification.length.value

        x_offset = radius + length / 2
        return np.array([x_offset, 0, 0])

    def _configure_joint(
            self,
            name: str,
            axis: np.ndarray,
            joint_specification: JumpingSpiderJointSpecification
    ) -> _ElementImpl:
        joint = self.mjcf_body.add(
            "joint",
            name=name,
            type="hinge",
            limited=True,
            range=[joint_specification.lower_range.value, joint_specification.upper_range.value],
            axis=axis,
            stiffness=joint_specification.stiffness.value,
            damping=joint_specification.damping.value
        )
        return joint

    def _configure_joints(
            self
    ) -> None:
        if (self._segment_specification.ip_joint_specification.lower_range.value == 0 and
                self._segment_specification.ip_joint_specification.upper_range.value == 0):
            self._ip_joint = None
        else:
            self._ip_joint = self._configure_joint(
                name=f"{self.base_name}_in_plane_joint",
                axis=[0, 0, 1],
                joint_specification=self._segment_specification.ip_joint_specification
            )
        self._oop_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_joint",
            axis=[0, 1, 0],
            joint_specification=self._segment_specification.oop_joint_specification
        )

    def _configure_torque_control_actuator(
            self,
            joint: _ElementImpl,
    ) -> _ElementImpl:
        if joint is not None:
            return self.mjcf_model.actuator.add(
                'motor',
                name=f"{joint.name}_torque_control",
                joint=joint,
                ctrllimited=True,
                ctrlrange=[-self._segment_specification.torque_limit.value,
                           self._segment_specification.torque_limit.value],
                forcelimited=True,
                forcerange=[-self._segment_specification.torque_limit.value,
                            self._segment_specification.torque_limit.value]
            )

    def _configure_position_control_actuator(
            self,
            joint: _ElementImpl,
    ) -> _ElementImpl:
        if joint is not None:
            return self.mjcf_model.actuator.add(
                'position',
                name=f"{joint.name}_position_control",
                joint=joint,
                ctrllimited=True,
                ctrlrange=joint.range,
                kp=500,
                forcelimited=True,
                forcerange=[-self._segment_specification.torque_limit.value,
                            self._segment_specification.torque_limit.value]
            )

    def _configure_actuators(
            self
    ) -> None:
        if self.morphology_specification.actuation_specification.position_control.value:
            self._ip_actuator = self._configure_position_control_actuator(joint=self._ip_joint)
            self._oop_actuator = self._configure_position_control_actuator(joint=self._oop_joint)
        else:
            self._ip_actuator = self._configure_torque_control_actuator(joint=self._ip_joint)
            self._oop_actuator = self._configure_torque_control_actuator(joint=self._oop_joint)

    def _configure_touch_sensors(
            self
    ) -> None:
        if self._segment_index == 6:
            touch_site = self.mjcf_body.add(
                "site",
                name=f"{self.base_name}_touch_site",
                type="capsule",
                pos=self._capsule.pos,
                euler=self._capsule.euler,
                rgba=np.zeros(4),
                size=self._capsule.size
            )
            self.mjcf_model.sensor.add(
                "touch", name=f"{self.base_name}_touch_sensor", site=touch_site
            )

    def _configure_joint_sensors(self, joint: _ElementImpl) -> None:
        if joint is not None:
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
        self._configure_joint_sensors(joint=self._ip_joint)
        self._configure_joint_sensors(joint=self._oop_joint)

    def _configure_actuator_sensors(self, actuator: _ElementImpl) -> None:
        if actuator is not None:
            self.mjcf_model.sensor.add(
                "actuatorfrc",
                actuator=actuator,
                name=f"{actuator.name}_actuatorfrc_sensor",
            )

    def _configure_actuators_sensors(self) -> None:
        self._configure_actuator_sensors(actuator=self._ip_actuator)
        self._configure_actuator_sensors(actuator=self._oop_actuator)

    def _configure_sensors(
            self
    ) -> None:
        self._configure_touch_sensors()
        self._configure_joints_sensors()
        self._configure_actuators_sensors()
