from typing import Union

import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from biorobot.utils import colors
from ..specification.specification import JumpingSpiderMorphologySpecification, JumpingSpiderJointSpecification


class MJCFJumpingSpiderAbdomen(MJCFMorphologyPart):
    def __init__(self, parent: Union[MJCFMorphology, MJCFMorphologyPart], name: str, pos: np.array, euler: np.array,
                 *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> JumpingSpiderMorphologySpecification:
        return super().morphology_specification

    def _build(self) -> None:
        self._abdomen_specification = self.morphology_specification.abdomen_specification
        self._build_ellipsoid()
        self._configure_joints()
        self._configure_actuators()
        self._configure_dragline_body_site()
        self._configure_sensors()

    def _build_ellipsoid(self) -> None:
        size_x = self._abdomen_specification.size_x.value
        size_y = self._abdomen_specification.size_y.value
        size_z = self._abdomen_specification.size_z.value
        inset_factor = 1 - self._abdomen_specification.inset_factor.value

        self._ellipsoid = self.mjcf_body.add("geom",
                                             name=f"{self.base_name}_ellipsoid",
                                             type=f"ellipsoid",
                                             pos=np.array([-size_x * inset_factor, 0, 0]),
                                             euler=np.zeros(3),
                                             size=[size_x, size_y, size_z],
                                             rgba=colors.rgba_green,
                                             )

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
            range=[-joint_specification.lower_range.value, joint_specification.upper_range.value],
            axis=axis,
            stiffness=joint_specification.stiffness.value,
            damping=joint_specification.damping.value
        )
        return joint

    def _configure_joints(
            self
    ) -> None:
        self._ip_joint = self._configure_joint(
            name=f"{self.base_name}_in_plane_joint",
            axis=[0, 0, 1],
            joint_specification=self._abdomen_specification.joint_specification
        )
        self._oop_joint = self._configure_joint(
            name=f"{self.base_name}_out_of_plane_joint",
            axis=[0, 1, 0],
            joint_specification=self._abdomen_specification.joint_specification
        )

    def _configure_p_control_actuator(
            self,
            joint: _ElementImpl
    ) -> _ElementImpl:
        return self.mjcf_model.actuator.add(
            'position',
            name=f"{joint.name}_p_control",
            joint=joint,
            kp=50,
            ctrllimited=True,
            ctrlrange=joint.range,
            forcelimited=True,
            forcerange=[-self._abdomen_specification.torque_limit.value, self._abdomen_specification.torque_limit.value]
        )

    def _configure_actuators(
            self
    ) -> None:
        self._ip_actuator = self._configure_p_control_actuator(joint=self._ip_joint)
        self._oop_actuator = self._configure_p_control_actuator(joint=self._oop_joint)

    def _configure_dragline_body_site(self) -> None:
        ip_angle = np.pi
        oop_angle = -np.pi / 6
        pos_x = self._abdomen_specification.size_x.value * np.cos(oop_angle) * np.cos(ip_angle)
        pos_y = self._abdomen_specification.size_y.value * np.cos(oop_angle) * np.sin(ip_angle)
        pos_z = self._abdomen_specification.size_z.value * np.sin(oop_angle)
        pos = np.array([pos_x, pos_y, pos_z]) + self._ellipsoid.pos

        self.dragline_body_site = self.mjcf_body.add("site",
                                                     type="sphere",
                                                     name=f"{self.base_name}_dragline_start_site",
                                                     pos=pos,
                                                     size=[0.01],
                                                     rgba=colors.rgba_red)

    def _configure_joint_sensors(self, joint: _ElementImpl) -> None:
        self.mjcf_model.sensor.add("jointpos", joint=joint, name=f"{joint.name}_jointpos_sensor")
        self.mjcf_model.sensor.add("jointvel", joint=joint, name=f"{joint.name}_jointvel_sensor")
        self.mjcf_model.sensor.add(
            "jointactuatorfrc", joint=joint, name=f"{joint.name}_actuatorfrc_sensor"
        )

    def _configure_joints_sensors(self) -> None:
        self._configure_joint_sensors(self._ip_joint)
        self._configure_joint_sensors(self._oop_joint)

    def _configure_actuator_sensors(self, actuator: _ElementImpl) -> None:
        self.mjcf_model.sensor.add(
            "actuatorfrc",
            actuator=actuator,
            name=f"{actuator.name}_actuatorfrc_sensor",
        )

    def _configure_actuators_sensors(self) -> None:
        self._configure_actuator_sensors(actuator=self._ip_actuator)
        self._configure_actuator_sensors(actuator=self._oop_actuator)

    def _configure_sensors(self) -> None:
        self._configure_joints_sensors()
        self._configure_actuators_sensors()
