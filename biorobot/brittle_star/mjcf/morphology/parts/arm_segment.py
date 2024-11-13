from __future__ import annotations

from typing import Union, List

import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphology, MJCFMorphologyPart

from biorobot.brittle_star.mjcf.morphology.parts.disk import MJCFBrittleStarDisk
from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarJointSpecification,
    BrittleStarMorphologySpecification,
)
from biorobot.utils import colors
from biorobot.utils.colors import rgba_red, rgba_tendon_contracted, rgba_tendon_relaxed


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
    def parent(self) -> Union[MJCFBrittleStarDisk, MJCFBrittleStarArmSegment]:
        return super().parent

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
        self._configure_tendons()
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
        self._joints = [
            self._configure_joint(
                name=f"{self.base_name}_in_plane_joint",
                axis=[0, 0, 1],
                joint_specification=self._segment_specification.in_plane_joint_specification,
            ),
            self._configure_joint(
                name=f"{self.base_name}_out_of_plane_joint",
                axis=[0, -1, 0],
                joint_specification=self._segment_specification.out_of_plane_joint_specification,
            ),
        ]

    def _configure_tendon_attachment_points(self) -> None:
        angles = np.linspace(np.pi / 4, 7 * np.pi / 4, 4)
        self._proximal_taps = []
        self.distal_taps = []
        for i, angle in enumerate(angles):
            # proximal
            pos = (
                0.8
                * self._segment_specification.radius.value
                * np.array([0, np.cos(angle), np.sin(angle)])
            )
            pos[0] = self._segment_specification.radius.value
            self._proximal_taps.append(
                self.mjcf_body.add(
                    "site",
                    name=f"{self.base_name}_proximal_tap_{i}",
                    type="sphere",
                    rgba=rgba_red,
                    pos=pos,
                    size=[0.001],
                )
            )

            # distal
            pos[0] = (
                self._segment_specification.radius.value
                + self._segment_specification.length.value
            )
            self.distal_taps.append(
                self.mjcf_body.add(
                    "site",
                    name=f"{self.base_name}_distal_tap_{i}",
                    type="sphere",
                    rgba=rgba_red,
                    pos=pos,
                    size=[0.001],
                )
            )

    def _build_tendons(self) -> None:
        if self._segment_index == 0:
            parent: MJCFBrittleStarDisk = self.parent.parent
            distal_taps = parent.distal_taps[self._arm_index]
        else:
            distal_taps = self.parent.distal_taps

        self._tendons = []
        for tendon_index, (parent_tap, segment_tap) in enumerate(
            zip(distal_taps, self._proximal_taps)
        ):
            tendon = self.mjcf_model.tendon.add(
                "spatial",
                name=f"{self.base_name}_tendon_{tendon_index}",
                rgba=rgba_tendon_relaxed,
                width=self._segment_specification.radius.value * 0.1,
            )
            tendon.add("site", site=parent_tap)
            tendon.add("site", site=segment_tap)
            self._tendons.append(tendon)

    def _configure_tendons(self) -> None:
        if self.morphology_specification.actuation_specification.use_tendons.value:
            self._configure_tendon_attachment_points()
            self._build_tendons()

    def _is_first_segment(self) -> bool:
        return self._segment_index == 0

    def _is_last_segment(self) -> bool:
        number_of_segments = len(self._arm_specification.segment_specifications)
        return self._segment_index == number_of_segments - 1

    @property
    def _actuator_strength(self) -> float:
        strength = (
            self._segment_specification.radius.value
            * self.morphology_specification.actuation_specification.radius_to_strength_factor.value
        )
        return strength

    @property
    def _transmissions(self) -> List[_ElementImpl]:
        if self.morphology_specification.actuation_specification.use_tendons.value:
            return self._tendons
        else:
            return self._joints

    def _configure_p_control_actuator(self, transmission: _ElementImpl) -> _ElementImpl:
        actuator_attributes = {
            "name": f"{transmission.name}_p_control",
            "kp": 50,
            "ctrllimited": True,
            "ctrlrange": transmission.range,
            "forcelimited": True,
            "forcerange": [-self._actuator_strength, self._actuator_strength],
            "joint": transmission,
        }

        return self.mjcf_model.actuator.add("position", **actuator_attributes)

    def _configure_p_control_actuators(self) -> None:
        if self.morphology_specification.actuation_specification.use_p_control.value:
            self._actuators = [
                self._configure_p_control_actuator(transmission)
                for transmission in self._transmissions
            ]

    def _configure_torque_control_actuator(
        self, transmission: _ElementImpl
    ) -> _ElementImpl:
        actuator_attributes = {
            "name": f"{transmission.name}_torque_control",
            "ctrllimited": True,
            "forcelimited": True,
            "ctrlrange": [-self._actuator_strength, self._actuator_strength],
            "forcerange": [-self._actuator_strength, self._actuator_strength],
        }

        if self.morphology_specification.actuation_specification.use_tendons.value:
            actuator_attributes["tendon"] = transmission
            actuator_attributes["ctrlrange"] = [-self._actuator_strength, 0]
            gear = 15
            actuator_attributes["gear"] = [gear]
            actuator_attributes["forcerange"] = [-self._actuator_strength * gear, 0]
        else:
            actuator_attributes["joint"] = transmission
            actuator_attributes["ctrlrange"] = [
                -self._actuator_strength,
                self._actuator_strength,
            ]
            actuator_attributes["forcerange"] = [
                -self._actuator_strength,
                self._actuator_strength,
            ]

        return self.mjcf_model.actuator.add("motor", **actuator_attributes)

    def _configure_torque_control_actuators(self) -> None:
        if (
            self.morphology_specification.actuation_specification.use_torque_control.value
        ):
            self._actuators = [
                self._configure_torque_control_actuator(transmission)
                for transmission in self._transmissions
            ]

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

    def _configure_joints_sensors(self) -> None:
        for joint in self._joints:
            self.mjcf_model.sensor.add(
                "jointpos", joint=joint, name=f"{joint.name}_jointpos_sensor"
            )
            self.mjcf_model.sensor.add(
                "jointvel", joint=joint, name=f"{joint.name}_jointvel_sensor"
            )
            self.mjcf_model.sensor.add(
                "jointactuatorfrc", joint=joint, name=f"{joint.name}_actuatorfrc_sensor"
            )

    def _configure_actuator_sensors(self) -> None:
        for actuator in self._actuators:
            self.mjcf_model.sensor.add(
                "actuatorfrc",
                actuator=actuator,
                name=f"{actuator.name}_actuatorfrc_sensor",
            )

    def _configure_tendon_sensors(self) -> None:
        if self.morphology_specification.actuation_specification.use_tendons.value:
            for tendon in self._tendons:
                self.mjcf_model.sensor.add(
                    "tendonpos", name=f"{tendon.name}_tendonpos_sensor", tendon=tendon
                )
                self.mjcf_model.sensor.add(
                    "tendonvel", name=f"{tendon.name}_tendonvel_sensor", tendon=tendon
                )

    def _configure_sensors(self) -> None:
        self._configure_position_sensor()
        self._configure_joints_sensors()
        self._configure_actuator_sensors()
        self._configure_tendon_sensors()
