"""The brittle star morphology, built with ``mujoco.MjSpec``.

Layout (names are prefixed with ``BrittleStarMorphology/`` once attached to an arena)::

    central_disk                              free body (pentagon visual, cylinder collider)
    └── arm_{a}                               fixed frame on the disk rim, rotated by a * 72 degrees
        └── arm_{a}_segment_0                 capsule along +x with an in-plane and an out-of-plane hinge
            └── arm_{a}_segment_1 ...

Each segment carries joint position/velocity/actuator-force sensors, an actuator per transmission (joint or tendon),
a capsule position sensor, and touch sensors on (box or capsule) sites around the capsule.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import mujoco
import numpy as np

from biorobot.brittle_star.specification import (
    BrittleStarArmSegmentSpecification,
    BrittleStarJointSpecification,
    BrittleStarMorphologySpecification,
    default_brittle_star_morphology_specification,
)
from biorobot.core import Morphology, euler_to_quat
from biorobot.utils import colors

#: Arms are placed at fixed angles, so removing arms (0 segments) does not rotate the others.
ARM_ANGLES = np.linspace(0, 2 * np.pi, 6)[:-1]
#: Angles (around the arm axis) of the four tendon attachment points of a segment.
TENDON_TAP_ANGLES = np.linspace(np.pi / 4, 7 * np.pi / 4, 4)
#: Gear of tendon motors: control is in "segment strength" units, the tendon force is ``gear`` times larger.
TENDON_GEAR = 15
P_CONTROL_KP = 20
TAP_SIZE = 0.001

_JOINT = mujoco.mjtObj.mjOBJ_JOINT
_TRUE = mujoco.mjtLimited.mjLIMITED_TRUE


class BrittleStarMorphology(Morphology[BrittleStarMorphologySpecification]):
    """A brittle star: a central disk with up to five segmented arms."""

    def __init__(
        self, specification: Optional[BrittleStarMorphologySpecification] = None
    ) -> None:
        if specification is None:
            specification = default_brittle_star_morphology_specification(
                use_p_control=True
            )
        super().__init__(specification=specification, name="BrittleStarMorphology")

    @property
    def _actuation(self):
        return self.specification.actuation_specification

    def _build(self) -> None:
        # Contacts are only generated between the morphology (contype 1) and the arena (conaffinity 1), never between
        # body parts of the brittle star itself.
        self.spec.default.geom.condim = 4
        self.spec.default.geom.contype = 1
        self.spec.default.geom.conaffinity = 0

        self._disk = self.spec.worldbody.add_body(name="central_disk")
        self._build_disk()
        self._disk_taps = self._build_disk_tendon_taps()
        self._add_frame_sensors()
        for arm_index in range(self.specification.number_of_arms):
            self._build_arm(arm_index)
        self._disk.add_camera(
            name="side_camera",
            pos=[0.0, -2.0, 2.5],
            quat=euler_to_quat(np.array([40 / 180 * np.pi, 0, 0])),
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Disk
    # ------------------------------------------------------------------------------------------------------------------
    def _build_disk(self) -> None:
        disk = self.specification.disk_specification
        radius, height = disk.radius, disk.height

        # The pentagon is purely visual (five massless boxes); a cylinder handles collisions and mass.
        box_y_size = radius * np.sin(np.pi / 5)
        box_x_size = radius * np.cos(np.pi / 5) / 2
        side_angles = ARM_ANGLES + ARM_ANGLES[1] / 2
        for i, angle in enumerate(side_angles):
            self._disk.add_geom(
                name=f"central_disk_pentagon_side_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=box_x_size * np.array([np.cos(angle), np.sin(angle), 0.0]),
                quat=euler_to_quat(np.array([0, 0, angle])),
                size=[box_x_size, box_y_size, height],
                rgba=colors.rgba_green,
                contype=0,
                conaffinity=0,
                mass=0,
            )
        self._disk.add_geom(
            name="central_disk_pentagon_collider",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[radius, height, 0],
            rgba=[0, 0, 0, 0],
            contype=1,
            conaffinity=0,
            density=disk.density,
        )

        connector_length = radius * 0.2
        for i, angle in enumerate(ARM_ANGLES):
            self._disk.add_geom(
                name=f"central_disk_arm_connector_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(radius - connector_length)
                * np.array([np.cos(angle), np.sin(angle), 0.0]),
                quat=euler_to_quat(np.array([0.0, 0.0, angle])),
                size=[connector_length, height, height],
                rgba=colors.rgba_green,
                contype=0,
                conaffinity=0,
            )

    def _build_disk_tendon_taps(self) -> dict:
        """Disk-side attachment sites of the first segment's tendons, per (non-empty) arm."""
        taps = {}
        if not self._actuation.use_tendons:
            return taps
        center = np.array([self.specification.disk_specification.radius, 0, 0])
        for arm_index, arm_angle in enumerate(ARM_ANGLES):
            if arm_index >= self.specification.number_of_arms:
                break
            arm = self.specification.arm_specifications[arm_index]
            if arm.number_of_segments == 0:
                continue
            base_radius = arm.segment_specifications[0].radius
            rotation = np.array(
                [
                    [np.cos(arm_angle), -np.sin(arm_angle), 0],
                    [np.sin(arm_angle), np.cos(arm_angle), 0],
                    [0, 0, 1],
                ]
            )
            taps[arm_index] = [
                self._disk.add_site(
                    name=f"central_disk_arm_{arm_index}_tap_{tap_index}",
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    rgba=colors.rgba_red,
                    pos=rotation
                    @ (
                        0.8 * center
                        + 0.8
                        * base_radius
                        * np.array([0, np.cos(angle), np.sin(angle)])
                    ),
                    size=[TAP_SIZE, 0, 0],
                ).name
                for tap_index, angle in enumerate(TENDON_TAP_ANGLES)
            ]
        return taps

    def _add_frame_sensors(self) -> None:
        for sensor_type, suffix in (
            (mujoco.mjtSensor.mjSENS_FRAMEPOS, "framepos"),
            (mujoco.mjtSensor.mjSENS_FRAMEQUAT, "framequat"),
            (mujoco.mjtSensor.mjSENS_FRAMELINVEL, "framelinvel"),
            (mujoco.mjtSensor.mjSENS_FRAMEANGVEL, "frameangvel"),
        ):
            self.spec.add_sensor(
                name=f"central_disk_{suffix}_sensor",
                type=sensor_type,
                objtype=mujoco.mjtObj.mjOBJ_XBODY,
                objname="central_disk",
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Arms
    # ------------------------------------------------------------------------------------------------------------------
    def _build_arm(self, arm_index: int) -> None:
        angle = ARM_ANGLES[arm_index]
        disk_radius = self.specification.disk_specification.radius
        parent = self._disk.add_body(
            name=f"arm_{arm_index}",
            pos=disk_radius * np.array([np.cos(angle), np.sin(angle), 0]),
            quat=euler_to_quat(np.array([0, 0, angle])),
        )
        arm = self.specification.arm_specifications[arm_index]
        parent_distal_taps = self._disk_taps.get(arm_index)
        position = np.zeros(3)
        for segment_index, segment in enumerate(arm.segment_specifications):
            parent, parent_distal_taps = self._build_segment(
                parent=parent,
                name=f"arm_{arm_index}_segment_{segment_index}",
                position=position,
                segment=segment,
                parent_distal_taps=parent_distal_taps,
            )
            # The next segment starts where this capsule ends.
            position = 2 * _capsule_center(segment)

    def _build_segment(
        self,
        parent: mujoco.MjsBody,
        name: str,
        position: np.ndarray,
        segment: BrittleStarArmSegmentSpecification,
        parent_distal_taps: Optional[List[str]],
    ):
        body = parent.add_body(name=name, pos=position)
        radius, length = segment.radius, segment.length
        capsule_center = _capsule_center(segment)
        capsule_quat = euler_to_quat(np.array([0, np.pi / 2, 0]))

        capsule = body.add_geom(
            name=f"{name}_capsule",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            pos=capsule_center,
            quat=capsule_quat,
            size=[radius, length / 2, 0],
            rgba=colors.rgba_green,
            density=segment.density,
        )
        body.add_geom(
            name=f"{name}_connector",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.5 * radius, 0, 0],
            rgba=colors.rgba_gray,
            contype=0,
            conaffinity=0,
            mass=0,
        )

        joints = [
            self._add_joint(
                body,
                f"{name}_in_plane_joint",
                [0, 0, 1],
                segment.in_plane_joint_specification,
            ),
            self._add_joint(
                body,
                f"{name}_out_of_plane_joint",
                [0, -1, 0],
                segment.out_of_plane_joint_specification,
            ),
        ]

        distal_taps = None
        tendons: List[str] = []
        if self._actuation.use_tendons:
            proximal_taps, distal_taps, connector_taps = self._add_segment_tendon_taps(
                body, name, segment
            )
            for i, (parent_tap, connector_tap, segment_tap) in enumerate(
                zip(parent_distal_taps, connector_taps, proximal_taps)
            ):
                tendon = self.spec.add_tendon(
                    name=f"{name}_tendon_{i}",
                    rgba=colors.rgba_tendon_relaxed,
                    width=radius * 0.1,
                )
                for site in (parent_tap, connector_tap, segment_tap):
                    tendon.wrap_site(site)
                tendons.append(tendon.name)

        strength = radius * self._actuation.radius_to_strength_factor
        if self._actuation.use_tendons:
            actuators = [self._add_tendon_motor(tendon, strength) for tendon in tendons]
        elif self._actuation.use_p_control:
            actuators = [self._add_position_servo(joint, strength) for joint in joints]
        else:
            actuators = [self._add_joint_motor(joint, strength) for joint in joints]

        self._add_segment_sensors(
            body, name, capsule, segment, joints, actuators, tendons
        )
        return body, distal_taps

    def _add_joint(
        self,
        body: mujoco.MjsBody,
        name: str,
        axis: Sequence[float],
        specification: BrittleStarJointSpecification,
    ) -> mujoco.MjsJoint:
        return body.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            limited=_TRUE,
            range=[-specification.range, specification.range],
            axis=axis,
            stiffness=specification.stiffness,
            damping=specification.damping,
            armature=specification.armature,
        )

    def _add_segment_tendon_taps(
        self,
        body: mujoco.MjsBody,
        name: str,
        segment: BrittleStarArmSegmentSpecification,
    ):
        """Four tendons run along the segment: from the previous segment's distal tap, past a tap on this segment's
        connector (the vertebra), to a tap on the proximal end of this segment's capsule.
        """
        proximal, distal, connector = [], [], []
        radius, length = segment.radius, segment.length
        for i, angle in enumerate(TENDON_TAP_ANGLES):
            offset = 0.8 * radius * np.array([0, np.cos(angle), np.sin(angle)])
            for taps, kind, x in (
                (proximal, "proximal", radius),
                (distal, "distal", radius + length),
                (connector, "connector", 0.0),
            ):
                taps.append(
                    body.add_site(
                        name=f"{name}_{kind}_tap_{i}",
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        rgba=colors.rgba_red,
                        pos=offset + np.array([x, 0, 0]),
                        size=[TAP_SIZE, 0, 0],
                    ).name
                )
        return proximal, distal, connector

    # ------------------------------------------------------------------------------------------------------------------
    # Actuators
    # ------------------------------------------------------------------------------------------------------------------
    def _add_position_servo(self, joint: mujoco.MjsJoint, strength: float) -> str:
        actuator = self.spec.add_actuator(
            name=f"{joint.name}_p_control",
            target=joint.name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=_TRUE,
            ctrlrange=joint.range,
            forcelimited=_TRUE,
            forcerange=[-strength, strength],
        )
        actuator.set_to_position(kp=P_CONTROL_KP)
        return actuator.name

    def _add_joint_motor(self, joint: mujoco.MjsJoint, strength: float) -> str:
        actuator = self.spec.add_actuator(
            name=f"{joint.name}_torque_control",
            target=joint.name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=_TRUE,
            ctrlrange=[-strength, strength],
            forcelimited=_TRUE,
            forcerange=[-strength, strength],
        )
        actuator.set_to_motor()
        return actuator.name

    def _add_tendon_motor(self, tendon: str, strength: float) -> str:
        # Tendons can only pull: negative controls contract the tendon.
        actuator = self.spec.add_actuator(
            name=f"{tendon}_torque_control",
            target=tendon,
            trntype=mujoco.mjtTrn.mjTRN_TENDON,
            ctrllimited=_TRUE,
            ctrlrange=[-strength, 0],
            forcelimited=_TRUE,
            forcerange=[-strength * TENDON_GEAR, 0],
            gear=[TENDON_GEAR, 0, 0, 0, 0, 0],
        )
        actuator.set_to_motor()
        return actuator.name

    # ------------------------------------------------------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------------------------------------------------------
    def _add_segment_sensors(
        self,
        body: mujoco.MjsBody,
        name: str,
        capsule: mujoco.MjsGeom,
        segment: BrittleStarArmSegmentSpecification,
        joints: List[mujoco.MjsJoint],
        actuators: List[str],
        tendons: List[str],
    ) -> None:
        sensor = self.spec.add_sensor
        sensor(
            name=f"{name}_position_sensor",
            type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
            objtype=mujoco.mjtObj.mjOBJ_GEOM,
            objname=capsule.name,
        )
        for joint in joints:
            for sensor_type, suffix in (
                (mujoco.mjtSensor.mjSENS_JOINTPOS, "jointpos_sensor"),
                (mujoco.mjtSensor.mjSENS_JOINTVEL, "jointvel_sensor"),
                (mujoco.mjtSensor.mjSENS_JOINTACTFRC, "actuatorfrc_sensor"),
            ):
                sensor(
                    name=f"{joint.name}_{suffix}",
                    type=sensor_type,
                    objtype=_JOINT,
                    objname=joint.name,
                )
        for actuator in actuators:
            sensor(
                name=f"{actuator}_actuatorfrc_sensor",
                type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
                objtype=mujoco.mjtObj.mjOBJ_ACTUATOR,
                objname=actuator,
            )
        for tendon in tendons:
            for sensor_type, suffix in (
                (mujoco.mjtSensor.mjSENS_TENDONPOS, "tendonpos_sensor"),
                (mujoco.mjtSensor.mjSENS_TENDONVEL, "tendonvel_sensor"),
            ):
                sensor(
                    name=f"{tendon}_{suffix}",
                    type=sensor_type,
                    objtype=mujoco.mjtObj.mjOBJ_TENDON,
                    objname=tendon,
                )
        for site in self._add_contact_sites(body, name, capsule, segment):
            sensor(
                name=f"{site}_contact_sensor",
                type=mujoco.mjtSensor.mjSENS_TOUCH,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                objname=site,
            )

    def _add_contact_sites(
        self,
        body: mujoco.MjsBody,
        name: str,
        capsule: mujoco.MjsGeom,
        segment: BrittleStarArmSegmentSpecification,
    ) -> List[str]:
        """Touch sensor zones (group 3, hidden by default): one capsule enclosing the segment, or N boxes around it."""
        num_sensors = (
            self.specification.sensor_specification.num_contact_sensors_per_segment
        )
        rgba = colors.rgba_red * np.array([1, 1, 1, 0.5])
        if num_sensors == 1:
            return [
                body.add_site(
                    name=f"{name}_contact_site",
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    pos=capsule.pos,
                    quat=capsule.quat,
                    size=np.array(capsule.size) * np.array([1.01, 1, 1]),
                    rgba=rgba,
                    group=3,
                ).name
            ]

        radius, length = segment.radius, segment.length
        angles = np.linspace(-np.pi / 2, 1.5 * np.pi, num_sensors + 1)[:num_sensors]
        return [
            body.add_site(
                name=f"{name}_contact_site_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=_capsule_center(segment)
                + 0.95 * radius * np.array([0, np.cos(angle), np.sin(angle)]),
                quat=euler_to_quat(np.array([angle, 0, 0])),
                size=[length / 2 + radius / 2, 0.05 * radius, 0.4 * radius],
                rgba=rgba,
                group=3,
            ).name
            for i, angle in enumerate(angles)
        ]


def _capsule_center(segment: BrittleStarArmSegmentSpecification) -> np.ndarray:
    """Capsule centre in the segment frame: the capsule starts one radius after the joint (room for the sphere)."""
    return np.array([segment.radius + segment.length / 2, 0, 0])
