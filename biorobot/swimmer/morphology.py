"""The swimmer morphology, built with ``mujoco.MjSpec``.

Layout (names are prefixed with ``SwimmerMorphology/`` once attached to an arena)::

    segment_0                    the head: free body, capsule along +x
    └── segment_1                at the end of segment 0's axis, hinge around z
        └── segment_2 ...

The swimmer lies along +x with its head at the origin, so it swims head first towards -x. Every joint has a position
servo and joint position, velocity and actuator-force sensors; the head has frame sensors and a tracking camera.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from biorobot.core import Morphology
from biorobot.swimmer import colors
from biorobot.swimmer.specification import (
    SwimmerMorphologySpecification,
    default_swimmer_morphology_specification,
)

MORPHOLOGY_NAME = "SwimmerMorphology"

_JOINT = mujoco.mjtObj.mjOBJ_JOINT
_TRUE = mujoco.mjtLimited.mjLIMITED_TRUE


class SwimmerMorphology(Morphology[SwimmerMorphologySpecification]):
    """A planar chain of capsules that swims through MuJoCo's fluid model."""

    def __init__(
        self, specification: Optional[SwimmerMorphologySpecification] = None
    ) -> None:
        super().__init__(
            specification=specification or default_swimmer_morphology_specification(),
            name=MORPHOLOGY_NAME,
        )

    def _build(self) -> None:
        spec = self.specification
        parent = self.spec.worldbody
        position = np.zeros(3)
        for index, segment in enumerate(spec.segment_specifications):
            body = parent.add_body(name=f"segment_{index}", pos=position)
            if index > 0:
                self._add_joint(body, index)
            body.add_geom(
                name=f"segment_{index}_capsule",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, segment.length, 0, 0],
                size=[segment.radius, 0, 0],
                density=segment.density,
                rgba=colors.BODY,
                fluid_ellipsoid=int(spec.fluid_model == "ellipsoid"),
            )
            parent, position = body, np.array([segment.length, 0, 0])

        head = self.spec.body("segment_0")
        for sensor_type, suffix in (
            (mujoco.mjtSensor.mjSENS_FRAMEPOS, "framepos"),
            (mujoco.mjtSensor.mjSENS_FRAMEQUAT, "framequat"),
            (mujoco.mjtSensor.mjSENS_FRAMELINVEL, "framelinvel"),
            (mujoco.mjtSensor.mjSENS_FRAMEANGVEL, "frameangvel"),
        ):
            self.spec.add_sensor(
                name=f"head_{suffix}_sensor",
                type=sensor_type,
                objtype=mujoco.mjtObj.mjOBJ_XBODY,
                objname=head.name,
            )
        camera = head.add_camera(
            name="tracking_camera",
            pos=[0, 0, 2.5],
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
            fovy=60,
        )
        camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        camera.alt.xyaxes = [1, 0, 0, 0, 1, 0]

    def _add_joint(self, body: mujoco.MjsBody, segment_index: int) -> None:
        joint_spec = self.specification.joint_specifications[segment_index - 1]
        actuation = self.specification.actuation_specification
        joint = body.add_joint(
            name=f"segment_{segment_index}_joint",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=[0, 0, 1],
            limited=_TRUE,
            range=[-joint_spec.range, joint_spec.range],
            stiffness=joint_spec.stiffness,
            damping=joint_spec.damping,
            armature=joint_spec.armature,
        )
        actuator = self.spec.add_actuator(
            name=f"{joint.name}_p_control",
            target=joint.name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=_TRUE,
            ctrlrange=[-actuation.control_range, actuation.control_range],
        )
        actuator.set_to_position(kp=actuation.kp)

        for sensor_type, suffix in (
            (mujoco.mjtSensor.mjSENS_JOINTPOS, "jointpos_sensor"),
            (mujoco.mjtSensor.mjSENS_JOINTVEL, "jointvel_sensor"),
            (mujoco.mjtSensor.mjSENS_JOINTACTFRC, "actuatorfrc_sensor"),
        ):
            self.spec.add_sensor(
                name=f"{joint.name}_{suffix}",
                type=sensor_type,
                objtype=_JOINT,
                objname=joint.name,
            )
