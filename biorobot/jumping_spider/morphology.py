"""MjSpec builder of the jumping spider morphology.

Structure (names without the ``JumpingSpiderMorphology/`` prefix that is added when the morphology is attached):

* ``cephalothorax`` (root body): ellipsoid, plus the tracking camera ``side_camera``.
* ``abdomen``: ellipsoid behind the cephalothorax with the joints ``abdomen_in_plane`` and ``abdomen_out_of_plane``,
  and the ``spinneret`` site where the optional dragline starts.
* Eight legs ``leg_<side><pair>`` (side ``L``/``R``, pair ``1``-``4`` for legs I-IV), each a chain of seven segment
  bodies (and capsule geoms) ``leg_L1_coxa`` ... ``leg_L1_tarsus``. Segment joints are named
  ``<segment>_in_plane`` / ``<segment>_out_of_plane`` (e.g. ``leg_L1_coxa_in_plane``); every tarsus has a touch site
  and sensor ``leg_L1_tarsus_touch``.

Every joint has one actuator with the same name as the joint, and ``jointpos``/``jointvel``/``actuatorfrc`` sensors
named ``<joint>_jointpos``, ``<joint>_jointvel`` and ``<joint>_actuatorfrc``.

The ellipsoids only carry mass and appearance; collisions use capsule proxies (geom group 3) because MJX has no
ellipsoid-box collision function, and platforms are boxes. Only morphology-arena contacts are simulated (the
morphology's geoms have ``contype=1, conaffinity=0``, arena geoms the opposite).
"""

from __future__ import annotations

from typing import List

import mujoco
import numpy as np

from biorobot.core.builder import Morphology, euler_to_quat
from biorobot.jumping_spider import colors
from biorobot.jumping_spider.specification import (
    JointSpecification,
    JumpingSpiderMorphologySpecification,
    default_jumping_spider_morphology_specification,
)

MORPHOLOGY_NAME = "JumpingSpiderMorphology"
COLLISION_GROUP = 3


class JumpingSpiderMorphology(Morphology[JumpingSpiderMorphologySpecification]):
    def __init__(self, specification: JumpingSpiderMorphologySpecification) -> None:
        super().__init__(specification=specification, name=MORPHOLOGY_NAME)

    def _build(self) -> None:
        spec = self.specification
        default = self.spec.default.geom
        default.contype = 1
        default.conaffinity = 0
        default.condim = 3
        default.density = spec.density

        self._cephalothorax = self._build_cephalothorax()
        self._build_abdomen()
        for pair_index in range(len(spec.legs)):
            for side in ("L", "R"):
                self._build_leg(pair_index=pair_index, side=side)
        self._build_camera()

    # ------------------------------------------------------------------------------------------------------------------
    # Body
    # ------------------------------------------------------------------------------------------------------------------
    def _build_cephalothorax(self) -> mujoco.MjsBody:
        size = np.asarray(self.specification.cephalothorax.size, dtype=float)
        body = self.spec.worldbody.add_body(name="cephalothorax")
        self._add_ellipsoid(body, name="cephalothorax", size=size, pos=np.zeros(3))
        for sensor_type, name in (
            (mujoco.mjtSensor.mjSENS_FRAMEPOS, "framepos"),
            (mujoco.mjtSensor.mjSENS_FRAMEQUAT, "framequat"),
            (mujoco.mjtSensor.mjSENS_FRAMELINVEL, "framelinvel"),
            (mujoco.mjtSensor.mjSENS_FRAMEANGVEL, "frameangvel"),
        ):
            self.spec.add_sensor(
                name=f"cephalothorax_{name}",
                type=sensor_type,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                objname="cephalothorax",
            )
        return body

    def _build_abdomen(self) -> None:
        abdomen = self.specification.abdomen
        size = np.asarray(abdomen.size, dtype=float)
        body = self._cephalothorax.add_body(
            name="abdomen",
            pos=[-self.specification.cephalothorax.size[0], 0, 0],
        )
        center = np.array([-size[0] * (1 - abdomen.inset_factor), 0, 0])
        self._add_ellipsoid(body, name="abdomen", size=size, pos=center)

        # The abdomen points backwards, so positive rotations around +z / +y turn its tip right / raise it.
        self._add_joint(body, "abdomen_in_plane", [0, 0, 1], abdomen.ip_joint)
        self._add_joint(body, "abdomen_out_of_plane", [0, 1, 0], abdomen.oop_joint)
        for joint_name in ("abdomen_in_plane", "abdomen_out_of_plane"):
            self._add_actuator(joint_name, abdomen.torque_limit)

        # The spinnerets sit at the rear underside of the abdomen.
        angle = -np.pi / 6
        spinneret = center + np.array(
            [-size[0] * np.cos(angle), 0.0, size[2] * np.sin(angle)]
        )
        body.add_site(
            name="spinneret",
            pos=spinneret,
            size=[0.002, 0, 0],
            rgba=colors.SITE,
        )

    def _add_ellipsoid(
        self, body: mujoco.MjsBody, name: str, size: np.ndarray, pos: np.ndarray
    ) -> None:
        """A massive, non-colliding ellipsoid plus a massless capsule (along x) collision proxy inscribed in it."""
        body.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=size,
            pos=pos,
            rgba=colors.BODY,
            contype=0,
            conaffinity=0,
        )
        radius = min(size[1], size[2])
        half_length = max(size[0] - radius, 0.0)
        body.add_geom(
            name=f"{name}_collision",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=np.concatenate(
                [pos - [half_length, 0, 0], pos + [half_length, 0, 0]]
            ),
            size=[radius, 0, 0],
            density=0,
            friction=[self.specification.body_friction, 0.005, 0.0001],
            group=COLLISION_GROUP,
            rgba=colors.BODY,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Legs
    # ------------------------------------------------------------------------------------------------------------------
    def _build_leg(self, pair_index: int, side: str) -> None:
        leg = self.specification.legs[pair_index]
        ceph_size = self.specification.cephalothorax.size
        sign = 1.0 if side == "L" else -1.0
        ip_angle, oop_angle = leg.in_plane_angle, leg.out_of_plane_angle
        position = np.array(
            [
                ceph_size[0] * np.cos(oop_angle) * np.cos(ip_angle),
                sign * ceph_size[1] * np.cos(oop_angle) * np.sin(ip_angle),
                ceph_size[2] * np.sin(oop_angle),
            ]
        )
        leg_name = f"leg_{side}{pair_index + 1}"
        parent = self._cephalothorax.add_body(
            name=leg_name,
            pos=position,
            quat=euler_to_quat(np.array([0.0, 0.0, sign * ip_angle])),
        )

        # Mirror the in-plane axis on the right side so that positive angles swing both legs of a pair backwards.
        ip_axis = [0, 0, sign]
        pos = np.zeros(3)
        for segment in leg.segments:
            name = f"{leg_name}_{segment.name}"
            body = parent.add_body(
                name=name,
                pos=pos,
                quat=euler_to_quat(np.array([0.0, segment.attachment_angle, 0.0])),
            )
            body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, segment.length, 0, 0],
                size=[segment.radius, 0, 0],
                friction=[self.specification.leg_friction, 0.005, 0.0001],
                rgba=colors.LEG,
            )
            if segment.ip_joint is not None:
                self._add_joint(body, f"{name}_in_plane", ip_axis, segment.ip_joint)
                self._add_actuator(f"{name}_in_plane", segment.torque_limit)
            self._add_joint(body, f"{name}_out_of_plane", [0, 1, 0], segment.oop_joint)
            self._add_actuator(f"{name}_out_of_plane", segment.torque_limit)

            parent, pos = body, np.array([segment.length, 0.0, 0.0])

        # Touch sensor on the last segment (the leg tip), slightly larger than the capsule to catch all its contacts.
        tip = leg.segments[-1]
        parent.add_site(
            name=f"{parent.name}_touch",
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, 0, 0, tip.length, 0, 0],
            size=[1.2 * tip.radius, 0, 0],
            rgba=[0, 0, 0, 0],
        )
        self.spec.add_sensor(
            name=f"{parent.name}_touch",
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=f"{parent.name}_touch",
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Joints, actuators and sensors
    # ------------------------------------------------------------------------------------------------------------------
    def _add_joint(
        self,
        body: mujoco.MjsBody,
        name: str,
        axis: List[float],
        specification: JointSpecification,
    ) -> None:
        body.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=axis,
            range=list(specification.range),
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            stiffness=specification.stiffness,
            damping=specification.damping,
            armature=specification.armature,
        )
        for sensor_type, suffix in (
            (mujoco.mjtSensor.mjSENS_JOINTPOS, "jointpos"),
            (mujoco.mjtSensor.mjSENS_JOINTVEL, "jointvel"),
        ):
            self.spec.add_sensor(
                name=f"{name}_{suffix}",
                type=sensor_type,
                objtype=mujoco.mjtObj.mjOBJ_JOINT,
                objname=name,
            )

    def _add_actuator(self, joint_name: str, torque_limit: float) -> None:
        actuation = self.specification.actuation
        joint = self.spec.joint(joint_name)
        actuator = self.spec.add_actuator(
            name=joint_name,
            target=joint_name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            forcerange=[-torque_limit, torque_limit],
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )
        if actuation.position_control:
            actuator.set_to_position(
                kp=torque_limit / actuation.saturation_error,
                dampratio=actuation.damping_ratio,
            )
            actuator.ctrlrange = joint.range
        else:
            actuator.set_to_motor()
            actuator.ctrlrange = [-torque_limit, torque_limit]
        # First-order activation dynamics: the control signal is low-pass filtered before it drives the joint.
        actuator.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
        actuator.dynprm[0] = actuation.activation_time_constant
        self.spec.add_sensor(
            name=f"{joint_name}_actuatorfrc",
            type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
            objtype=mujoco.mjtObj.mjOBJ_ACTUATOR,
            objname=joint_name,
        )

    def _build_camera(self) -> None:
        self._cephalothorax.add_camera(
            name="side_camera",
            pos=[0.0, -0.6, 0.25],
            quat=xyaxes_to_quat([1, 0, 0], [0, 0.38, 0.92]),
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK,
        )


def add_dragline(
    spec: mujoco.MjSpec,
    specification: JumpingSpiderMorphologySpecification,
    anchor_site: str,
    prefix: str = f"{MORPHOLOGY_NAME}/",
) -> None:
    """Connect the spinneret to ``anchor_site`` (a site of the arena) in a composed spec.

    Adds the spatial tendon ``dragline``, its brake actuator ``dragline_brake`` (``ctrl`` in [0, 1]) and the sensors
    ``dragline_length``, ``dragline_velocity`` and ``dragline_brake_actuatorfrc`` (all prefixed).
    """
    dragline = specification.dragline
    tendon = spec.add_tendon(
        name=f"{prefix}dragline",
        stiffness=dragline.stiffness,
        damping=dragline.damping,
        width=0.0005,
        rgba=colors.DRAGLINE,
    )
    tendon.wrap_site(f"{prefix}spinneret")
    tendon.wrap_site(anchor_site)

    brake = spec.add_actuator(
        name=f"{prefix}dragline_brake",
        target=f"{prefix}dragline",
        trntype=mujoco.mjtTrn.mjTRN_TENDON,
        ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
        ctrlrange=[0, 1],
        forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
        # Silk can only pull (negative tendon force).
        forcerange=[-dragline.max_brake_force, 0],
    )
    brake.set_to_damper(kv=dragline.brake_damping)
    brake.ctrlrange = [0, 1]

    for sensor_type, name, objtype, objname in (
        (
            mujoco.mjtSensor.mjSENS_TENDONPOS,
            "dragline_length",
            mujoco.mjtObj.mjOBJ_TENDON,
            "dragline",
        ),
        (
            mujoco.mjtSensor.mjSENS_TENDONVEL,
            "dragline_velocity",
            mujoco.mjtObj.mjOBJ_TENDON,
            "dragline",
        ),
        (
            mujoco.mjtSensor.mjSENS_ACTUATORFRC,
            "dragline_brake_actuatorfrc",
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "dragline_brake",
        ),
    ):
        spec.add_sensor(
            name=f"{prefix}{name}",
            type=sensor_type,
            objtype=objtype,
            objname=f"{prefix}{objname}",
        )


def xyaxes_to_quat(x_axis: List[float], y_axis: List[float]) -> np.ndarray:
    """Quaternion of a frame given its x- and y-axis (like MJCF's ``xyaxes``)."""
    x = np.asarray(x_axis, dtype=float)
    x /= np.linalg.norm(x)
    y = np.asarray(y_axis, dtype=float)
    y -= x * (x @ y)
    y /= np.linalg.norm(y)
    mat = np.stack([x, y, np.cross(x, y)], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


def default_morphology(
    dragline: bool = False, position_control: bool = True
) -> JumpingSpiderMorphology:
    return JumpingSpiderMorphology(
        default_jumping_spider_morphology_specification(
            dragline=dragline, position_control=position_control
        )
    )
