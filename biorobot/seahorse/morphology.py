"""MjSpec builder of the seahorse tail.

Structure (names without the ``SeahorseMorphology/`` prefix that is added when the morphology is attached; ``s`` is the
segment index, ``p`` the plate index):

* ``segment_<s>`` bodies form a chain: ``segment_<s>`` is a child of ``segment_<s-1>_vertebrae``.
* ``segment_<s>_vertebrae``: the vertebra with the hinge joints ``segment_<s>_vertebrae_vertebrae_joint_{pitch,roll,
  yaw}`` (all but the first segment, which is fixed to the world) and the mesh bodies ``..._vertebrae_body``,
  ``..._ball_bearing_body`` and ``..._connector_<side>_body``.
* ``segment_<s>_plate_<p>``: a plate with the slide joints ``segment_<s>_plate_<p>_{x,y}_axis`` (gliding joints; all
  but the first segment), its mesh body ``..._plate_body`` and, on inner plates, ``..._connector_{x,y}_body``.
  Neighbouring gliding joints are coupled by joint equality constraints.
* Spatial tendons: ``segment_<s>_vertebral_vertebral_strut_<side>`` (passive), ``hm_beam_<sagittal>_<coronal>_<end>``
  and ``mvm_beam_<side>_(<s>, <s+1>)``; every beam has an actuator with the same name.

Sensors: ``<joint>_sensor`` (jointpos of the vertebral joints), ``<beam>_position_sensor`` (tendonpos),
``<joint>_actuatorfrc_sensor`` (jointactuatorfrc of the vertebral joints: the torque that the beams exert on them) and
``<actuator>_actuatorfrc_sensor`` (actuatorfrc).

Mass and inertia come from the measured values of the printed parts (``inertia.json``), not from the meshes. All
geoms have the default ``contype = conaffinity = 1``; whether contacts are simulated is a global option of the
environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

from biorobot.core.builder import Morphology, euler_to_quat
from biorobot.seahorse import colors
from biorobot.seahorse.specification import (
    MeshSpecification,
    SeahorseMorphologySpecification,
    SeahorsePlateSpecification,
    is_inner_plate_x_axis,
    is_inner_plate_y_axis,
)

MORPHOLOGY_NAME = "SeahorseMorphology"
#: Joint equality coefficients ``joint1 = coef * joint2``.
_SAME, _OPPOSITE = [0, 1, 0, 0, 0], [0, -1, 0, 0, 0]
_TAP_Z_OFFSETS = (-0.0092, -0.0057)
_GLIDING_SOLIMPLIMIT = [0.90, 0.9999, 0.001, 0.1, 6]


@dataclass
class _Plate:
    body: mujoco.MjsBody
    plate_position: np.ndarray
    connector_positions: Dict[str, np.ndarray] = field(default_factory=dict)
    joints: Dict[str, str] = field(default_factory=dict)
    intermediate_hm_taps: List[List[str]] = field(default_factory=list)
    lower_ghost_hm_taps: List[List[str]] = field(default_factory=list)
    hm_tap_end: List[str] = field(default_factory=list)
    mvm_taps: Dict[str, List[str]] = field(default_factory=dict)
    s_taps: Dict[str, str] = field(default_factory=dict)


@dataclass
class _Segment:
    vertebrae: mujoco.MjsBody
    vertebral_joints: List[str]
    vertebrae_s_taps: List[str]
    plates: List[_Plate]


def plate_position(
    plate_index: int, plate_specification: SeahorsePlateSpecification
) -> np.ndarray:
    """Position (m) of a plate's origin relative to its vertebra."""
    angle = [5 * np.pi / 4, 7 * np.pi / 4, np.pi / 4, 3 * np.pi / 4][plate_index]
    position = plate_specification.offset_from_vertebrae * np.array(
        [-np.sin(angle), np.cos(angle), 0.0]
    )
    position[2] = plate_specification.depth / 2
    return position


def actuator_beam_plate_indices(side: str, segment_index: int) -> Tuple[int, int]:
    """The plate (first) of a sagittal side that holds the HM ghost attachment points, and the other one."""
    if side == "ventral":
        return 1, 0
    return (2, 3) if segment_index % 2 == 0 else (3, 2)


def actuator_ghost_taps_index(x_side: str, y_side: str, segment_index: int) -> int:
    """Which pair of ghost attachment points (0: ``a``, 1: ``b``) an HM beam on the given sides starts from."""
    if x_side == "ventral" or segment_index % 2 == 0:
        return 1 if y_side == "dextral" else 0
    return 0 if y_side == "dextral" else 1


def inner_and_outer_plate_indices(segment_index: int, side: str) -> Tuple[int, int]:
    """The (inner, outer) plates on one side of a segment."""
    plates = {
        "ventral": (0, 1),
        "dextral": (0, 3),
        "sinistral": (1, 2),
        "dorsal": (2, 3),
    }[side]
    is_inner = (
        is_inner_plate_x_axis
        if side in ("ventral", "dorsal")
        else is_inner_plate_y_axis
    )
    inner, outer = sorted(
        plates,
        key=lambda plate_index: not is_inner(
            segment_index=segment_index, plate_index=plate_index
        ),
    )
    return inner, outer


class SeahorseMorphology(Morphology[SeahorseMorphologySpecification]):
    def __init__(self, specification: SeahorseMorphologySpecification) -> None:
        super().__init__(specification=specification, name=MORPHOLOGY_NAME)

    def _build(self) -> None:
        self._mesh_names: Dict[Tuple[str, Tuple[float, ...]], str] = {}
        self._segments: List[_Segment] = []
        parent = self.spec.worldbody
        for segment_index in range(self.specification.num_segments):
            segment = self._build_segment(parent=parent, segment_index=segment_index)
            self._segments.append(segment)
            parent = segment.vertebrae
        self._couple_gliding_joints_between_segments()
        hm_beams = self._build_hm_beams()
        mvm_beams = self._build_mvm_beams()
        self._build_actuators(hm_beams=hm_beams, mvm_beams=mvm_beams)

    # ------------------------------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------------------------------
    def _mesh(self, mesh_specification: MeshSpecification) -> str:
        """Name of the mesh asset of ``mesh_specification`` (one asset per file and scale)."""
        scale = tuple(float(s) for s in mesh_specification.scale)
        key = (mesh_specification.mesh_path, scale)
        if key not in self._mesh_names:
            stem = Path(mesh_specification.mesh_path).stem
            name = (
                stem
                if stem not in self._mesh_names.values()
                else f"{stem}_{len(self._mesh_names)}"
            )
            self.spec.add_mesh(
                name=name, file=mesh_specification.mesh_path, scale=list(scale)
            )
            self._mesh_names[key] = name
        return self._mesh_names[key]

    def _add_mesh_body(
        self,
        parent: mujoco.MjsBody,
        name: str,
        mesh_specification: MeshSpecification,
        position: np.ndarray,
        euler: np.ndarray,
        rgba: np.ndarray,
        group: int,
    ) -> mujoco.MjsBody:
        """A body ``<name>_body`` with the part's measured inertia and its mesh geom ``<name>_geom``."""
        body = parent.add_body(
            name=f"{name}_body", pos=position, quat=euler_to_quat(euler)
        )
        body.explicitinertial = True
        body.mass = mesh_specification.mass
        body.ipos = mesh_specification.center_of_mass
        body.fullinertia = mesh_specification.fullinertia
        body.add_geom(
            name=f"{name}_geom",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=self._mesh(mesh_specification),
            rgba=rgba,
            group=group,
        )
        return body

    @staticmethod
    def _add_site(
        body: mujoco.MjsBody, name: str, position: np.ndarray, radius: float
    ) -> str:
        site = body.add_site(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=np.array(position, dtype=float),
            rgba=colors.SITE,
        )
        site.size[0] = radius
        return name

    def _add_joint_sensor(
        self, joint: str, sensor_type: mujoco.mjtSensor, suffix: str
    ) -> None:
        self.spec.add_sensor(
            name=f"{joint}_{suffix}",
            type=sensor_type,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=joint,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Segments
    # ------------------------------------------------------------------------------------------------------------------
    def _build_segment(self, parent: mujoco.MjsBody, segment_index: int) -> _Segment:
        segment_specification = self.specification.segment_specifications[segment_index]
        name = f"segment_{segment_index}"
        body = parent.add_body(
            name=name,
            pos=[0.0, 0.0, segment_specification.z_offset_from_previous_segment],
        )
        vertebrae, joints, s_taps = self._build_vertebrae(body, segment_index)
        plates = [
            self._build_plate(vertebrae, segment_index, plate_index)
            for plate_index in range(4)
        ]
        segment = _Segment(
            vertebrae=vertebrae,
            vertebral_joints=joints,
            vertebrae_s_taps=s_taps,
            plates=plates,
        )
        self._build_vertebral_struts(segment, segment_index)
        self._couple_gliding_joints_within_segment(plates)
        return segment

    def _build_vertebrae(
        self, parent: mujoco.MjsBody, segment_index: int
    ) -> Tuple[mujoco.MjsBody, List[str], List[str]]:
        specification = self.specification.segment_specifications[
            segment_index
        ].vertebrae_specification
        name = f"segment_{segment_index}_vertebrae"
        body = parent.add_body(name=name)

        self._add_mesh_body(
            body,
            f"{name}_vertebrae",
            specification.vertebral_mesh_specification,
            np.zeros(3),
            np.zeros(3),
            colors.VERTEBRAE,
            group=1,
        )
        self._add_mesh_body(
            body,
            f"{name}_ball_bearing",
            specification.ball_bearing_mesh_specification,
            np.array([0, 0, specification.z_offset_to_ball_bearing]),
            np.zeros(3),
            colors.CONNECTOR,
            group=1,
        )
        radius = specification.offset_to_bar_end + specification.connector_length / 2
        for side_index, side in enumerate(
            ["ventral", "dextral", "dorsal", "sinistral"]
        ):
            angle = np.pi / 2 * side_index
            self._add_mesh_body(
                body,
                f"{name}_connector_{side}",
                specification.connector_mesh_specification,
                radius * np.array([np.cos(angle), np.sin(angle), 0.0]),
                np.array([0, 0, angle]),
                colors.CONNECTOR,
                group=1,
            )

        s_tap_radius = specification.offset_to_vertebral_strut_attachment_point
        s_taps = [
            self._add_site(
                body,
                f"{name}_s_tap_{side}",
                s_tap_radius * np.array([np.cos(angle), np.sin(angle), 0.0]),
                0.001,
            )
            for side, angle in zip(
                self.specification.sides, [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            )
        ]

        joints = []
        if segment_index > 0:
            # All three rotations are centred on the ball bearing between this vertebra and the previous one.
            for axis_name, axis, joint_specification, armature in (
                ("pitch", [0, 1, 0], specification.pitch_joint_specification, True),
                ("roll", [1, 0, 0], specification.roll_joint_specification, True),
                ("yaw", [0, 0, 1], specification.yaw_joint_specification, False),
            ):
                if joint_specification.range == 0:
                    continue
                joint = body.add_joint(
                    name=f"{name}_vertebrae_joint_{axis_name}",
                    type=mujoco.mjtJoint.mjJNT_HINGE,
                    pos=[0.0, 0.0, -specification.z_offset_to_ball_bearing],
                    axis=axis,
                    limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                    range=[-joint_specification.range, joint_specification.range],
                    stiffness=joint_specification.stiffness,
                    damping=joint_specification.damping,
                    frictionloss=joint_specification.friction_loss,
                )
                if armature:
                    joint.armature = joint_specification.armature
                joints.append(joint.name)
                self._add_joint_sensor(
                    joint.name, mujoco.mjtSensor.mjSENS_JOINTPOS, "sensor"
                )
        return body, joints, s_taps

    def _build_plate(
        self, parent: mujoco.MjsBody, segment_index: int, plate_index: int
    ) -> _Plate:
        specification = self.specification.segment_specifications[
            segment_index
        ].plate_specifications[plate_index]
        name = f"segment_{segment_index}_plate_{plate_index}"
        body = parent.add_body(name=name)
        position = plate_position(plate_index, specification)
        plate = _Plate(body=body, plate_position=position)

        self._add_mesh_body(
            body,
            f"{name}_plate",
            specification.plate_mesh_specification,
            position,
            np.zeros(3),
            colors.PLATE,
            group=0,
        )
        self._build_plate_connectors(
            plate, name, specification, segment_index, plate_index
        )
        if segment_index > 0:
            self._build_gliding_joints(plate, name, specification, plate_index)
        self._build_plate_hm_taps(
            plate, name, specification, segment_index, plate_index
        )
        mvm = (
            self.specification.beam_actuation_specification.mvm_beam_actuation_specification
        )
        if mvm.enabled and plate_index == 1:
            self._build_plate_mvm_taps(plate, name, specification)
        self._build_plate_s_taps(plate, name, specification, plate_index)
        return plate

    def _build_plate_connectors(
        self,
        plate: _Plate,
        name: str,
        specification: SeahorsePlateSpecification,
        segment_index: int,
        plate_index: int,
    ) -> None:
        offset = specification.connector_offset_from_vertebrae
        if is_inner_plate_x_axis(segment_index=segment_index, plate_index=plate_index):
            direction = 1 if plate_index < 2 else -1
            plate.connector_positions["x"] = np.array([direction * offset, 0.0, 0.0])
            self._add_mesh_body(
                plate.body,
                f"{name}_connector_x",
                specification.connector_mesh_specification,
                plate.connector_positions["x"],
                np.zeros(3),
                colors.CONNECTOR,
                group=0,
            )
        if is_inner_plate_y_axis(segment_index=segment_index, plate_index=plate_index):
            direction = 1 if 1 <= plate_index <= 2 else -1
            plate.connector_positions["y"] = np.array([0.0, direction * offset, 0.0])
            self._add_mesh_body(
                plate.body,
                f"{name}_connector_y",
                specification.connector_mesh_specification,
                plate.connector_positions["y"],
                np.array([0.0, 0.0, np.pi / 2]),
                colors.CONNECTOR,
                group=0,
            )

    def _build_gliding_joints(
        self,
        plate: _Plate,
        name: str,
        specification: SeahorsePlateSpecification,
        plate_index: int,
    ) -> None:
        """Slide joints along x and y; each plate can only slide away from the plate it overlaps."""
        for axis_name, axis, joint_specification, negative in (
            (
                "x",
                [1, 0, 0],
                specification.x_axis_gliding_joint_specification,
                plate_index in (0, 1),
            ),
            (
                "y",
                [0, 1, 0],
                specification.y_axis_gliding_joint_specification,
                plate_index in (1, 2),
            ),
        ):
            if joint_specification.range == 0:
                continue
            joint_range = (
                [-joint_specification.range, 0]
                if negative
                else [0, joint_specification.range]
            )
            joint = plate.body.add_joint(
                name=f"{name}_{axis_name}_axis",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=axis,
                limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                range=joint_range,
                damping=joint_specification.damping,
                stiffness=joint_specification.stiffness,
                solimp_limit=_GLIDING_SOLIMPLIMIT,
            )
            plate.joints[axis_name] = joint.name

    def _build_plate_hm_taps(
        self,
        plate: _Plate,
        name: str,
        specification: SeahorsePlateSpecification,
        segment_index: int,
        plate_index: int,
    ) -> None:
        # Intermediate attachment points: pairs of holes from the plate's side (0) towards the vertebra.
        first_tap_position = (
            np.array(
                [
                    specification.hm_intermediate_first_tap_x_offset_from_plate_origin,
                    specification.hm_intermediate_first_tap_y_offset_from_plate_origin,
                    0.0,
                ]
            )
            + plate.plate_position
        )
        translation = np.array(
            [0.0, specification.hm_y_offset_between_intermediate_taps, 0.0]
        )
        for tap_index in range(specification.hm_num_intermediate_taps):
            position = (
                first_tap_position
                + -np.sign(first_tap_position[1]) * tap_index * translation
            )
            plate.intermediate_hm_taps.append(
                self._add_tap_pair(
                    plate.body,
                    f"{name}_intermediate_hm_tap_{tap_index}",
                    position,
                    0.0005,
                )
            )

        # Ghost (start) attachment points: on one plate per sagittal side, a pair for each coronal side.
        side = self.specification.corners[plate_index].split("_")[0]
        if plate_index == actuator_beam_plate_indices(side, segment_index)[0]:
            position = (
                np.array(
                    [
                        specification.hm_ghost_tap_x_offset_from_plate_origin,
                        specification.hm_ghost_tap_y_offset_from_plate_origin,
                        0.0,
                    ]
                )
                + plate.plate_position
            )
            y_offsets = [0.0, -np.sign(position[1]) * 0.012]
            position[0] += -0.002 if side == "dorsal" else 0.002
            for label, y_offset in zip(["a", "b"], y_offsets):
                position[1] += y_offset
                plate.lower_ghost_hm_taps.append(
                    self._add_tap_pair(
                        plate.body, f"{name}_ghost_hm_tap_{label}", position, 0.0005
                    )
                )
                position[1] -= y_offset

        for tap_index, z_offset in enumerate([-0.002, 0.0]):
            plate.hm_tap_end.append(
                self._add_site(
                    plate.body,
                    f"{name}_end_hm_tap_{tap_index}",
                    plate.plate_position + np.array([0.0, 0.0, z_offset]),
                    0.001,
                )
            )

    def _build_plate_mvm_taps(
        self, plate: _Plate, name: str, specification: SeahorsePlateSpecification
    ) -> None:
        position = (
            np.array(
                [
                    specification.hm_ghost_tap_x_offset_from_plate_origin,
                    specification.hm_ghost_tap_y_offset_from_plate_origin,
                    0.0,
                ]
            )
            + plate.plate_position
        )
        y_offsets = [0.0, -np.sign(position[1]) * 0.012]
        position[0] += -0.002
        for side, y_offset in zip(["sinistral", "dextral"], y_offsets):
            position[1] += y_offset
            plate.mvm_taps[side] = self._add_tap_pair(
                plate.body, f"{name}_mvm_tap_{side}", position, 0.001
            )
            position[1] -= y_offset

    def _add_tap_pair(
        self, body: mujoco.MjsBody, name: str, position: np.ndarray, radius: float
    ) -> List[str]:
        """Two attachment points ``<name>_0`` and ``<name>_1`` at the depths of a hole's entry and exit.

        ``position`` is shifted along z and back in place, which reproduces the original model's round-off exactly.
        """
        taps = []
        for sub_index, z_offset in enumerate(_TAP_Z_OFFSETS):
            position[2] += z_offset
            taps.append(self._add_site(body, f"{name}_{sub_index}", position, radius))
            position[2] -= z_offset
        return taps

    def _build_plate_s_taps(
        self,
        plate: _Plate,
        name: str,
        specification: SeahorsePlateSpecification,
        plate_index: int,
    ) -> None:
        x_direction = 1 if plate_index < 2 else -1
        y_direction = 1 if 1 <= plate_index <= 2 else -1
        plate.s_taps["plate-x"] = self._add_site(
            plate.body,
            f"{name}_s_tap_x",
            np.array(
                [x_direction * specification.s_tap_x_offset_from_vertebrae, 0.0, 0.0]
            ),
            0.001,
        )
        plate.s_taps["plate-y"] = self._add_site(
            plate.body,
            f"{name}_s_tap_y",
            np.array(
                [0.0, y_direction * specification.s_tap_y_offset_from_vertebrae, 0.0]
            ),
            0.001,
        )
        for axis, position in plate.connector_positions.items():
            # Slightly closer to the vertebra than the connector itself.
            plate.s_taps[f"connector-{axis}"] = self._add_site(
                plate.body, f"{name}_connector_{axis}_body_s_tap", 0.9 * position, 0.001
            )

    def _build_vertebral_struts(self, segment: _Segment, segment_index: int) -> None:
        """One strut per side: vertebra -> connector of the inner plate -> inner plate -> outer plate."""
        strut = self.specification.segment_specifications[
            segment_index
        ].vertebral_strut_specification
        for side, vertebrae_s_tap in zip(
            self.specification.sides, segment.vertebrae_s_taps
        ):
            inner_index, outer_index = inner_and_outer_plate_indices(
                segment_index, side
            )
            inner, outer = segment.plates[inner_index], segment.plates[outer_index]
            axis = "x" if side in ("ventral", "dorsal") else "y"
            self._add_spatial_tendon(
                name=f"segment_{segment_index}_vertebral_vertebral_strut_{side}",
                sites=[
                    vertebrae_s_tap,
                    inner.s_taps[f"connector-{axis}"],
                    inner.s_taps[f"plate-{axis}"],
                    outer.s_taps[f"plate-{axis}"],
                ],
                width=strut.beam_width,
                rgba=colors.STRUT,
                damping=strut.damping,
                stiffness=strut.stiffness,
            )

    # ------------------------------------------------------------------------------------------------------------------
    # Gliding joint couplings
    # ------------------------------------------------------------------------------------------------------------------
    def _couple(
        self, joint1: Optional[str], joint2: Optional[str], polycoef: List[int]
    ) -> None:
        if joint1 is not None and joint2 is not None:
            self.spec.add_equality(
                type=mujoco.mjtEq.mjEQ_JOINT,
                name1=joint1,
                name2=joint2,
                objtype=mujoco.mjtObj.mjOBJ_JOINT,
                data=polycoef + [0] * 7,
            )

    def _couple_gliding_joints_within_segment(self, plates: Sequence[_Plate]) -> None:
        """Plates that are neighbours along x move together along x and mirrored along y (and vice versa)."""
        for (index1, index2), x_coef, y_coef in (
            ((0, 1), _SAME, _OPPOSITE),
            ((2, 3), _SAME, _OPPOSITE),
            ((0, 3), _OPPOSITE, _SAME),
            ((1, 2), _OPPOSITE, _SAME),
        ):
            plate1, plate2 = plates[index1], plates[index2]
            self._couple(plate1.joints.get("x"), plate2.joints.get("x"), x_coef)
            self._couple(plate1.joints.get("y"), plate2.joints.get("y"), y_coef)

    def _couple_gliding_joints_between_segments(self) -> None:
        """Corresponding plates of neighbouring segments glide together."""
        for segment, next_segment in zip(self._segments, self._segments[1:]):
            for plate, next_plate in zip(segment.plates, next_segment.plates):
                self._couple(plate.joints.get("x"), next_plate.joints.get("x"), _SAME)
                self._couple(plate.joints.get("y"), next_plate.joints.get("y"), _SAME)

    # ------------------------------------------------------------------------------------------------------------------
    # Muscles
    # ------------------------------------------------------------------------------------------------------------------
    def _add_spatial_tendon(
        self,
        name: str,
        sites: Sequence[str],
        width: float,
        rgba: np.ndarray,
        damping: float,
        stiffness: float = 0.0,
    ) -> str:
        tendon = self.spec.add_tendon(
            name=name, width=width, rgba=rgba, damping=damping, stiffness=stiffness
        )
        for site in sites:
            tendon.wrap_site(site)
        return name

    def _build_hm_beams(self) -> List[str]:
        hm = (
            self.specification.beam_actuation_specification.hm_beam_actuation_specification
        )
        beams = []
        for routing in hm.routing_specifications:
            for x_side in ("ventral", "dorsal"):
                if x_side not in routing.sagittal_sides:
                    continue
                for y_side in ("dextral", "sinistral"):
                    if y_side not in routing.coronal_sides:
                        continue
                    start, end = routing.start_segment, routing.end_segment
                    taps = []
                    if start >= 0:
                        plate_index, _ = actuator_beam_plate_indices(x_side, start)
                        ghost_index = actuator_ghost_taps_index(x_side, y_side, start)
                        taps += (
                            self._segments[start]
                            .plates[plate_index]
                            .lower_ghost_hm_taps[ghost_index]
                        )
                    plate_index = self.specification.corners.index(f"{x_side}_{y_side}")
                    intermediate_points = iter(routing.intermediate_points)
                    for segment_index in range(start + 1, end):
                        tap_index = next(intermediate_points)
                        if segment_index >= 0:
                            plate = self._segments[segment_index].plates[plate_index]
                            taps += plate.intermediate_hm_taps[tap_index]
                    taps += self._segments[end].plates[plate_index].hm_tap_end
                    beams.append(
                        self._add_spatial_tendon(
                            name=f"hm_beam_{x_side}_{y_side}_{end}",
                            sites=taps,
                            width=hm.beam_width,
                            rgba=colors.HM_BEAM,
                            damping=hm.damping,
                        )
                    )
        return beams

    def _build_mvm_beams(self) -> List[str]:
        mvm = (
            self.specification.beam_actuation_specification.mvm_beam_actuation_specification
        )
        if not mvm.enabled:
            return []
        beams = []
        for index, (segment, next_segment) in enumerate(
            zip(self._segments, self._segments[1:])
        ):
            for side in ("sinistral", "dextral"):
                beams.append(
                    self._add_spatial_tendon(
                        name=f"mvm_beam_{side}_{index, index + 1}",
                        sites=[
                            segment.plates[1].mvm_taps[side][1],
                            next_segment.plates[1].mvm_taps[side][0],
                        ],
                        width=mvm.beam_width,
                        rgba=colors.MVM_BEAM,
                        damping=mvm.damping,
                    )
                )
        return beams

    def _add_beam_actuator(
        self,
        beam: str,
        kp: Optional[float],
        gear: Optional[float],
        ctrlrange: List[float],
    ) -> None:
        """A position (``kp``) or force (``gear``) actuator on ``beam`` that can only pull."""
        actuator = self.spec.add_actuator(
            name=beam,
            target=beam,
            trntype=mujoco.mjtTrn.mjTRN_TENDON,
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
            ctrlrange=ctrlrange,
            forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )
        if kp is not None:
            actuator.gainprm[0] = kp
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            actuator.biasprm[1] = -kp
            actuator.forcerange = [-kp, 0]
        else:
            actuator.gear[0] = gear
            actuator.forcerange = [-gear, 0]

    def _build_actuators(self, hm_beams: List[str], mvm_beams: List[str]) -> None:
        # Control ranges depend on the rest lengths of the beams.
        model = self.spec.compile()
        rest_length = {
            beam: float(model.tendon_length0[model.tendon(beam).id])
            for beam in hm_beams + mvm_beams
        }
        actuation = self.specification.beam_actuation_specification
        hm = actuation.hm_beam_actuation_specification
        for beam in hm_beams:
            if hm.p_control:
                self._add_beam_actuator(
                    beam,
                    kp=hm.p_control_kp,
                    gear=None,
                    ctrlrange=[0, rest_length[beam]],
                )
            else:
                self._add_beam_actuator(
                    beam, kp=None, gear=hm.f_control_gear, ctrlrange=[-1, 0]
                )
        mvm = actuation.mvm_beam_actuation_specification
        for beam in mvm_beams:
            self._add_beam_actuator(
                beam,
                kp=mvm.p_control_kp,
                gear=None,
                ctrlrange=[rest_length[beam] * mvm.strain, rest_length[beam]],
            )

        for beam in hm_beams + mvm_beams:
            self.spec.add_sensor(
                name=f"{beam}_position_sensor",
                type=mujoco.mjtSensor.mjSENS_TENDONPOS,
                objtype=mujoco.mjtObj.mjOBJ_TENDON,
                objname=beam,
            )
        for segment in self._segments:
            for joint in segment.vertebral_joints:
                self._add_joint_sensor(
                    joint, mujoco.mjtSensor.mjSENS_JOINTACTFRC, "actuatorfrc_sensor"
                )
        for beam in hm_beams + mvm_beams:
            self.spec.add_sensor(
                name=f"{beam}_actuatorfrc_sensor",
                type=mujoco.mjtSensor.mjSENS_ACTUATORFRC,
                objtype=mujoco.mjtObj.mjOBJ_ACTUATOR,
                objname=beam,
            )
