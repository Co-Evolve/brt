"""Parameters of the in-silico seahorse tail (Marzougui et al. 2025, J. R. Soc. Interface 22: 20240876).

The model is a digital twin of the 3D-printed tail prototype of the paper: a chain of segments, each made of a vertebra
(with a ball bearing that forms the joint with the previous vertebra) surrounded by four bony plates. Neighbouring
plates are coupled by sliding (gliding) joints and every vertebra is connected to its plates by four elastic
vertebral struts. Two muscle groups can be added as spatial tendons ("beams"):

* **HM** (hypaxial myomere) beams: long muscle-tendon units that run from a start segment to the plate of an end
  segment and can span many segments. Their routing is given by :class:`HMMTendonRoutingSpecification`.
* **MVM** (median ventral muscle) beams: short intersegmental beams between the ventral-sinistral plates of every pair
  of neighbouring segments.

Units: SI (metres, kilograms, seconds, radians, newtons). The mesh files are in millimetres and are scaled by 0.001;
their masses (g), centres of mass (mm) and inertia tensors (g mm^2, about the centre of mass) are read from
``assets/meshes/inertia.json``.

Frames: the tail is built along +z of its root frame (segment 0 at the origin, every next segment ``SEGMENT_Z_OFFSET``
further). +x is ventral, +y sinistral (left), -x dorsal and -y dextral (right). Plates are indexed 0-3 as
ventral-dextral, ventral-sinistral, dorsal-sinistral and dorsal-dextral. The environment attaches the tail upside down
(rotated by pi around x), so that it hangs downwards from segment 0.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

MESH_DIRECTORY = Path(__file__).parent / "assets" / "meshes"

PLATE_INDEX_TO_SIDE = [
    "ventral_dextral",
    "ventral_sinistral",
    "dorsal_sinistral",
    "dorsal_dextral",
]

VERTEBRAE_MESH_NAME = "vertebrae.stl"
BALL_BEARING_MESH_NAME = "ball_bearing.stl"
VERTEBRAE_CONNECTOR_MESH_NAME = "connector_vertebrae.stl"
PLATE_CONNECTOR_MESH_NAME = "connector_plate.stl"

#: Distance (m) from the vertebra's centre to the end of its bars, and length (m) of a vertebral connector.
VERTEBRAE_OFFSET_TO_BAR_END = 0.013 + 0.003
VERTEBRAE_CONNECTOR_LENGTH = 0.0036
#: Distance (m) from the vertebra's centre to the vertebral strut attachment points on the vertebra.
VERTEBRAE_S_TAP_OFFSET = 0.015
#: Distance (m) along the tail between the vertebra and its ball bearing (the joint is halfway, at -this value).
VERTEBRAE_BALL_BEARING_Z_OFFSET = 0.016

#: Distance (m) from the vertebra's centre to each plate's origin (in the plate's diagonal direction).
PLATE_MESH_NAME_TO_OFFSET_FROM_VERTEBRAE = {
    "ventral_dextral.stl": 0.05098,
    "ventral_sinistral.stl": 0.05119,
    "dorsal_sinistral_even.stl": 0.0512,
    "dorsal_sinistral_odd.stl": 0.0512,
    "dorsal_dextral_even.stl": 0.05145,
    "dorsal_dextral_odd.stl": 0.0512,
}
#: (x, y) offset (m) of the HM ghost attachment points from the plate's origin.
PLATE_MESH_NAME_TO_HMM_GHOST_TAP_OFFSET = {
    "ventral_sinistral.stl": [0.00219, -0.02981],
    "ventral_dextral.stl": [0.00235, 0.0304],
    "dorsal_sinistral_even.stl": [-0.00224, -0.02986],
    "dorsal_dextral_even.stl": [-0.00202, 0.03072],
    "dorsal_sinistral_odd.stl": [-0.00219, -0.03061],
    "dorsal_dextral_odd.stl": [-0.00219, 0.0298],
}
#: Number of intermediate HM attachment points (routing holes) per plate and their spacing (m).
PLATE_HMM_NUM_INTERMEDIATE_TAPS = 10
PLATE_HMM_Y_OFFSET_BETWEEN_INTERMEDIATE_TAPS = 0.00273
#: (x, y) offset (m) of the first (most lateral) intermediate HM attachment point from the plate's origin.
PLATE_MESH_NAME_TO_FIRST_HMM_INTERMEDIATE_TAP_OFFSET = {
    "ventral_sinistral.stl": [-0.01, -0.00273],
    "ventral_dextral.stl": [-0.01, 0.00273],
    "dorsal_sinistral_even.stl": [0.01, -0.00273],
    "dorsal_dextral_even.stl": [0.01, 0.00273],
    "dorsal_sinistral_odd.stl": [0.01, -0.00273],
    "dorsal_dextral_odd.stl": [0.01, 0.00273],
}
MVM_TAP_OFFSET = [0.0435, 0.004]
#: Distance (m) from the vertebra's centre to the vertebral strut attachment points on the outer / inner plates.
OUTER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE = 0.052
INNER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE = 0.048

PLATE_CONNECTOR_LENGTH = 0.010
VERTEBRAE_TO_PLATE_CONNECTOR_HOLE = 0.04101

PLATE_DEPTH = 0.015
#: Total travel (m) of a gliding joint; every gliding joint can slide half of it in one direction.
PLATE_GLIDING_JOINT_RANGE = 0.01
#: Distance (m) between two consecutive vertebrae.
SEGMENT_Z_OFFSET = 0.032


def _load_inertia_values() -> Dict[str, dict]:
    with open(MESH_DIRECTORY / "inertia.json") as file:
        return json.load(file)


MESH_NAME_TO_INERTIA_VALUES = _load_inertia_values()


@dataclass
class MeshSpecification:
    """A mesh and the inertial properties of the part it represents.

    Args:
        mesh_path: Path of the STL file.
        scale: Scale of the mesh vertices (0.001: the files are in millimetres).
        mass: Mass (kg).
        center_of_mass: Centre of mass (m) in the mesh frame.
        fullinertia: Inertia tensor (kg m^2) about the centre of mass as ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)``.
    """

    mesh_path: str
    scale: np.ndarray
    mass: float
    center_of_mass: np.ndarray
    fullinertia: np.ndarray


@dataclass
class JointSpecification:
    """A joint's passive properties.

    Args:
        stiffness: Spring stiffness (N m/rad or N/m).
        damping: Damping (N m s/rad or N s/m).
        friction_loss: Dry friction (N m or N).
        range: Range of motion (rad or m), see the joint's documentation for how it is applied. A range of 0 removes
            the joint.
        armature: Armature (rotor) inertia (kg m^2 or kg).
    """

    stiffness: float
    damping: float
    friction_loss: float
    range: float
    armature: float


@dataclass
class SeahorsePlateSpecification:
    """One of the four bony plates around a vertebra.

    Offsets with ``_from_vertebrae`` are measured from the vertebra's centre, offsets with ``_from_plate_origin`` from
    the origin of the plate mesh. ``tap`` is short for tendon attachment point.

    Args:
        plate_mesh_specification: The plate.
        connector_mesh_specification: The connector between the plate and the vertebral strut (inner plates only).
        offset_from_vertebrae: Distance (m) to the plate's origin, along the plate's diagonal.
        depth: Plate thickness (m) along the tail.
        connector_offset_from_vertebrae: Distance (m) to the connector.
        s_tap_x_offset_from_vertebrae: Distance (m) to the vertebral strut attachment point along x.
        s_tap_y_offset_from_vertebrae: Distance (m) to the vertebral strut attachment point along y.
        hm_ghost_tap_x_offset_from_plate_origin: x-offset (m) of the HM start attachment points.
        hm_ghost_tap_y_offset_from_plate_origin: y-offset (m) of the HM start attachment points.
        hm_num_intermediate_taps: Number of intermediate HM attachment points (routing holes).
        hm_y_offset_between_intermediate_taps: Spacing (m) between the intermediate HM attachment points.
        hm_intermediate_first_tap_x_offset_from_plate_origin: x-offset (m) of the first intermediate HM attachment
            point.
        hm_intermediate_first_tap_y_offset_from_plate_origin: y-offset (m) of the first intermediate HM attachment
            point.
        mvm_tap_x_offset: Unused by the model (kept from the original specification).
        mvm_tap_z_offset: Unused by the model (kept from the original specification).
        x_axis_gliding_joint_specification: Gliding joint along x with the previous segment's plate (only its
            ``range``, ``stiffness`` and ``damping`` are used, as in the original model).
        y_axis_gliding_joint_specification: Gliding joint along y (idem).
    """

    plate_mesh_specification: MeshSpecification
    connector_mesh_specification: MeshSpecification
    offset_from_vertebrae: float
    depth: float
    connector_offset_from_vertebrae: float
    s_tap_x_offset_from_vertebrae: float
    s_tap_y_offset_from_vertebrae: float
    hm_ghost_tap_x_offset_from_plate_origin: float
    hm_ghost_tap_y_offset_from_plate_origin: float
    hm_num_intermediate_taps: int
    hm_y_offset_between_intermediate_taps: float
    hm_intermediate_first_tap_x_offset_from_plate_origin: float
    hm_intermediate_first_tap_y_offset_from_plate_origin: float
    mvm_tap_x_offset: float
    mvm_tap_z_offset: float
    x_axis_gliding_joint_specification: JointSpecification
    y_axis_gliding_joint_specification: JointSpecification


@dataclass
class SeahorseVertebraeSpecification:
    """A vertebra, its ball bearing and its connectors.

    The pitch (around y, ventral bending), roll (around x, lateral bending) and yaw (around z, torsion) joints connect
    the vertebra to the previous one at the centre of the ball bearing; their range is symmetric: ``(-range, range)``.
    The yaw joint has no armature (as in the original model).

    Args:
        z_offset_to_ball_bearing: Distance (m) along the tail between the vertebra and its ball bearing.
        offset_to_vertebral_strut_attachment_point: Distance (m) to the vertebral strut attachment points.
        connector_length: Length (m) of a connector.
        offset_to_bar_end: Distance (m) to the end of the vertebra's bars.
    """

    z_offset_to_ball_bearing: float
    offset_to_vertebral_strut_attachment_point: float
    connector_length: float
    offset_to_bar_end: float
    vertebral_mesh_specification: MeshSpecification
    ball_bearing_mesh_specification: MeshSpecification
    connector_mesh_specification: MeshSpecification
    yaw_joint_specification: JointSpecification
    pitch_joint_specification: JointSpecification
    roll_joint_specification: JointSpecification


@dataclass
class SeahorseVertebralStrutSpecification:
    """The elastic struts between a vertebra and its plates (spatial tendons; rest length is the initial length).

    Args:
        stiffness: Tendon stiffness (N/m).
        damping: Tendon damping (N s/m).
        beam_width: Rendered width (m).
    """

    stiffness: float
    damping: float
    beam_width: float


@dataclass
class SeahorseSegmentSpecification:
    """A segment: a vertebra, its four plates and four vertebral struts.

    Args:
        z_offset_from_previous_segment: Distance (m) to the previous segment along the tail (0 for the first one).
    """

    z_offset_from_previous_segment: float
    vertebrae_specification: SeahorseVertebraeSpecification
    vertebral_strut_specification: SeahorseVertebralStrutSpecification
    plate_specifications: List[SeahorsePlateSpecification]


@dataclass
class HMMTendonRoutingSpecification:
    """The route of one or more HM beams.

    One beam is created per combination of ``sagittal_sides`` (``"ventral"``/``"dorsal"``) and ``coronal_sides``
    (``"dextral"``/``"sinistral"``). A beam starts at the ghost attachment points of ``start_segment`` (a negative
    start segment means that the beam starts beyond the base of the tail: the missing segments are skipped), passes one
    pair of intermediate attachment points on the plate of every segment strictly between start and end, and ends at
    the plate of ``end_segment``.

    Args:
        start_segment: Index of the start segment (may be negative).
        end_segment: Index of the end segment.
        intermediate_points: Intermediate attachment point index (0 is the most lateral one, 9 the most medial one)
            for every segment strictly between start and end, in order from the start segment.
        sagittal_sides: Sides of the plate (``"ventral"``, ``"dorsal"``).
        coronal_sides: Sides of the plate (``"dextral"``, ``"sinistral"``).
    """

    start_segment: int
    end_segment: int
    intermediate_points: List[int]
    sagittal_sides: List[str]
    coronal_sides: List[str]


@dataclass
class SeahorseHMMTendonActuationSpecification:
    """The HM beams and their actuators.

    With position control (``p_control``), the action of a beam is its target length (m) in ``[0, rest length]`` and
    the actuator force is ``p_control_kp * (target - length)``, limited to ``[-p_control_kp, 0]`` (beams can only
    pull). With force control, the action is in ``[-1, 0]`` and the beam force is ``f_control_gear * action`` (N).

    Args:
        p_control: Position (True) or force (False) control.
        p_control_kp: Position gain (N/m).
        f_control_gear: Maximum contraction force (N) under force control.
        beam_width: Rendered width (m).
        segment_span: Number of segments the beams span. Informational: the routing specifications define the beams
            (see :func:`default_hm_routing_specification`).
        damping: Beam damping (N s/m).
        routing_specifications: The beams; none by default.
        strain: Unused by the model (kept from the original specification).
    """

    p_control: bool
    p_control_kp: float
    f_control_gear: float
    beam_width: float
    segment_span: int
    damping: float
    routing_specifications: List[HMMTendonRoutingSpecification]
    strain: float


@dataclass
class SeahorseMVMTendonActuationSpecification:
    """The MVM beams (always position controlled) and their actuators.

    The action of a beam is its target length (m) in ``[strain * rest length, rest length]``; the force is
    ``p_control_kp * (target - length)`` limited to ``[-p_control_kp, 0]``.

    Args:
        enabled: Add one MVM beam per side (sinistral, dextral) between every pair of neighbouring segments.
        p_control_kp: Position gain (N/m).
        beam_width: Rendered width (m).
        damping: Beam damping (N s/m).
        strain: Shortest target length as a fraction of the rest length.
    """

    enabled: bool
    p_control_kp: float
    beam_width: float
    damping: float
    strain: float


@dataclass
class SeahorseTendonActuationSpecification:
    hm_beam_actuation_specification: SeahorseHMMTendonActuationSpecification
    mvm_beam_actuation_specification: SeahorseMVMTendonActuationSpecification


@dataclass
class SeahorseMorphologySpecification:
    """The complete tail: its segments (from the base to the tip) and its muscle architecture."""

    segment_specifications: List[SeahorseSegmentSpecification]
    beam_actuation_specification: SeahorseTendonActuationSpecification

    sides = ["ventral", "sinistral", "dorsal", "dextral"]
    corners = PLATE_INDEX_TO_SIDE

    @property
    def num_segments(self) -> int:
        return len(self.segment_specifications)


def default_mesh_specification(*, mesh_name: str) -> MeshSpecification:
    """The mesh ``mesh_name`` (a file in ``assets/meshes``) with the inertial values of ``inertia.json``.

    The base plates (``*_base.stl``, with an extra flange at the first segment) use the values of the regular plates.
    """
    inertia_values = MESH_NAME_TO_INERTIA_VALUES[mesh_name.replace("_base", "")]
    indices = [[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]
    fullinertia = [inertia_values["inertia_matrix"][x][y] for x, y in indices]
    return MeshSpecification(
        mesh_path=str(MESH_DIRECTORY / mesh_name),
        scale=0.001 * np.ones(3),  # mm to m
        mass=inertia_values["mass"] / 1000,  # g to kg
        center_of_mass=np.array(inertia_values["center_of_mass"]) / 1000,  # mm to m
        fullinertia=np.array(fullinertia) / 1e9,  # g mm^2 to kg m^2
    )


def default_gliding_joint_specification(
    *, plate_index: int, axis: str
) -> JointSpecification:
    return JointSpecification(
        stiffness=1,
        damping=1,
        friction_loss=0.3,
        armature=0.045,
        range=PLATE_GLIDING_JOINT_RANGE / 2,
    )


def is_inner_plate_x_axis(segment_index: int, plate_index: int) -> bool:
    """Whether the plate overlaps its x-neighbour on the inside (plates alternate between segments)."""
    if plate_index == 0:
        return False
    if plate_index == 1:
        return True
    if segment_index % 2 == 0:
        return plate_index == 2
    return plate_index != 2


def is_inner_plate_y_axis(segment_index: int, plate_index: int) -> bool:
    """Whether the plate overlaps its y-neighbour on the inside (the dorsal plates)."""
    return plate_index >= 2


def default_seahorse_plate_specification(
    *, segment_index: int, plate_index: int
) -> SeahorsePlateSpecification:
    side = PLATE_INDEX_TO_SIDE[plate_index]

    if side.startswith("ventral"):
        plate_mesh_name = side
    else:
        alternator = "even" if segment_index % 2 == 0 else "odd"
        plate_mesh_name = f"{side}_{alternator}"
    if segment_index == 0:
        plate_mesh_name += "_base"
    plate_mesh_name += ".stl"

    plate_mesh_specification = default_mesh_specification(mesh_name=plate_mesh_name)
    plate_mesh_name = plate_mesh_name.replace("_base", "")
    connector_mesh_specification = default_mesh_specification(
        mesh_name=PLATE_CONNECTOR_MESH_NAME
    )

    if is_inner_plate_x_axis(segment_index=segment_index, plate_index=plate_index):
        s_tap_x_offset = INNER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE
    else:
        s_tap_x_offset = OUTER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE
    if is_inner_plate_y_axis(segment_index=segment_index, plate_index=plate_index):
        s_tap_y_offset = INNER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE
    else:
        s_tap_y_offset = OUTER_PLATE_S_TAP_OFFSET_FROM_VERTEBRAE

    ghost_x, ghost_y = PLATE_MESH_NAME_TO_HMM_GHOST_TAP_OFFSET[plate_mesh_name]
    intermediate_x, intermediate_y = (
        PLATE_MESH_NAME_TO_FIRST_HMM_INTERMEDIATE_TAP_OFFSET[plate_mesh_name]
    )
    return SeahorsePlateSpecification(
        plate_mesh_specification=plate_mesh_specification,
        connector_mesh_specification=connector_mesh_specification,
        offset_from_vertebrae=PLATE_MESH_NAME_TO_OFFSET_FROM_VERTEBRAE[plate_mesh_name],
        depth=PLATE_DEPTH,
        connector_offset_from_vertebrae=VERTEBRAE_TO_PLATE_CONNECTOR_HOLE
        - 2 / 5 * PLATE_CONNECTOR_LENGTH,
        s_tap_x_offset_from_vertebrae=s_tap_x_offset,
        s_tap_y_offset_from_vertebrae=s_tap_y_offset,
        hm_ghost_tap_x_offset_from_plate_origin=ghost_x,
        hm_ghost_tap_y_offset_from_plate_origin=ghost_y,
        hm_num_intermediate_taps=PLATE_HMM_NUM_INTERMEDIATE_TAPS,
        hm_y_offset_between_intermediate_taps=PLATE_HMM_Y_OFFSET_BETWEEN_INTERMEDIATE_TAPS,
        hm_intermediate_first_tap_x_offset_from_plate_origin=intermediate_x,
        hm_intermediate_first_tap_y_offset_from_plate_origin=intermediate_y,
        mvm_tap_x_offset=MVM_TAP_OFFSET[0],
        mvm_tap_z_offset=MVM_TAP_OFFSET[1],
        x_axis_gliding_joint_specification=default_gliding_joint_specification(
            plate_index=plate_index, axis="x"
        ),
        y_axis_gliding_joint_specification=default_gliding_joint_specification(
            plate_index=plate_index, axis="y"
        ),
    )


def default_seahorse_vertebrae_specification() -> SeahorseVertebraeSpecification:
    return SeahorseVertebraeSpecification(
        z_offset_to_ball_bearing=VERTEBRAE_BALL_BEARING_Z_OFFSET,
        offset_to_vertebral_strut_attachment_point=VERTEBRAE_S_TAP_OFFSET,
        offset_to_bar_end=VERTEBRAE_OFFSET_TO_BAR_END,
        connector_length=VERTEBRAE_CONNECTOR_LENGTH,
        vertebral_mesh_specification=default_mesh_specification(
            mesh_name=VERTEBRAE_MESH_NAME
        ),
        ball_bearing_mesh_specification=default_mesh_specification(
            mesh_name=BALL_BEARING_MESH_NAME
        ),
        connector_mesh_specification=default_mesh_specification(
            mesh_name=VERTEBRAE_CONNECTOR_MESH_NAME
        ),
        roll_joint_specification=JointSpecification(
            stiffness=0.0001,
            damping=0.00001,
            friction_loss=0.0015,
            armature=0.01,
            range=14 / 180 * np.pi,
        ),
        pitch_joint_specification=JointSpecification(
            stiffness=0.0001,
            damping=0.00001,
            friction_loss=0.0015,
            armature=0.01,
            range=14 / 180 * np.pi,
        ),
        yaw_joint_specification=JointSpecification(
            stiffness=0.5,
            damping=0.1,
            friction_loss=0.0015,
            armature=0.045,
            range=5 / 180 * np.pi,
        ),
    )


def default_seahorse_vertebral_strut_specification() -> (
    SeahorseVertebralStrutSpecification
):
    return SeahorseVertebralStrutSpecification(
        stiffness=3.043, damping=1.0, beam_width=0.0005
    )


def default_seahorse_segment_specification(
    *, segment_index: int
) -> SeahorseSegmentSpecification:
    return SeahorseSegmentSpecification(
        z_offset_from_previous_segment=0 if segment_index == 0 else SEGMENT_Z_OFFSET,
        vertebrae_specification=default_seahorse_vertebrae_specification(),
        plate_specifications=[
            default_seahorse_plate_specification(
                segment_index=segment_index, plate_index=plate_index
            )
            for plate_index in range(4)
        ],
        vertebral_strut_specification=default_seahorse_vertebral_strut_specification(),
    )


def default_beam_actuation_specification(
    hm_segment_span: int,
    mvm_enabled: bool,
    p_control: bool,
    hm_strain: float,
    mvm_strain: float,
) -> SeahorseTendonActuationSpecification:
    return SeahorseTendonActuationSpecification(
        hm_beam_actuation_specification=SeahorseHMMTendonActuationSpecification(
            p_control=p_control,
            p_control_kp=1000,
            beam_width=0.001,
            segment_span=hm_segment_span,
            damping=0.01,
            f_control_gear=10,
            routing_specifications=[],
            strain=hm_strain,
        ),
        mvm_beam_actuation_specification=SeahorseMVMTendonActuationSpecification(
            enabled=mvm_enabled,
            beam_width=0.001,
            p_control_kp=1000,
            damping=1,
            strain=mvm_strain,
        ),
    )


def default_seahorse_morphology_specification(
    *,
    num_segments: int,
    hm_segment_span: int,
    p_control: bool = True,
    mvm_enabled: bool = True,
    mvm_strain: float = 0.0,
    hm_strain: float = 0.0,
) -> SeahorseMorphologySpecification:
    """The prototype's tail with ``num_segments`` segments.

    No HM beams are created until routing specifications are added (e.g. with
    :func:`default_hm_routing_specification`); ``hm_segment_span`` is only stored.
    """
    return SeahorseMorphologySpecification(
        segment_specifications=[
            default_seahorse_segment_specification(segment_index=segment_index)
            for segment_index in range(num_segments)
        ],
        beam_actuation_specification=default_beam_actuation_specification(
            hm_segment_span=hm_segment_span,
            mvm_enabled=mvm_enabled,
            p_control=p_control,
            hm_strain=hm_strain,
            mvm_strain=mvm_strain,
        ),
    )


def default_hm_routing_specification(
    *,
    num_segments: int,
    segment_span: int,
    sagittal_sides: Sequence[str] = ("ventral",),
    coronal_sides: Sequence[str] = ("sinistral", "dextral"),
    end_segment: Optional[int] = None,
    intermediate_points: Optional[Sequence[int]] = None,
) -> HMMTendonRoutingSpecification:
    """HM beams that span ``segment_span`` segments and end at ``end_segment`` (default: the tip).

    By default, the intermediate attachment points follow the routing of the paper's HM experiment: the segments
    from start to end use the points ``end_segment - 1, end_segment - 2, ...`` (clipped to the available points 0-9),
    i.e. a beam that spans the whole 11-segment tail runs diagonally from the medial ghost points at its start to the
    lateral end point. Pass ``intermediate_points`` (one per segment strictly between start and end, including
    negative ones) for another routing.
    """
    if end_segment is None:
        end_segment = num_segments - 1
    start_segment = end_segment - segment_span
    if intermediate_points is None:
        intermediate_points = [
            int(np.clip(end_segment - 1 - k, 0, PLATE_HMM_NUM_INTERMEDIATE_TAPS - 1))
            for k in range(max(segment_span - 1, 0))
        ]
    return HMMTendonRoutingSpecification(
        start_segment=start_segment,
        end_segment=end_segment,
        intermediate_points=list(intermediate_points),
        sagittal_sides=list(sagittal_sides),
        coronal_sides=list(coronal_sides),
    )
