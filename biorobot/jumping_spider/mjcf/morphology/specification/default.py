from typing import List, Tuple

import numpy as np

from biorobot.jumping_spider.mjcf.morphology.specification.specification import JumpingSpiderJointSpecification, \
    JumpingSpiderLegSegmentSpecification, JumpingSpiderLegSpecification, JumpingSpiderMorphologySpecification, \
    JumpingSpiderAbdomenSpecification, JumpingSpiderCephalothoraxSpecification, JumpingSpiderDraglineSpecification, \
    JumpingSpiderActuationSpecification

CONNECTOR_RADIUS = .01333
LEG_SEGMENT_NAMES = ["coxa", "trochanter", "femur", "patella", "tibia", "metatarsus", "tarsus"]
LEG_SEGMENT_LENGTHS = (0.116, 0.068, 0.246, 0.127, 0.176, 0.143, 0.118)
LEG_SEGMENT_RADII = (0.0585, 0.0415, 0.0513, 0.0425, 0.047, 0.0275, 0.0285)
LEG_SEGMENT_ATTACHMENT_ANGLES = np.array([-21, 0, -11, 40, 56, 23, -53]) / 180 * np.pi
LEG_SEGMENT_OOP_LOWER_ROMS = np.array([-13, 0, -46, -40, -20, -96, -31]) / 180 * np.pi
LEG_SEGMENT_OOP_UPPER_ROMS = np.array([39, 60, 56, 107, 0, 27, 91]) / 180 * np.pi
LEG_SEGMENT_IP_ROMS = np.array([35, 70, 0, 0, 70, 15, 65]) / 180 * np.pi


def default_joint_specification(
        upper_range: float,
        lower_range: float,
) -> JumpingSpiderJointSpecification:
    joint_specification = JumpingSpiderJointSpecification(
        lower_range=lower_range,
        upper_range=upper_range,
        stiffness=0.05,
        damping=0.05,
        armature=0.01
    )
    return joint_specification


def default_leg_segment_specifications() -> List[
    JumpingSpiderLegSegmentSpecification]:
    leg_segment_specifications = []
    for length, radius, attachment_angle, oop_lower_rom, oop_upper_rom, ip_rom in zip(LEG_SEGMENT_LENGTHS,
                                                                                      LEG_SEGMENT_RADII,
                                                                                      LEG_SEGMENT_ATTACHMENT_ANGLES,
                                                                                      LEG_SEGMENT_OOP_LOWER_ROMS,
                                                                                      LEG_SEGMENT_OOP_UPPER_ROMS,
                                                                                      LEG_SEGMENT_IP_ROMS):
        leg_segment_specifications.append(
            JumpingSpiderLegSegmentSpecification(
                ip_joint_specification=default_joint_specification(lower_range=-ip_rom / 2,
                                                                   upper_range=ip_rom / 2),
                oop_joint_specification=default_joint_specification(lower_range=oop_lower_rom,
                                                                    upper_range=oop_upper_rom),
                torque_limit=1.0,
                length=length,
                radius=radius,
                oop_attachment_angle=attachment_angle
            )
        )
    return leg_segment_specifications


def default_leg_specification(
        in_plane_connection_angle: float
) -> JumpingSpiderLegSpecification:
    segment_specifications = default_leg_segment_specifications()

    leg_specification = JumpingSpiderLegSpecification(
        in_plane_connection_angle=in_plane_connection_angle,
        out_of_plane_connection_angle=-40 / 180 * np.pi,
        segment_specifications=segment_specifications
    )
    return leg_specification


def default_jumping_spider_specification(
        cephalothorax_size: Tuple[float] = (0.3, 0.3, 0.25),
        abdomen_size: Tuple[float] = (0.5, 0.4, 0.3),
        dragline: bool = False,
        position_control: bool = False
) -> JumpingSpiderMorphologySpecification:
    cephalothorax_specification = JumpingSpiderCephalothoraxSpecification(size_x=cephalothorax_size[0],
                                                                          size_y=cephalothorax_size[1],
                                                                          size_z=cephalothorax_size[2],
                                                                          leg_connector_radius=CONNECTOR_RADIUS)

    abdomen_specification = JumpingSpiderAbdomenSpecification(size_x=abdomen_size[0], size_y=abdomen_size[1],
                                                              size_z=abdomen_size[2],
                                                              inset_factor=0.2,
                                                              joint_specification=default_joint_specification(
                                                                  upper_range=45 / 180 * np.pi,
                                                                  lower_range=45 / 180 * np.pi),
                                                              torque_limit=1.0)

    leg_specifications = list()
    for in_plane_connection_angle in np.linspace(30, 130, 4) / 180 * np.pi:
        leg_specification = default_leg_specification(in_plane_connection_angle=in_plane_connection_angle)
        leg_specifications.append(leg_specification)

    dragline_specification = JumpingSpiderDraglineSpecification(stiffness=0.0, damping=1.0, enabled=dragline)

    actuation_specification = JumpingSpiderActuationSpecification(position_control=position_control, kp=200)

    specification = JumpingSpiderMorphologySpecification(
        cephalothorax_specification=cephalothorax_specification,
        leg_specifications=leg_specifications,
        abdomen_specification=abdomen_specification,
        dragline_specification=dragline_specification,
        actuation_specification=actuation_specification
    )

    return specification
