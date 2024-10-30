from typing import Union

import numpy as np
from moojoco.mjcf.morphology import MJCFMorphology, MJCFMorphologyPart
from scipy.spatial.transform import Rotation as R

from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarMorphologySpecification,
)
from biorobot.utils import colors
from biorobot.utils.colors import rgba_red


class MJCFBrittleStarDisk(MJCFMorphologyPart):
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

    def _build(self) -> None:
        self._disk_specification = self.morphology_specification.disk_specification

        self._build_pentagon()
        self._build_arm_connections()
        self._configure_tendon_attachment_points()

    def _build_pentagon(self) -> None:
        # Todo: replace this with a dynamically generated mesh
        radius = self.morphology_specification.disk_specification.radius.value
        height = self.morphology_specification.disk_specification.height.value
        # box size
        arm_angle_delta = np.pi / 5
        box_y_size = radius * np.sin(arm_angle_delta)
        box_x_size = radius * np.cos(arm_angle_delta) / 2

        angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        angles += angles[1] / 2

        for i, angle in enumerate(angles):
            pos = box_x_size * np.array([np.cos(angle), np.sin(angle), 0.0])
            self.mjcf_body.add(
                "geom",
                type="box",
                name=f"{self.base_name}_pentagon_side_{i}",
                pos=pos,
                euler=[0, 0, angle],
                size=[box_x_size, box_y_size, height],
                rgba=colors.rgba_green,
                contype=0,
                conaffinity=0,
            )

        self.mjcf_body.add(
            "geom",
            type="box",
            name=f"{self.base_name}_pentagon_collider",
            pos=[0.0, 0.0, 0.0],
            euler=[0, 0, 0],
            size=[radius * 0.55, radius * 0.55, height],
            rgba=colors.rgba_green,
            contype=1,
            conaffinity=0,
        )

    def _build_arm_connections(self) -> None:
        radius = self.morphology_specification.disk_specification.radius.value
        height = self.morphology_specification.disk_specification.height.value

        arm_angles = np.linspace(0, 2 * np.pi, 6)[:-1]

        connector_length = radius * 0.2
        for i, angle in enumerate(arm_angles):
            pos = (radius - connector_length) * np.array(
                [np.cos(angle), np.sin(angle), 0.0]
            )
            self.mjcf_body.add(
                "geom",
                type="box",
                name=f"{self.base_name}_arm_connector_{i}",
                pos=pos,
                euler=[0.0, 0.0, angle],
                size=[connector_length, height, height],
                rgba=colors.rgba_green,
                contype=0,
                conaffinity=0,
            )

    def _configure_tendon_attachment_points(self) -> None:
        if self.morphology_specification.actuation_specification.use_tendons.value:
            self.distal_taps = []

            disk_radius = self.morphology_specification.disk_specification.radius.value
            center_pos = np.array([disk_radius, 0, 0])

            arm_angles = np.linspace(0, 2 * np.pi, 6)[:-1]
            tap_angles = np.linspace(np.pi / 4, 7 * np.pi / 4, 4)

            for arm_index, arm_angle in enumerate(arm_angles):
                arm_specification = self.morphology_specification.arm_specifications[arm_index]
                if arm_specification.number_of_segments == 0:
                    continue

                base_segment_radius = arm_specification.segment_specifications[0].radius.value

                arm_taps = []
                positions = []
                for angle in tap_angles:
                    pos = center_pos + 0.8 * base_segment_radius * np.array([0, np.cos(angle), np.sin(angle)])
                    positions.append(pos)

                for tap_index, position in enumerate(positions):
                    # rotate position around arm_angle degress
                    # Define the rotation
                    rotation = R.from_euler('z', arm_angle, degrees=False)

                    # Rotate point A around point B
                    rotated_point = rotation.apply(position)

                    arm_taps.append(self.mjcf_body.add("site",
                                                       name=f"{self.base_name}_arm_{arm_index}_tap_{tap_index}",
                                                       type="sphere",
                                                       rgba=rgba_red,
                                                       pos=rotated_point,
                                                       size=[0.001]))
                self.distal_taps.append(arm_taps)
