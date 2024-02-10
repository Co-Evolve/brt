from typing import Union

import numpy as np
from moojoco.mjcf.morphology import MJCFMorphology, MJCFMorphologyPart

from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarMorphologySpecification,
)
from biorobot.utils import colors


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
        self._configure_sensors()

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

        # connector_length = radius / 2
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
                size=[connector_length, 1.1 * height, height],
                rgba=colors.rgba_green,
                contype=0,
                conaffinity=0,
            )

    def _configure_sensors(self) -> None:
        self.mjcf_model.sensor.add(
            "framepos",
            name=f"{self.base_name}_framepos",
            objtype="body",
            objname=self._name,
        )
        self.mjcf_model.sensor.add(
            "framequat",
            name=f"{self.base_name}_framequat",
            objtype="body",
            objname=self._name,
        )
        self.mjcf_model.sensor.add(
            "framelinvel",
            name=f"{self.base_name}_framelinvel",
            objtype="body",
            objname=self._name,
        )
        self.mjcf_model.sensor.add(
            "frameangvel",
            name=f"{self.base_name}_frameangvel",
            objtype="body",
            objname=self._name,
        )
