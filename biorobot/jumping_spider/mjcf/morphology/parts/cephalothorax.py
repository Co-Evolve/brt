from typing import Union

import numpy as np
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from biorobot.jumping_spider.mjcf.morphology.specification.specification import JumpingSpiderMorphologySpecification
from biorobot.utils import colors


class MJCFJumpingSpiderCephalothorax(MJCFMorphologyPart):
    def __init__(self, parent: Union[MJCFMorphology, MJCFMorphologyPart], name: str, pos: np.array, euler: np.array,
                 *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> JumpingSpiderMorphologySpecification:
        return super().morphology_specification

    def _build(self) -> None:
        self._cephalothorax_specification = self.morphology_specification.cephalothorax_specification
        self._build_ellipsoid()
        self._configure_sensors()

    def _build_ellipsoid(self) -> None:
        size_x = self._cephalothorax_specification.size_x.value
        size_y = self._cephalothorax_specification.size_y.value
        size_z = self._cephalothorax_specification.size_z.value

        self._disc = self.mjcf_body.add("geom",
                                        name=f"{self.base_name}_ellipsoid",
                                        type=f"ellipsoid",
                                        pos=np.zeros(3),
                                        euler=np.zeros(3),
                                        size=[size_x, size_y, size_z],
                                        density=20,
                                        rgba=colors.rgba_green,
                                        friction=[0.001, 0.1, 0.1],
                                        )

    def _configure_sensors(self) -> None:
        self.mjcf_model.sensor.add("framepos",
                                   name=f"{self.base_name}_framepos",
                                   objtype="body",
                                   objname=self._name)
        self.mjcf_model.sensor.add("framequat",
                                   name=f"{self.base_name}_framequat",
                                   objtype="body",
                                   objname=self._name)
        self.mjcf_model.sensor.add("framelinvel",
                                   name=f"{self.base_name}_framelinvel",
                                   objtype="body",
                                   objname=self._name)
        self.mjcf_model.sensor.add("frameangvel",
                                   name=f"{self.base_name}_frameangvel",
                                   objtype="body",
                                   objname=self._name)
