from typing import Union

import numpy as np
from moojoco.mjcf.morphology import MJCFMorphology, MJCFMorphologyPart

from biorobot.toy_example.mjcf.morphology.specification.specification import (
    ToyExampleMorphologySpecification,
)
from biorobot.utils.colors import rgba_green


class MJCFToyExampleTorso(MJCFMorphologyPart):
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
    def morphology_specification(self) -> ToyExampleMorphologySpecification:
        return super().morphology_specification

    def _build(
        self,
    ) -> None:
        self._torso_specification = self.morphology_specification.torso_specification
        self._build_torso()
        self._configure_sensors()

    def _build_torso(self) -> None:
        self.mjcf_body.add(
            "geom",
            name=f"{self.base_name}_sphere",
            type="sphere",
            size=[self._torso_specification.radius.value],
            rgba=rgba_green,
        )

    def _configure_sensors(self) -> None:
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
