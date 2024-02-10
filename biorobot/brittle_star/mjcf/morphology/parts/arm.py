from typing import Union

import numpy as np
from moojoco.mjcf.morphology import MJCFMorphology, MJCFMorphologyPart

from biorobot.brittle_star.mjcf.morphology.parts.arm_segment import (
    MJCFBrittleStarArmSegment,
)
from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarMorphologySpecification,
)


class MJCFBrittleStarArm(MJCFMorphologyPart):
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

    def _build(self, arm_index: int) -> None:
        self._arm_index = arm_index
        self._arm_specification = self.morphology_specification.arm_specifications[
            self._arm_index
        ]

        self._build_segments()

    def _build_segments(self) -> None:
        self._segments = []

        number_of_segments = self._arm_specification.number_of_segments

        for segment_index in range(number_of_segments):
            try:
                parent = self._segments[-1]
                position = 2 * self._segments[-1].center_of_capsule
            except IndexError:
                position = np.zeros(3)
                parent = self

            segment = MJCFBrittleStarArmSegment(
                parent=parent,
                name=f"{self.base_name}_segment_{segment_index}",
                pos=position,
                euler=np.zeros(3),
                arm_index=self._arm_index,
                segment_index=segment_index,
            )
            self._segments.append(segment)
