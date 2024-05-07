from typing import Union

import numpy as np
from moojoco.mjcf.morphology import MJCFMorphologyPart, MJCFMorphology

from biorobot.jumping_spider.mjcf.morphology.parts.leg_segment import MJCFJumpingSpiderLegSegment
from biorobot.jumping_spider.mjcf.morphology.specification.default import LEG_SEGMENT_NAMES
from biorobot.jumping_spider.mjcf.morphology.specification.specification import JumpingSpiderMorphologySpecification


class MJCFJumpingSpiderLeg(MJCFMorphologyPart):
    def __init__(self, parent: Union[MJCFMorphology, MJCFMorphologyPart], name: str, pos: np.array, euler: np.array,
                 *args, **kwargs):
        super().__init__(parent, name, pos, euler, *args, **kwargs)

    @property
    def morphology_specification(self) -> JumpingSpiderMorphologySpecification:
        return super().morphology_specification

    def _build(self, leg_index: int) -> None:
        self._leg_index = leg_index
        self._leg_specification = self.morphology_specification.leg_specifications[self._leg_index]

        self._build_segments()

    def _build_segments(self) -> None:
        self._segments = []

        for segment_index, segment_specification in enumerate(self._leg_specification.segment_specifications):
            try:
                parent = self._segments[-1]
                position = 2 * self._segments[-1].center_of_capsule
            except IndexError:
                position = np.zeros(3)
                parent = self

            position[0] -= segment_specification.radius.value
            segment = MJCFJumpingSpiderLegSegment(parent=parent,
                                                  name=f"{self.base_name}_{LEG_SEGMENT_NAMES[segment_index]}",
                                                  pos=position,
                                                  euler=[0, segment_specification.oop_attachment_angle.value, 0],
                                                  leg_index=self._leg_index,
                                                  segment_index=segment_index)
            self._segments.append(segment)
