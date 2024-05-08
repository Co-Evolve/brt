import abc
from typing import Optional

import numpy as np
from dm_control.mjcf.element import _AttachmentFrame
from moojoco.mjcf.arena import ArenaConfiguration, MJCFArena

from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology
from biorobot.utils import colors


class SpiderArenaConfiguration(ArenaConfiguration):
    def __init__(
            self,
            name: str = "SpiderArena",
    ) -> None:
        super().__init__(name=name)


class MJCFSpiderArena(MJCFArena, abc.ABC):
    @property
    def arena_configuration(self) -> SpiderArenaConfiguration:
        return super().arena_configuration

    def _build(self, *args, **kwargs) -> None:
        self._configure_lights()
        self._configure_sky()
        self._configure_cameras()
        self._configure_dragline_attachment_site()

    def _configure_dragline_attachment_site(self) -> None:
        self.dragline_attachment_site = self.mjcf_body.add("site",
                                                           type="sphere",
                                                           name=f"{self.base_name}_dragline_attachment_site",
                                                           rgba=colors.rgba_red,
                                                           pos=np.zeros(3),
                                                           size=[0.01])

    def _configure_cameras(self) -> None:
        raise NotImplementedError

    def _configure_lights(self) -> None:
        self.mjcf_model.worldbody.add(
            "light",
            pos=[-20, 0, 20],
            directional=True,
            dir=[0, 0, -0.5],
            diffuse=[0.1, 0.1, 0.1],
            castshadow=False,
        )
        self.mjcf_model.visual.headlight.set_attributes(
            ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1]
        )

    def _configure_sky(self) -> None:
        # white sky
        self.mjcf_model.asset.add(
            "texture",
            type="skybox",
            builtin="flat",
            rgb1="1.0 1.0 1.0",
            rgb2="1.0 1.0 1.0",
            width=200,
            height=200,
        )

    def attach(
            self,
            other: MJCFJumpingSpiderMorphology,
            position: Optional[np.ndarray] = None,
            euler: Optional[np.ndarray] = None,
            free_joint: bool = False,
    ) -> _AttachmentFrame:
        frame = super().attach(other=other, position=position, euler=euler, free_joint=free_joint)
        other.add_dragline(attachment_site=self.dragline_attachment_site)

        # Reposition dragline attachment site to be under abdomen end
        abdomen_site_pos = other._abdomen.coordinates_of_point_in_root_frame(other._abdomen.dragline_body_site.pos)
        self.dragline_attachment_site.pos[0] = abdomen_site_pos[0] + np.sign(abdomen_site_pos[0]) * 0.5 / np.tan(45 / 180 * np.pi)

        return frame
