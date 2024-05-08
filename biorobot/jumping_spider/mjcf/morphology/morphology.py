import numpy as np
from dm_control.mjcf.element import _ElementImpl
from moojoco.mjcf.morphology import MJCFMorphology
from transforms3d.euler import euler2quat

from biorobot.jumping_spider.mjcf.morphology.parts.abdomen import MJCFJumpingSpiderAbdomen
from biorobot.jumping_spider.mjcf.morphology.parts.cephalothorax import MJCFJumpingSpiderCephalothorax
from biorobot.jumping_spider.mjcf.morphology.parts.leg import MJCFJumpingSpiderLeg
from biorobot.jumping_spider.mjcf.morphology.specification.default import default_jumping_spider_specification
from biorobot.jumping_spider.mjcf.morphology.specification.specification import JumpingSpiderMorphologySpecification
from biorobot.utils import colors


class MJCFJumpingSpiderMorphology(MJCFMorphology):
    def __init__(
            self,
            specification: JumpingSpiderMorphologySpecification
    ) -> None:
        super().__init__(specification, name="JumpingSpiderMorphology")

    @property
    def morphology_specification(
            self
    ) -> JumpingSpiderMorphologySpecification:
        return super().morphology_specification

    def _build(
            self,
            *args,
            **kwargs
    ) -> None:
        self._configure_compiler()
        self._configure_defaults()

        self._build_cephalothorax()
        self._build_abdomen()
        self._build_legs()

        self._configure_camera()

    def _configure_compiler(
            self
    ) -> None:
        self.mjcf_model.compiler.angle = "radian"

    def _configure_defaults(self) -> None:
        self.mjcf_model.default.geom.condim = 3
        self.mjcf_model.default.geom.contype = 1
        self.mjcf_model.default.geom.conaffinity = 0
        self.mjcf_model.default.geom.density = 1  # CGS

    def _build_cephalothorax(
            self
    ) -> None:
        self._cephalothorax = MJCFJumpingSpiderCephalothorax(
            parent=self, name="cephalothorax", pos=np.zeros(3), euler=np.zeros(3)
        )

    def _build_abdomen(
            self
    ) -> None:
        size_x = self.morphology_specification.cephalothorax_specification.size_x.value

        self._abdomen = MJCFJumpingSpiderAbdomen(
            parent=self._cephalothorax, name="abdomen",
            pos=np.array([-size_x, 0, 0]),
            euler=np.zeros(3)
        )

    def _build_legs(
            self
    ) -> None:
        # Equally spaced over cephalothorax, at correct angles
        self._legs = []

        size_x = self.morphology_specification.cephalothorax_specification.size_x.value
        size_y = self.morphology_specification.cephalothorax_specification.size_y.value
        size_z = self.morphology_specification.cephalothorax_specification.size_z.value

        for leg_index, leg_specification in enumerate(self.morphology_specification.leg_specifications):
            ip_angle = leg_specification.in_plane_connection_angle.value
            oop_angle = leg_specification.out_of_plane_connection_angle.value

            pos_x = size_x * np.cos(oop_angle) * np.cos(ip_angle)
            pos_y = size_y * np.cos(oop_angle) * np.sin(ip_angle)
            pos_z = size_z * np.sin(oop_angle)

            self._legs.append(MJCFJumpingSpiderLeg(
                parent=self._cephalothorax, name=f"right_leg_{leg_index}", pos=np.array([pos_x, pos_y, pos_z]),
                euler=[0, 0, ip_angle], leg_index=leg_index
            ))
            self._legs.append(MJCFJumpingSpiderLeg(
                parent=self._cephalothorax, name=f"left_leg_{leg_index}", pos=np.array([pos_x, -pos_y, pos_z]),
                euler=[0, 0, -ip_angle], leg_index=leg_index
            ))

    def _configure_camera(self) -> None:
        self._cephalothorax.mjcf_body.add(
            "camera",
            name="side_camera",
            pos=[0.0, -5.0, 6.25],
            quat=euler2quat(40 / 180 * np.pi, 0, 0),
            mode="track",
        )

    def add_dragline(self, attachment_site: _ElementImpl) -> None:
        self._dragline = self.mjcf_model.tendon.add(
            'spatial',
            name=f"{self.base_name}_dragline",
            width=0.01,
            rgba=colors.rgba_blue,
            stiffness=self.morphology_specification.dragline_specification.stiffness.value,
            damping=self.morphology_specification.dragline_specification.stiffness.value
        )
        self._dragline.add('site', site=self._abdomen.dragline_body_site)
        self._dragline.add('site', site=attachment_site)


if __name__ == '__main__':
    morphology_specification = default_jumping_spider_specification()
    MJCFJumpingSpiderMorphology(specification=morphology_specification).export_to_xml_with_assets("./morphtest")
