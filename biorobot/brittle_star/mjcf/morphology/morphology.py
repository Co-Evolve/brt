import numpy as np
from moojoco.mjcf.morphology import MJCFMorphology
from transforms3d.euler import euler2quat

from biorobot.brittle_star.mjcf.morphology.parts.arm import MJCFBrittleStarArm
from biorobot.brittle_star.mjcf.morphology.parts.disk import MJCFBrittleStarDisk
from biorobot.brittle_star.mjcf.morphology.specification.default import (
    default_brittle_star_morphology_specification,
)
from biorobot.brittle_star.mjcf.morphology.specification.specification import (
    BrittleStarMorphologySpecification,
)


class MJCFBrittleStarMorphology(MJCFMorphology):
    def __init__(self, specification: BrittleStarMorphologySpecification) -> None:
        super().__init__(specification, name="BrittleStarMorphology")

    @property
    def morphology_specification(self) -> BrittleStarMorphologySpecification:
        return super().morphology_specification

    def _build(self, *args, **kwargs) -> None:
        self._configure_compiler()
        self._configure_defaults()
        self._build_disk()
        self._build_arms()
        self._configure_camera()

    def _configure_compiler(self) -> None:
        self.mjcf_model.compiler.angle = "radian"

    def _configure_defaults(self) -> None:
        self.mjcf_model.default.geom.condim = 6
        self.mjcf_model.default.geom.contype = 1
        self.mjcf_model.default.geom.conaffinity = 0

    def _build_disk(self) -> None:
        self._disk = MJCFBrittleStarDisk(
            parent=self, name="central_disk", pos=np.zeros(3), euler=np.zeros(3)
        )

    def _build_arms(self) -> None:
        # Equally spaced over the disk
        self.arms = []

        disk_radius = self.morphology_specification.disk_specification.radius.value
        arm_angles = np.linspace(0, 2 * np.pi, 6)[:-1]
        number_of_arms = self.morphology_specification.number_of_arms

        for arm_index in range(number_of_arms):
            angle = arm_angles[arm_index]
            position = disk_radius * np.array([np.cos(angle), np.sin(angle), 0])
            arm = MJCFBrittleStarArm(
                parent=self._disk,
                name=f"arm_{arm_index}",
                pos=position,
                euler=[0, 0, angle],
                arm_index=arm_index,
            )
            self.arms.append(arm)

    def _configure_camera(self) -> None:
        self._disk.mjcf_body.add(
            "camera",
            name="side_camera",
            pos=[0.0, -2.0, 2.5],
            quat=euler2quat(40 / 180 * np.pi, 0, 0),
            mode="track",
        )


if __name__ == "__main__":
    spec = default_brittle_star_morphology_specification(
        num_arms=5, num_segments_per_arm=5, use_p_control=True
    )
    MJCFBrittleStarMorphology(spec).export_to_xml_with_assets("./mjcf")
