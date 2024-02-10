import numpy as np
from moojoco.mjcf.morphology import MJCFMorphology
from transforms3d.euler import euler2quat

from biorobot.toy_example.mjcf.morphology.parts.arm import MJCFToyExampleArm
from biorobot.toy_example.mjcf.morphology.parts.torso import MJCFToyExampleTorso
from biorobot.toy_example.mjcf.morphology.specification.specification import (
    ToyExampleMorphologySpecification,
)


class MJCFToyExampleMorphology(MJCFMorphology):
    def __init__(self, specification: ToyExampleMorphologySpecification) -> None:
        super().__init__(specification=specification, name="ToyExampleMorphology")

    @property
    def morphology_specification(self) -> ToyExampleMorphologySpecification:
        return super().morphology_specification

    def _build(self, *args, **kwargs) -> None:
        self._configure_compiler()
        self._configure_defaults()

        self._build_torso()
        self._build_arms()

        self._configure_camera()

    def _configure_compiler(self) -> None:
        self.mjcf_model.compiler.angle = "radian"

    def _configure_defaults(self) -> None:
        self.mjcf_model.default.geom.condim = 3
        self.mjcf_model.default.geom.friction = [1.0, 0.5, 0.5]
        self.mjcf_model.default.geom.contype = 1
        self.mjcf_model.default.geom.conaffinity = 0

    def _build_torso(self) -> None:
        self.torso = MJCFToyExampleTorso(
            parent=self, name="torso", pos=np.zeros(3), euler=np.zeros(3)
        )

    def _build_arms(self) -> None:
        # Equally spaced over the torso
        self.arms = []

        torso_radius = self.morphology_specification.torso_specification.radius.value
        number_of_arms = len(self.morphology_specification.arm_specifications)
        arm_angles = np.linspace(0, 2 * np.pi, number_of_arms + 1)[:-1]

        for arm_index in range(number_of_arms):
            angle = arm_angles[arm_index]
            position = torso_radius * np.array([np.cos(angle), np.sin(angle), 0])
            arm = MJCFToyExampleArm(
                parent=self.torso,
                name=f"arm_{arm_index}",
                pos=position,
                euler=[0, 0, angle],
                arm_index=arm_index,
            )
            self.arms.append(arm)

    def _configure_camera(self) -> None:
        self.torso.mjcf_body.add(
            "camera",
            name="side_camera",
            pos=[0.0, -2.0, 2.5],
            quat=euler2quat(40 / 180 * np.pi, 0, 0),
            mode="track",
        )
