from typing import Callable, List

import cv2
import gymnasium
import numpy as np

from biorobot.toy_example.environment.mjc_env import (
    ToyExampleEnvironmentConfiguration,
    ToyExampleMJCEnvironment,
)
from biorobot.toy_example.mjcf.arena.arena import (
    MJCFPlaneWithTargetArena,
    PlaneWithTargetArenaConfiguration,
)
from biorobot.toy_example.mjcf.morphology.morphology import MJCFToyExampleMorphology
from biorobot.toy_example.mjcf.morphology.specification.default import (
    default_toy_example_morphology_specification,
)


def create_mjc_environment(
    environment_configuration: ToyExampleEnvironmentConfiguration,
) -> ToyExampleMJCEnvironment:
    morphology_specification = default_toy_example_morphology_specification(
        num_arms=4, num_segments_per_arm=2
    )
    morphology = MJCFToyExampleMorphology(specification=morphology_specification)
    arena_configuration = PlaneWithTargetArenaConfiguration()
    arena = MJCFPlaneWithTargetArena(configuration=arena_configuration)

    env = ToyExampleMJCEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=environment_configuration
    )
    return env


def create_mjc_open_loop_controller(
    single_action_space: gymnasium.spaces.Box, num_envs: int
) -> Callable[[float], np.ndarray]:
    def open_loop_controller(t: float) -> np.ndarray:
        actions = np.ones(single_action_space.shape)
        actions[::2] = np.cos(5 * t)
        actions[1::2] = np.sin(5 * t)
        actions[-actions.shape[0] // 2 :: 2] *= -1
        return actions

    if num_envs > 1:
        batched_open_loop_controller = lambda t: np.stack(
            [open_loop_controller(tt) for tt in t]
        )
        return batched_open_loop_controller
    return open_loop_controller


def post_render(
    render_output: List[np.ndarray],
    environment_configuration: ToyExampleEnvironmentConfiguration,
) -> None:
    if environment_configuration.render_mode == "human":
        return

    num_cameras = len(environment_configuration.camera_ids)
    num_envs = len(render_output) // num_cameras

    if num_cameras > 1:
        # Horizontally stack frames of the same environment
        frames_per_env = np.array_split(render_output, num_envs)
        render_output = [
            np.concatenate(env_frames, axis=1) for env_frames in frames_per_env
        ]

    # Vertically stack frames of different environments
    stacked_frames = np.concatenate(render_output, axis=0)
    cv2.imshow("render", stacked_frames)
    cv2.waitKey(1)


if __name__ == "__main__":
    environment_configuration = ToyExampleEnvironmentConfiguration(
        render_mode="human", camera_ids=[0, 1]
    )
    env = create_mjc_environment(environment_configuration=environment_configuration)
    state = env.reset(rng=np.random.RandomState(seed=0))

    controller = create_mjc_open_loop_controller(
        single_action_space=env.action_space, num_envs=1
    )

    steps = 0
    fps = 30
    while not (state.truncated | state.terminated):
        action = controller(state.info["time"])
        state = env.step(state=state, action=action)

        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(
                env.render(state=state),
                environment_configuration=environment_configuration,
            )
            print(state.observations["segment_ground_contact"])

        steps += 1
    env.close()
