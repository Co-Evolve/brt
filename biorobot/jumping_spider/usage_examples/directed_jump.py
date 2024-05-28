from typing import List

import cv2
import numpy as np

from biorobot.jumping_spider.environment.directed_jump.mjc_env import JumpingSpiderDirectedJumpEnvironmentConfiguration, \
    JumpingSpiderDirectedJumpMJCEnvironment
from biorobot.jumping_spider.mjcf.arena.platform_jump import PlatformJumpArenaConfiguration, MJCFPlatformJumpArena
from biorobot.jumping_spider.mjcf.morphology.morphology import MJCFJumpingSpiderMorphology
from biorobot.jumping_spider.mjcf.morphology.specification.default import default_jumping_spider_specification


def post_render(
        render_output: List[np.ndarray],
        environment_configuration: JumpingSpiderDirectedJumpEnvironmentConfiguration,
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


def create_env(
        render_mode: str
) -> JumpingSpiderDirectedJumpMJCEnvironment:
    morphology_specification = default_jumping_spider_specification()
    morphology = MJCFJumpingSpiderMorphology(specification=morphology_specification)

    arena_configuration = PlatformJumpArenaConfiguration()
    arena = MJCFPlatformJumpArena(configuration=arena_configuration)

    env_config = JumpingSpiderDirectedJumpEnvironmentConfiguration(
        render_mode=render_mode,
        num_physics_steps_per_control_step=10,
        simulation_time=5,
        time_scale=1,
        camera_ids=[0, 1],
        color_contacts=True,
    )
    env = JumpingSpiderDirectedJumpMJCEnvironment.from_morphology_and_arena(
        morphology=morphology, arena=arena, configuration=env_config
    )
    return env


if __name__ == "__main__":
    RENDER_MODE = "human"

    env = create_env(render_mode=RENDER_MODE)

    env_rng = np.random.RandomState(0)
    step_fn = env.step
    reset_fn = env.reset

    state = reset_fn(env_rng)
    while True:
        action = env.action_space.sample()
        state = step_fn(state=state, action=action)
        post_render(env.render(state=state), env.environment_configuration)
        if state.terminated | state.truncated:
            state = reset_fn(env_rng)
    env.close()
