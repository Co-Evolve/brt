import functools

import numpy as np
from moojoco.environment.mjc_env import ThreadedVectorMJCEnvWrapper

from biorobot.toy_example.environment.mjc_env import ToyExampleEnvironmentConfiguration
from biorobot.toy_example.usage_examples.mjc.example_usage_single import (
    create_mjc_environment,
    create_mjc_open_loop_controller,
    post_render,
)


def create_batch_mjc_environment(
    num_envs: int, environment_configuration: ToyExampleEnvironmentConfiguration
) -> ThreadedVectorMJCEnvWrapper:

    batched_env = ThreadedVectorMJCEnvWrapper(
        create_env_fn=functools.partial(
            create_mjc_environment, environment_configuration=environment_configuration
        ),
        num_environments=num_envs,
    )
    return batched_env


if __name__ == "__main__":
    num_envs = 3
    environment_configuration = ToyExampleEnvironmentConfiguration(
        render_mode="rgb_array", camera_ids=[0, 1], joint_randomization_noise_scale=0.01
    )
    batched_env = create_batch_mjc_environment(
        num_envs=num_envs, environment_configuration=environment_configuration
    )
    rng = [np.random.RandomState(i) for i in range(num_envs)]
    batched_state = batched_env.reset(rng=rng)

    controller = create_mjc_open_loop_controller(
        single_action_space=batched_env.single_action_space, num_envs=num_envs
    )

    steps = 0
    fps = 30
    while not np.any(batched_state.terminated | batched_state.truncated):
        ts = batched_state.info["time"]
        batched_action = controller(ts)
        batched_state = batched_env.step(state=batched_state, action=batched_action)

        if steps % int((1 / fps) / environment_configuration.control_timestep) == 0:
            post_render(
                batched_env.render(state=batched_state),
                environment_configuration=environment_configuration,
            )

        steps += 1
    batched_env.close()
