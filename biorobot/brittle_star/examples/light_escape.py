"""Light escape with random torques on a Perlin-noise light map."""

from __future__ import annotations

from typing import Any

import biorobot
from biorobot.brittle_star import default_brittle_star_morphology_specification
from biorobot.brittle_star.examples.common import parse_args, render_mode, run
from biorobot.core import EnvState


def main() -> None:
    args = parse_args(__doc__)
    env = biorobot.make(
        "brittle_star/light_escape",
        backend=args.backend,
        morphology_specification=default_brittle_star_morphology_specification(
            num_arms=5, num_segments_per_arm=5, use_torque_control=True
        ),
        light_perlin_noise_scale=10,
        simulation_time=5,
        render_mode=render_mode(args),
        camera_ids=(0, 1),
        color_contacts=True,
    )

    def policy(state: EnvState, rng: Any):
        action, _ = env.backend.uniform(
            rng, env.action_low, env.action_high, env.action_low.shape
        )
        return action

    run(env, policy, args)


if __name__ == "__main__":
    main()
