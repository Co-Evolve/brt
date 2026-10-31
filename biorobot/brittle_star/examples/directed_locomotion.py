"""Directed locomotion with random torques towards a target at (-1, 0); press Ctrl+C to stop."""

from __future__ import annotations

from typing import Any

import numpy as np

import biorobot
from biorobot.brittle_star import default_brittle_star_morphology_specification
from biorobot.brittle_star.examples.common import parse_args, render_mode, run
from biorobot.core import EnvState


def main() -> None:
    args = parse_args(__doc__)
    env = biorobot.make(
        "brittle_star/directed_locomotion",
        backend=args.backend,
        morphology_specification=default_brittle_star_morphology_specification(
            num_arms=5,
            num_segments_per_arm=5,
            use_torque_control=True,
            num_contact_sensors_per_segment=2,
        ),
        simulation_time=5,
        render_mode=render_mode(args),
        camera_ids=(0, 1),
        color_contacts=True,
        random_initial_rotation=True,
    )

    def policy(state: EnvState, rng: Any):
        action, _ = env.backend.uniform(
            rng, env.action_low, env.action_high, env.action_low.shape
        )
        return action

    # Leave out target_position (or pass NaNs) for a random target at config.target_distance from the origin.
    run(
        env,
        policy,
        args,
        reset_options={"target_position": np.array([-1.0, 0.0, 0.05])},
    )


if __name__ == "__main__":
    main()
