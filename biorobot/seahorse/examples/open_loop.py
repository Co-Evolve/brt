"""Open-loop playground: curl the tip of the default tail with its short MVM beams, then the whole tail with its long
HM beams.

The default tail has 11 segments, a pair of force-controlled ventral HM beams that span the whole tail and 20
position-controlled MVM beams (two between every pair of neighbouring segments). From 0 to 2 s, the MVM beams of the
four most distal segment pairs contract gradually, which curls the tip (without gravity, the reaction of the curling tip
also makes the nearly stiffness-free proximal joints drift to their limits; with ``enable_gravity=True`` they stay
straight); from 2.5 to 4.5 s, the HM beams contract, which bends the whole tail ventrally (positive pitch).

    python -m biorobot.seahorse.examples.open_loop --backend mjc --render human
"""

from __future__ import annotations

import numpy as np

from biorobot import make
from biorobot.core.env import BRTEnv
from biorobot.seahorse.examples.common import contraction, parse_args, ramp, rollout

#: Number of distal segment pairs whose MVM beams contract.
NUM_DISTAL_MVM_PAIRS = 4


def open_loop_action(env: BRTEnv, t: float) -> np.ndarray:
    """Actions at time ``t`` (s): the distal MVM beams contract over 0-2 s, the HM beams over 2.5-4.5 s."""
    is_hm = np.array(["/hm_beam_" in name for name in env.actuator_names])
    mvm = np.flatnonzero(~is_hm)
    level = np.zeros(len(is_hm))
    level[mvm[len(mvm) - 2 * NUM_DISTAL_MVM_PAIRS :]] = ramp(t, 0.0, 2.0)
    level[is_hm] = ramp(t, 2.5, 2.0)
    return contraction(env, level)


def main() -> None:
    args = parse_args(__doc__)
    render_mode = None if args.render == "none" else args.render
    env = make(
        "seahorse/tail",
        backend=args.backend,
        render_mode=render_mode,
        camera_ids=(0, 2),
    )
    observations = rollout(env, open_loop_action)
    steps_per_second = round(1 / env.config.control_timestep)
    for t in (2.5, env.config.simulation_time):
        pitch = observations[round(t * steps_per_second) - 1][
            "vertebrae_pitch_joint_pos"
        ]
        print(
            f"Vertebral pitch (rad, base to tip) at {t:.1f} s: {np.array2string(pitch, precision=2)}, "
            f"total curl {np.rad2deg(pitch.sum()):.0f} deg."
        )
    env.close()


if __name__ == "__main__":
    main()
