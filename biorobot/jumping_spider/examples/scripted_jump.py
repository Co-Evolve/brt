"""An open-loop jump: crouch with the front legs raised, then rapidly extend the hind legs.

This shows that the default morphology can jump: in the long jump environment (identically on mjc and mjx) the spider
takes off after ~0.56 s, is fully airborne for ~0.39 s with an apex of ~0.23 m, and lands upright ~0.466 m further,
i.e. ~3.5 body lengths (0.132 m). In the platform jump it lands with all legs on the end platform for gaps of
0.15-0.25 m. The controller requires position control (the default), since its actions are joint angle targets; a
dragline brake, if present, stays released. The poses were found with a small random search and then rounded.

    python -m biorobot.jumping_spider.examples.scripted_jump --backend mjc --render human
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

import biorobot
from biorobot.jumping_spider.examples.common import parse_args

#: Duration (s) of the crouch phase; the crouch pose is reached halfway and then held to settle.
CROUCH_DURATION = 0.53
#: Duration (s) of the push-off; afterwards all joints return to the resting pose to prepare for landing.
LAUNCH_DURATION = 0.12
#: Joint targets (rad) per (leg pairs, joint), where leg pairs ``None`` selects the abdomen. Joints that are not
#: listed stay at their resting angle of 0.
CROUCH_POSE: Dict[Tuple[Optional[str], str], float] = {
    # Raise the abdomen and lift legs I and II off the ground, as salticids do before a jump.
    (None, "abdomen_out_of_plane"): 0.28,
    ("12", "metatarsus_out_of_plane"): 0.47,
    # Lower the body onto legs III and IV, folding legs IV at the femur-patella and tibia-metatarsus joints.
    ("34", "femur_out_of_plane"): -0.36,
    ("3", "patella_out_of_plane"): 0.23,
    ("4", "patella_out_of_plane"): 1.18,
    ("4", "metatarsus_out_of_plane"): -1.68,
}
LAUNCH_POSE: Dict[Tuple[Optional[str], str], float] = {
    (None, "abdomen_out_of_plane"): 0.09,
    # All legs swing backwards; legs I and II stay lifted and reach forward for the landing.
    ("1234", "coxa_in_plane"): 0.19,
    ("12", "femur_out_of_plane"): -0.29,
    ("12", "patella_out_of_plane"): 0.16,
    ("12", "tibia_in_plane"): -0.2,
    ("12", "metatarsus_out_of_plane"): 0.15,
    ("12", "tarsus_out_of_plane"): 0.69,
    # Legs III and IV push off: depress the femora and extend the femur-patella (legs III) and tibia-metatarsus
    # (legs IV) joints, the two joints that spiders extend hydraulically.
    ("3", "coxa_out_of_plane"): -0.22,
    ("3", "femur_out_of_plane"): 0.84,
    ("3", "patella_out_of_plane"): -0.69,
    ("3", "tarsus_out_of_plane"): 0.88,
    ("4", "coxa_out_of_plane"): 0.68,
    ("4", "femur_out_of_plane"): 0.97,
    ("4", "patella_out_of_plane"): -0.26,
    ("4", "tibia_in_plane"): 0.31,
    ("4", "metatarsus_out_of_plane"): -1.34,
    ("4", "tarsus_out_of_plane"): 1.31,
}


def _pose_action(
    pose: Dict[Tuple[Optional[str], str], float], actuator_names: Sequence[str]
) -> np.ndarray:
    action = np.zeros(len(actuator_names))
    for i, name in enumerate(actuator_names):
        # Leg actuators are named "<prefix>/leg_<side><pair>_<segment>_<in|out_of>_plane".
        name = name.rsplit("/", 1)[-1]
        pair = name[5] if name.startswith("leg_") else None
        for (pairs, joint), angle in pose.items():
            if pairs is None:
                selected = pair is None
            else:
                selected = pair is not None and pair in pairs
            if selected and name.endswith(joint):
                action[i] = angle
    return action


def scripted_jump_action(t: float, actuator_names: Sequence[str]) -> np.ndarray:
    """Joint angle targets (rad) of the open-loop jump at time ``t`` (s) since the start of the episode.

    Args:
        t: Time since the start of the episode (s).
        actuator_names: The environment's actuator names (``env.actuator_names``), to order the targets.

    Returns:
        One target per actuator; all targets lie within the default joint ranges.
    """
    if t < CROUCH_DURATION:
        progress = min(t / (0.5 * CROUCH_DURATION), 1.0)
        return progress * _pose_action(CROUCH_POSE, actuator_names)
    if t < CROUCH_DURATION + LAUNCH_DURATION:
        return _pose_action(LAUNCH_POSE, actuator_names)
    return np.zeros(len(actuator_names))


def main() -> None:
    args = parse_args(__doc__)
    render_mode = None if args.render == "none" else args.render
    env = biorobot.make(
        "jumping_spider/long_jump",
        backend=args.backend,
        render_mode=render_mode,
        camera_ids=(1,),
    )
    if args.backend == "mjc":
        state, step = env.reset(0), env.step
    else:
        import jax

        state, step = jax.jit(env.reset)(jax.random.PRNGKey(0)), jax.jit(env.step)

    apex, longest_flight = 0.0, 0.0
    while not (bool(state.terminated) or bool(state.truncated)):
        action = scripted_jump_action(float(state.info["time"]), env.actuator_names)
        state = step(state, action)
        apex = max(apex, float(state.observations["cephalothorax_position"][2]))
        longest_flight = max(longest_flight, float(state.info["flight_time"]))
        if render_mode is not None:
            env.render(state)
    print(
        f"Landed at x = {float(state.observations['x_distance'][0]):.3f} m after a {longest_flight:.2f} s flight "
        f"(apex {apex:.3f} m)."
    )
    env.close()


if __name__ == "__main__":
    main()
