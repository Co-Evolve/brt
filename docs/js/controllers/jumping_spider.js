/**
 * Jumping spider demo controller: the open-loop jump of `biorobot/jumping_spider/examples/scripted_jump.py`.
 *
 * The spider crouches with its abdomen and legs I-II raised, then pushes off with legs III and IV and returns to the
 * resting pose for the landing. Targets are joint angles (rad) for the position actuators; joints that are not listed
 * stay at 0. In `jumping_spider/long_jump` (Python, mjc and mjx) it takes off after ~0.56 s, flies for ~0.39 s with an
 * apex of ~0.23 m and lands upright ~0.47 m forward. The viewer holds each control for the environment's control
 * timestep, as `env.step` does, so the browser reproduces that rollout; the page resets the simulation after the
 * landing so the jump repeats.
 *
 * Keep the poses in sync with `CROUCH_POSE` / `LAUNCH_POSE` in scripted_jump.py. Keys are "<leg pairs>:<joint>", where
 * an empty leg-pair list selects the abdomen.
 */
const CROUCH_DURATION = 0.53; // s; the crouch pose is reached halfway and then held to settle.
const LAUNCH_DURATION = 0.12; // s; afterwards all joints return to the resting pose.
const CROUCH_POSE = {
  ":abdomen_out_of_plane": 0.28,
  "12:metatarsus_out_of_plane": 0.47,
  "34:femur_out_of_plane": -0.36,
  "3:patella_out_of_plane": 0.23,
  "4:patella_out_of_plane": 1.18,
  "4:metatarsus_out_of_plane": -1.68,
};
const LAUNCH_POSE = {
  ":abdomen_out_of_plane": 0.09,
  "1234:coxa_in_plane": 0.19,
  "12:femur_out_of_plane": -0.29,
  "12:patella_out_of_plane": 0.16,
  "12:tibia_in_plane": -0.2,
  "12:metatarsus_out_of_plane": 0.15,
  "12:tarsus_out_of_plane": 0.69,
  "3:coxa_out_of_plane": -0.22,
  "3:femur_out_of_plane": 0.84,
  "3:patella_out_of_plane": -0.69,
  "3:tarsus_out_of_plane": 0.88,
  "4:coxa_out_of_plane": 0.68,
  "4:femur_out_of_plane": 0.97,
  "4:patella_out_of_plane": -0.26,
  "4:tibia_in_plane": 0.31,
  "4:metatarsus_out_of_plane": -1.34,
  "4:tarsus_out_of_plane": 1.31,
};

let crouch = null;
let launch = null;

/** One target per actuator. Leg actuators are named "<prefix>/leg_<side><pair>_<segment>_<in|out_of>_plane". */
function poseAction(pose, actuatorNames) {
  return Float64Array.from(actuatorNames, (fullName) => {
    const name = fullName.split("/").pop();
    const pair = name.startsWith("leg_") ? name[5] : null;
    let target = 0;
    for (const [key, angle] of Object.entries(pose)) {
      const [pairs, joint] = key.split(":");
      const selected = pairs === "" ? pair === null : pair !== null && pairs.includes(pair);
      if (selected && name.endsWith(joint)) target = angle;
    }
    return target;
  });
}

export function init(model, data, actuatorNames) {
  crouch = poseAction(CROUCH_POSE, actuatorNames);
  launch = poseAction(LAUNCH_POSE, actuatorNames);
}

export function control(t, model, data, actuatorNames) {
  if (!crouch) init(model, data, actuatorNames);
  if (t < CROUCH_DURATION) {
    const progress = Math.min(t / (0.5 * CROUCH_DURATION), 1);
    return crouch.map((angle) => progress * angle);
  }
  if (t < CROUCH_DURATION + LAUNCH_DURATION) return launch;
  return new Float64Array(model.nu);
}
