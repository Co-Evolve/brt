/**
 * Brittle star demo controller: an open-loop rowing gait for the default five-armed brittle star.
 *
 * This is the five-armed version of `biorobot/brittle_star/examples/rowing.py` (position control, targets scaled to
 * each actuator's ctrlrange): every segment traces an oval, its out-of-plane joint leading the in-plane joint by a
 * quarter period, so arms lift while sweeping forward and press down while sweeping back. In-plane strokes are
 * mirrored across the axis of arm 0 (amplitude scales with the sine of the arm's angle), so both sides push the disk
 * along arm 0, and a phase lag per segment turns each arm into a travelling wave.
 *
 * In `brittle_star/undirected_locomotion` (Python, mjc) this moves the disk 1.99 m along +x in 10 s while it stays
 * flat (tilt below 0.01 rad). The viewer holds each control for the environment's control timestep, as `env.step`
 * does, so the browser reproduces that rollout. Actuators are matched by name, so any prefix, number of arms or number
 * of segments works.
 */
const FREQUENCY_HZ = 1.0;
const AMPLITUDE = 0.8; // Fraction of the actuator's ctrlrange half-width.
const SEGMENT_PHASE_LAG = 0.4; // rad per segment, base to tip.
const NAME_PATTERN = /arm_(\d+)_segment_(\d+)_(in_plane|out_of_plane)/;

let plan = null;

export function init(model, data, actuatorNames) {
  const range = model.actuator_ctrlrange;
  const parsed = actuatorNames.map((name) => NAME_PATTERN.exec(name));
  const numArms = Math.max(1, ...parsed.filter(Boolean).map((match) => Number(match[1]) + 1));
  plan = parsed.map((match, i) => {
    if (!match) return null;
    const arm = Number(match[1]);
    const inPlane = match[3] === "in_plane";
    const low = range[2 * i];
    const high = range[2 * i + 1];
    const side = inPlane ? Math.sin((2 * Math.PI * arm) / numArms) : 1;
    return {
      segment: Number(match[2]),
      inPlane,
      centre: (low + high) / 2,
      amplitude: (side * AMPLITUDE * (high - low)) / 2,
    };
  });
}

export function control(t, model, data, actuatorNames) {
  if (!plan) init(model, data, actuatorNames);
  const u = new Float64Array(model.nu);
  const omega = 2 * Math.PI * FREQUENCY_HZ;
  plan.forEach((entry, i) => {
    if (!entry) return;
    const phase = omega * t - SEGMENT_PHASE_LAG * entry.segment;
    const wave = entry.inPlane ? Math.sin(phase) : -Math.cos(phase);
    u[i] = entry.centre + entry.amplitude * wave;
  });
  return u;
}
