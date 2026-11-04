/**
 * Swimmer demo controller: a central pattern generator (CPG) of coupled Van der Pol oscillators.
 *
 * This is a port of `biorobot/swimmer/examples/van_der_pol_cpg.py`. Joint i (head to tail) follows oscillator i + 1
 * of a chain of N = num_joints + 1 oscillators,
 *
 *   x_i'' = mu (1 - x_i^2) x_i' - omega^2 x_i + w_fwd (x_{i-1} - x_i) + w_bwd (x_{i+1} - x_i),   x_0 = x_{N+1} = 0,
 *
 * integrated with RK2 at the model's 2 ms control timestep, and scaled by `settings.amplitude` (rad). After every reset,
 * the chain first runs on its own for 40 cycles (at least 5 s), so the swimmer starts on the CPG's limit cycle.
 * Swapping the couplings alone does not reverse the wave of a chain that is already on its limit cycle (it keeps
 * swimming head first for at least 10 s), so toggling `settings.reverse` restarts the chain and warms it up again with
 * the new coupling, while the body keeps moving.
 *
 * With the default settings, `swimmer/head_first_swimming` (Python, mjc) moves the head 1.49 m head first in 10 s.
 * `settings.reverse` swaps the forward and backward coupling, so the body wave runs from tail to head and the swimmer
 * swims tail first. `swimmer.html` imports `settings` from this module and updates it from its sliders.
 */
const MU = 1.5;
const STRONG_COUPLING = 2.0;
const WEAK_COUPLING = 0.5;
const DT = 0.002; // s, the environment's control timestep (one physics step).
const NAME_PATTERN = /segment_(\d+)_joint/;

export const settings = { omega: 8.0, amplitude: 0.4, reverse: false };

let joints = null; // Oscillator index per actuator (or -1).
let x = null;
let y = null;
let lastTime = null;
let warmedUpReversed = null; // settings.reverse when the chain was last warmed up.

function derivatives(x, y, out) {
  const n = x.length;
  const wFwd = settings.reverse ? WEAK_COUPLING : STRONG_COUPLING;
  const wBwd = settings.reverse ? STRONG_COUPLING : WEAK_COUPLING;
  const omega2 = settings.omega * settings.omega;
  for (let i = 0; i < n; i++) {
    const left = i > 0 ? x[i - 1] : 0;
    const right = i < n - 1 ? x[i + 1] : 0;
    out.dx[i] = y[i];
    out.dy[i] = MU * (1 - x[i] * x[i]) * y[i] - omega2 * x[i] + wFwd * (left - x[i]) + wBwd * (right - x[i]);
  }
}

function rk2Step(dt) {
  const n = x.length;
  const k1 = { dx: new Float64Array(n), dy: new Float64Array(n) };
  const k2 = { dx: new Float64Array(n), dy: new Float64Array(n) };
  derivatives(x, y, k1);
  const xm = new Float64Array(n);
  const ym = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    xm[i] = x[i] + 0.5 * dt * k1.dx[i];
    ym[i] = y[i] + 0.5 * dt * k1.dy[i];
  }
  derivatives(xm, ym, k2);
  for (let i = 0; i < n; i++) {
    x[i] += dt * k2.dx[i];
    y[i] += dt * k2.dy[i];
  }
}

function warmUp(numOscillators) {
  // Staggered initial state: a uniform one synchronises the chain.
  x = Float64Array.from({ length: numOscillators }, (_, i) => 0.1 * Math.cos((2 * Math.PI * (i + 1)) / numOscillators));
  y = new Float64Array(numOscillators);
  const duration = Math.max(5.0, (40 * 2 * Math.PI) / settings.omega);
  for (let step = 0; step < Math.round(duration / DT); step++) rk2Step(DT);
  warmedUpReversed = settings.reverse;
}

export function init(model, data, actuatorNames) {
  joints = actuatorNames.map((name) => {
    const match = NAME_PATTERN.exec(name);
    return match ? Number(match[1]) : -1;
  });
  warmUp(Math.max(1, ...joints) + 1);
  lastTime = null;
}

export function control(t, model, data, actuatorNames) {
  if (!joints) init(model, data, actuatorNames);
  if (settings.reverse !== warmedUpReversed) warmUp(x.length);
  else if (lastTime !== null) {
    // Advance the CPG by the elapsed simulated time (one control step), in steps of DT.
    for (let step = 0; step < Math.round((t - lastTime) / DT); step++) rk2Step(DT);
  }
  lastTime = t;
  const u = new Float64Array(model.nu);
  joints.forEach((oscillator, i) => {
    if (oscillator >= 0) u[i] = settings.amplitude * x[oscillator];
  });
  return u;
}
