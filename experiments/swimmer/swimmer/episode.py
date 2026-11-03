"""
episode.py — one rollout of the VdP-CPG-driven swimmer and its descriptors.

``simulate`` is copied from MuJoCo-demo ``src/swimmer.py`` (commit f7ad026)
with the rendering branch removed. ``evaluate`` is copied from
``src/mapelites.py`` / ``src/run_sweep.py`` and compiles the model from the
XML string instead of a cached file. Three deliberate changes relative to the
runs of May 2026 (decided 2026-11-03, see VERIFICATION.md):

1. Warm-up and settling follow the post-erratum rule of MuJoCo-demo
   ``src/v4/v4_model.py``: CPG warm-up = max(5 s, 40 cycles) and body
   settling = max(2 s, 2 cycles), with the cycle period 2*pi/omega. The
   measurement window stays 5 s, so the episode lasts settling + 5 s.
2. Speed (the quality) is head-first: positive when the chain moves towards
   its head segment seg0, i.e. along -x. ``distance`` keeps its +x sign.
3. An evaluation in which MuJoCo reports an instability (bad qpos, qvel or
   qacc, after which MuJoCo silently resets the state) is invalid.

Joint for segment i in {2..N} takes target amp_i * x_i, where
    amp_i = amplitude_scale * (1 + amp_taper * (i - (N+1)/2) / (N/2))
gives a linear amplitude envelope along the body (amp_taper = 0 -> uniform,
> 0 -> tail-biased, < 0 -> head-biased).
"""

import numpy as np
import mujoco

from .cpg import rk2_step, warmup_vdp
from .model import make_swimmer_xml

PARAM_NAMES = (
    "N",
    "taper",
    "stiffness",
    "mu",
    "omega",
    "w_fwd",
    "w_bwd",
    "amplitude_scale",
    "amp_taper",
)
DESCRIPTOR_NAMES = (
    "speed",
    "energy",
    "joint_excursion",
    "lateral_amp",
    "path_straightness",
    "distance",
)


def simulate(
    model,
    data,
    controller_params: dict,
    duration: float = 6.0,
    measurement_start_s: float = 1.0,
    warmup_s: float = 5.0,
) -> dict:
    """Roll the VdP-CPG-driven swimmer through `duration` seconds.

    Two warm-ups happen in series:
      1. The VdP chain runs in isolation for `warmup_s` seconds so the body
         sees an already-limit-cycle CPG.
      2. The MuJoCo simulation runs for `duration` seconds; the first
         `measurement_start_s` seconds are treated as a body-settling
         transient and excluded from descriptor calculations.

    All six descriptors are computed over t in [measurement_start_s, duration]:
        speed (signed, head-first: minus the mean vx of the head frame),
        energy (integral of |tau * omega| dt),
        joint_excursion (mean peak-to-peak hinge angle, across joints),
        lateral_amp (head y peak-to-peak / 2),
        path_straightness (|xy displacement| / xy path length),
        distance (signed x displacement).
    Raises if the trajectory contains non-finite values (instability).
    """
    mu = controller_params["mu"]
    omega = controller_params["omega"]
    w_fwd = controller_params["w_fwd"]
    w_bwd = controller_params["w_bwd"]
    amplitude_scale = controller_params["amplitude_scale"]
    amp_taper = controller_params.get("amp_taper", 0.0)
    N = model.nu + 1  # hinges = N-1 = nu

    # Linear amplitude envelope along the body. Joints exist for segments
    # 2..N (1-indexed); segment 1 is the head with the freejoint.
    seg_idx = np.arange(2, N + 1, dtype=np.float64)
    amp_envelope = amplitude_scale * (
        1.0 + amp_taper * (seg_idx - (N + 1) / 2.0) / (N / 2.0)
    )

    hinge_qpos_slice = slice(7, 7 + (N - 1))
    hinge_qvel_slice = slice(6, 6 + (N - 1))
    dt = model.opt.timestep

    x_vdp, y_vdp = warmup_vdp(N, mu, omega, w_fwd, w_bwd, dt, warmup_s)

    n_steps = int(round(duration / dt))
    ms_step = int(round(measurement_start_s / dt))
    if ms_step >= n_steps:
        raise ValueError(
            f"measurement_start_s ({measurement_start_s}) "
            f"must be < duration ({duration})"
        )

    joint_angles = np.empty((n_steps, N - 1), dtype=np.float64)
    head_xy = np.empty((n_steps, 2), dtype=np.float64)
    step_power = np.empty(n_steps, dtype=np.float64)

    for step in range(n_steps):
        data.ctrl[:] = amp_envelope * x_vdp[1:N]
        mujoco.mj_step(model, data)
        x_vdp, y_vdp = rk2_step(x_vdp, y_vdp, dt, mu, omega, w_fwd, w_bwd)
        omegas = data.qvel[hinge_qvel_slice]
        torques = data.qfrc_actuator[hinge_qvel_slice]
        step_power[step] = float(np.sum(np.abs(torques * omegas)))
        joint_angles[step] = data.qpos[hinge_qpos_slice]
        head_xy[step] = (data.qpos[0], data.qpos[1])

    if not (np.all(np.isfinite(head_xy)) and np.all(np.isfinite(joint_angles))):
        raise ValueError("non-finite state (divergence)")
    n_bad = instability_count(data)
    # a silent mj_resetData also restarts the clock
    if n_bad or abs(data.time - n_steps * dt) > 0.5 * dt:
        raise UnstableError(f"MuJoCo instability ({n_bad} warnings, t={data.time})")

    win = slice(ms_step, n_steps)
    hxy = head_xy[win]
    ja = joint_angles[win]
    window_duration = (n_steps - ms_step) * dt

    x_start = float(hxy[0, 0])
    y_start = float(hxy[0, 1])
    x_end = float(hxy[-1, 0])
    y_end = float(hxy[-1, 1])

    distance = x_end - x_start  # signed +x displacement
    speed = -distance / window_duration  # head-first speed: + towards seg0 (-x)
    displacement = float(np.hypot(x_end - x_start, y_end - y_start))
    diffs = np.diff(hxy, axis=0)
    path_length = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    path_straightness = (displacement / path_length) if path_length > 0 else 0.0
    lateral_amp = float((hxy[:, 1].max() - hxy[:, 1].min()) / 2.0)
    joint_excursion = float(np.mean(ja.max(axis=0) - ja.min(axis=0)))
    energy = float(np.sum(step_power[win])) * dt

    return {
        "n_steps": n_steps,
        "measurement_start_step": ms_step,
        "timestep_s": float(dt),
        "speed_m_per_s": speed,
        "energy_J": energy,
        "joint_excursion_rad": joint_excursion,
        "lateral_amp_m": lateral_amp,
        "path_straightness": path_straightness,
        "distance_m": distance,
        "final_position_m": [x_end, y_end, float(data.qpos[2])],
        "joint_angles_window": ja,  # (T_window, N-1), post-settling
    }


class UnstableError(RuntimeError):
    """MuJoCo flagged a bad state during the episode (and reset it)."""


UNSTABLE_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
)


def instability_count(data) -> int:
    return int(sum(data.warning[int(w)].number for w in UNSTABLE_WARNINGS))


def episode_timing(omega: float, episode: dict) -> dict:
    """Warm-up, measurement start and duration (s) for one controller.

    Post-erratum rule (MuJoCo-demo src/v4/v4_model.py): warm-up =
    max(min_s, cycles * 2 pi / omega), settling likewise; the measurement
    window is a fixed window_s, so duration = settling + window_s. Settling is
    rounded to whole timesteps first so every window has the same step count.
    """
    dt = episode["timestep_s"]
    period = 2.0 * np.pi / omega
    w, st = episode["warmup"], episode["settle"]
    warmup_s = max(w["min_s"], w["cycles"] * period)
    settle_s = round(max(st["min_s"], st["cycles"] * period) / dt) * dt
    return {
        "warmup_s": warmup_s,
        "measurement_start_s": settle_s,
        "duration_s": settle_s + episode["window_s"],
    }


def build(params: dict, timestep_s: float):
    """Compile the MuJoCo model for one morphology; return (model, data)."""
    xml = make_swimmer_xml(
        N=int(params["N"]),
        taper=float(params["taper"]),
        stiffness=float(params["stiffness"]),
    )
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = timestep_s
    return model, mujoco.MjData(model)


def evaluate(params: dict, episode: dict) -> dict:
    """Run one simulation. Returns descriptors (or NaNs on failure).

    `episode` holds timestep_s, window_s and the warmup/settle rules (see
    episode_timing). Any exception marks the evaluation as failed, as in the
    original scripts; ``failure`` says why ("unstable" or "error").
    """
    ctrl = {
        "mu": float(params["mu"]),
        "omega": float(params["omega"]),
        "w_fwd": float(params["w_fwd"]),
        "w_bwd": float(params["w_bwd"]),
        "amplitude_scale": float(params["amplitude_scale"]),
        "amp_taper": float(params["amp_taper"]),
    }
    try:
        timing = episode_timing(ctrl["omega"], episode)
        model, data = build(params, episode["timestep_s"])
        r = simulate(
            model,
            data,
            ctrl,
            duration=timing["duration_s"],
            measurement_start_s=timing["measurement_start_s"],
            warmup_s=timing["warmup_s"],
        )
        return {
            "speed": float(r["speed_m_per_s"]),
            "energy": float(r["energy_J"]),
            "joint_excursion": float(r["joint_excursion_rad"]),
            "lateral_amp": float(r["lateral_amp_m"]),
            "path_straightness": float(r["path_straightness"]),
            "distance": float(r["distance_m"]),
            "failed": False,
            "failure": None,
        }
    except UnstableError:
        return {
            **{d: np.nan for d in DESCRIPTOR_NAMES},
            "failed": True,
            "failure": "unstable",
        }
    except Exception:
        return {
            **{d: np.nan for d in DESCRIPTOR_NAMES},
            "failed": True,
            "failure": "error",
        }
