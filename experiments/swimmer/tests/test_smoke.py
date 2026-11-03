"""Fast smoke test: build the model, simulate one second, outputs finite."""

import numpy as np
import pytest

from swimmer import make_swimmer_xml, simulate
from swimmer.episode import build, episode_timing, evaluate
from swimmer.runs import load_config
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MORPH = {"N": 8, "taper": 1.0, "stiffness": 0.01}
CTRL = {
    "mu": 1.5,
    "omega": 8.0,
    "w_fwd": 2.0,
    "w_bwd": 0.5,
    "amplitude_scale": 0.4,
    "amp_taper": 0.0,
}


@pytest.mark.parametrize("N", [4, 12])
def test_model_builds(N):
    model, _ = build({"N": N, "taper": 0.5, "stiffness": 0.001}, 0.002)
    assert model.nu == N - 1
    assert model.nbody == N + 1  # world + N segments
    assert "fluidshape" in make_swimmer_xml(N, 0.5, 0.001)


def test_one_second_is_finite():
    model, data = build(MORPH, 0.002)
    r = simulate(model, data, CTRL, duration=1.0, measurement_start_s=0.2, warmup_s=5.0)
    assert r["n_steps"] == 500
    for k in (
        "speed_m_per_s",
        "energy_J",
        "joint_excursion_rad",
        "lateral_amp_m",
        "path_straightness",
        "distance_m",
    ):
        assert np.isfinite(r[k]), k
    assert r["energy_J"] > 0
    assert np.all(np.isfinite(r["joint_angles_window"]))
    assert r["speed_m_per_s"] == pytest.approx(
        -r["distance_m"] / 0.8
    )  # head-first, 0.8 s window


def test_timing_rule():
    ep = load_config(ROOT / "configs" / "sweep.yaml")["episode"]
    slow, fast = episode_timing(3.0, ep), episode_timing(15.0, ep)
    assert slow["warmup_s"] == pytest.approx(40 * 2 * np.pi / 3.0)
    assert slow["measurement_start_s"] == pytest.approx(4 * np.pi / 3.0, abs=1e-3)
    assert fast["measurement_start_s"] == 2.0
    for t in (slow, fast):
        assert t["duration_s"] - t["measurement_start_s"] == pytest.approx(5.0)


def test_evaluate_is_deterministic():
    ep = load_config(ROOT / "configs" / "sweep.yaml")["episode"]
    a = evaluate({**MORPH, **CTRL}, ep)
    b = evaluate({**MORPH, **CTRL}, ep)
    assert not a["failed"] and a["failure"] is None
    assert a == b


def test_instability_is_rejected():
    ep = {
        "timestep_s": 0.05,  # far too coarse: MuJoCo flags bad qacc and resets
        "warmup": {"min_s": 5.0, "cycles": 40.0},
        "settle": {"min_s": 2.0, "cycles": 2.0},
        "window_s": 5.0,
    }
    r = evaluate({**MORPH, **CTRL, "amplitude_scale": 0.8}, ep)
    assert r["failed"] and r["failure"] == "unstable"
