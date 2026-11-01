"""The Gymnasium adapter passes gymnasium's environment checker for every registered environment."""

import numpy as np
import pytest

import biorobot
from biorobot.core.gym import GymnasiumEnv

gymnasium = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env  # noqa: E402


@pytest.mark.parametrize("name", biorobot.available())
def test_check_env(name):
    env = GymnasiumEnv(biorobot.make(name, simulation_time=0.2))
    check_env(env, skip_render_check=True)
    env.close()


def test_reset_options_and_info_copies():
    env = GymnasiumEnv(
        biorobot.make("brittle_star/directed_locomotion", simulation_time=0.2)
    )
    _, info = env.reset(seed=0, options={"target_position": [1.0, -2.0, 0.05]})
    np.testing.assert_allclose(info["xy_target_position"], [1.0, -2.0])
    info["xy_target_position"][:] = 0.0
    np.testing.assert_allclose(env.state.data.mocap_pos[0, :2], [1.0, -2.0])
    assert env.metadata["render_fps"] == pytest.approx(
        1 / env.env.config.control_timestep
    )
    env.close()
