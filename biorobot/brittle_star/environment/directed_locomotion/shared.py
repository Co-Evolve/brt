import abc
from typing import Sequence

import chex
from moojoco.environment.base import BaseEnvState

from biorobot.brittle_star.environment.shared.base import (
    BrittleStarEnvironmentBase,
    BrittleStarEnvironmentBaseConfiguration,
)


class BrittleStarDirectedLocomotionEnvironmentConfiguration(
    BrittleStarEnvironmentBaseConfiguration
):
    def __init__(
        self,
        target_distance: float = 3,
        joint_randomization_noise_scale: float = 0.0,
        color_contacts: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            joint_randomization_noise_scale=joint_randomization_noise_scale,
            color_contacts=color_contacts,
            *args,
            **kwargs,
        )
        self.target_distance = target_distance


class BrittleStarDirectedLocomotionEnvironmentBase(BrittleStarEnvironmentBase):
    def __init__(self) -> None:
        super().__init__()

    def _update_reward(
        self, state: BaseEnvState, previous_state: BaseEnvState
    ) -> BaseEnvState:
        previous_distance_to_target = self._get_xy_distance_to_target(
            state=previous_state
        )
        current_distance_to_target = self._get_xy_distance_to_target(state=state)
        reward = previous_distance_to_target - current_distance_to_target
        # noinspection PyUnresolvedReferences
        return state.replace(reward=reward)

    def _update_info(self, state: BaseEnvState) -> BaseEnvState:
        info = {
            "time": self._get_time(state=state),
            "xy_target_position": self._get_xy_target_position(state=state),
        }

        # noinspection PyUnresolvedReferences
        return state.replace(info=info)

    def _update_terminated(self, state: BaseEnvState) -> BaseEnvState:
        terminated = (
            self._get_xy_distance_to_target(state=state)
            < state.mj_model.site("target_site").size[0]
        )

        # noinspection PyUnresolvedReferences
        return state.replace(terminated=terminated)

    def _update_truncated(self, state: BaseEnvState) -> BaseEnvState:
        truncated = (
            self._get_time(state=state) > self.environment_configuration.simulation_time
        )

        # noinspection PyUnresolvedReferences
        return state.replace(truncated=truncated)

    @staticmethod
    @abc.abstractmethod
    def _get_xy_direction_to_target(state: BaseEnvState) -> chex.Array:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _get_xy_distance_to_target(state: BaseEnvState) -> float:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _get_xy_target_position(state: BaseEnvState) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def environment_configuration(
        self,
    ) -> BrittleStarDirectedLocomotionEnvironmentConfiguration:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _get_time(state: BaseEnvState) -> float:
        raise NotImplementedError
