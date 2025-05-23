import abc

from moojoco.environment.base import BaseEnvState

from biorobot.brittle_star.environment.shared.base import (
    BrittleStarEnvironmentBase,
    BrittleStarEnvironmentBaseConfiguration,
)


class BrittleStarUndirectedLocomotionEnvironmentConfiguration(
    BrittleStarEnvironmentBaseConfiguration
):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )


class BrittleStarUndirectedLocomotionEnvironmentBase(BrittleStarEnvironmentBase):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _update_terminated(state: BaseEnvState) -> BaseEnvState:
        # noinspection PyUnresolvedReferences
        return state.replace(terminated=False)

    def _update_truncated(self, state: BaseEnvState) -> BaseEnvState:
        truncated = (
            self._get_time(state=state) > self.environment_configuration.simulation_time
        )
        # noinspection PyUnresolvedReferences
        return state.replace(truncated=truncated)

    def _update_reward(
        self, state: BaseEnvState, previous_state: BaseEnvState
    ) -> BaseEnvState:
        previous_distance_from_origin = self._get_xy_distance_from_origin(
            state=previous_state
        )
        current_distance_from_origin = self._get_xy_distance_from_origin(state=state)
        reward = current_distance_from_origin - previous_distance_from_origin

        # noinspection PyUnresolvedReferences
        return state.replace(reward=reward)

    def _update_info(self, state: BaseEnvState) -> BaseEnvState:
        info = state.info
        info.update(
            {
                "time": self._get_time(state=state),
                "xy_distance_from_origin": self._get_xy_distance_from_origin(
                    state=state
                ),
            }
        )

        # noinspection PyUnresolvedReferences
        return state.replace(info=info)

    @staticmethod
    @abc.abstractmethod
    def _get_time(state: BaseEnvState) -> float:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def _get_xy_distance_from_origin(state: BaseEnvState) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def environment_configuration(
        self,
    ) -> BrittleStarUndirectedLocomotionEnvironmentConfiguration:
        raise NotImplementedError
