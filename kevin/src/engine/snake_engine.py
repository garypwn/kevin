from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Final


# noinspection PyMethodMayBeStatic
class GameState(ABC):
    """A game of snake. Contains methods to read the board state and submit moves for one or more snakes.
    Can be used for training or playing games.
    """

    #  Must be set by constructor
    height: Final[int]
    width: Final[int]
    player_count: Final[int]

    @abstractmethod
    def player_count(self) -> int:
        """
        How many players this board supports. This should be invariant.
        :return: The number of players supported by the game
        """
        pass

    @abstractmethod
    def get_observation(self, snake_id: str) -> dict:
        """
        A representation of the board state suitable for a state observation. See MultiSnakeEnv.
        :param snake_id: Which snake to flag as "you"
        :return: The board state
        """
        pass

    def get_reward(self, snake_id) -> float:
        """
        The reward function. Since we don't actually know the reward for a given move, this can just
        return 0 for now
        :param snake_id: The snake
        :return: The reward
        """
        return 0.

    @abstractmethod
    def get_terminated(self, snake_id) -> bool:
        """
        Gets whether a snake is terminated
        :param snake_id:
        :return:
        """
        pass

    @abstractmethod
    def get_truncated(self, snake_id) -> bool:
        """
        Gets whether a snake is truncated (due to a time limit)
        :param snake_id:
        :return:
        """
        pass

    def get_info(self, snake_id) -> dict:
        """
        Gets the info dictionary required by environments.
        :param snake_id: The snake id
        :return: An empty dictionary
        """
        return {}

    @abstractmethod
    def global_observation(self) -> dict:
        """
        A global (agnostic of snake perspective) representation of the board state. This is equal to
        the board state for any snake's perspective without any snake having the "you" flag set.
        :return: The board state.
        """
        pass

    @abstractmethod
    def step(self, moves: dict[str: int]) -> GameState:
        """
        Advances the game to the next step. This should result in new observations.
        :return: None
        """
        pass

    @abstractmethod
    def reset(self, options: dict | None = None) -> GameState:
        """
        Initialize a new board. Note: this must be deterministic if used for training!
        This will always be called when the environment is constructed.
        :return: None
        """
        pass

    def seed(self, seed) -> None:
        """
        Must be overriden if this class is used for training!
        Sets the rng for the game, making it deterministic.
        If used for training, a deterministic prng like ``jax.random`` must be used.
        :param seed: The prng seed
        :return: None
        """
        raise NotImplementedError("This board is not deterministic.")
