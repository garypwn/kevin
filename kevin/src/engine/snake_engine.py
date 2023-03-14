from abc import abstractmethod, ABC
from typing import Final


# noinspection PyMethodMayBeStatic
class SnakeEngine(ABC):
    r"""A game of snake. Contains methods to read the board state and submit moves for one or more snakes.
    Can be used for training or playing games.
    """

    #  Must be set by constructor
    height: Final[int]
    width: Final[int]
    player_count: Final[int]

    @abstractmethod
    def player_count(self) -> int:
        r"""
        How many players this board supports. This should be invariant.
        :return: The number of players supported by the game
        """
        pass

    @abstractmethod
    def get_observation(self, snake_id: str) -> dict:
        r"""
        A representation of the board state suitable for a state observation. See MultiSnakeEnv.
        :param snake_id: Which snake to flag as "you"
        :return: The board state
        """
        pass

    def get_reward(self, snake_id) -> float:
        r"""
        The reward function. Since we don't actually know the reward for a given move, this can just
        return 0 for now
        :param snake_id: The snake
        :return: The reward
        """
        return 0.

    @abstractmethod
    def get_terminated(self, snake_id) -> bool:
        r"""
        Gets whether a snake is terminated
        :param snake_id:
        :return:
        """
        pass

    @abstractmethod
    def get_truncated(self, snake_id) -> bool:
        r"""
        Gets whether a snake is truncated (due to a time limit)
        :param snake_id:
        :return:
        """
        pass

    def get_info(self, snake_id) -> dict:
        r"""
        Gets the info dictionary required by environments.
        :param snake_id: The snake id
        :return: An empty dictionary
        """
        return {}

    @abstractmethod
    def global_observation(self) -> dict:
        r"""
        A global (agnostic of snake perspective) representation of the board state. This is equal to
        the board state for any snake's perspective without any snake having the "you" flag set.
        :return: The board state.
        """
        pass

    @abstractmethod
    def submit_move(self, snake_id, move: int) -> None:
        r"""
        Submits a move for a snake. Moves are [0,3] such that move= [u, r, d, l][i]
        :param move: The integer representation of the move.
        :param snake_id: The snake to submit the move for.
        :return: None
        """
        pass

    @abstractmethod
    def step(self) -> None:
        r"""
        Advances the game to the next step. This should result in new observations.
        :return: None
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        r"""
        Initialize a new board. Note: this must be deterministic if used for training!
        This will always be called when the environment is constructed.
        :return: None
        """
        pass

    def seed(self, seed) -> None:
        r"""
        Must be overriden if this class is used for training!
        Sets the rng for the game, making it deterministic.
        If used for training, a deterministic prng like ``jax.random`` must be used.
        :param seed: The prng seed
        :return: None
        """
        raise NotImplementedError("This board is not deterministic.")
