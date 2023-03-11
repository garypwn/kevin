from abc import abstractmethod, ABC
from typing import List, Tuple, Dict, Optional

import gymnasium
import jax.numpy as jnp
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ActionDict, ObsDict


class SnakeGameState:
    r"""A board state in a game of snake."""

    turn: int
    dimensions: (int, int)
    food: list[(int, int)]
    hazards: list[(int, int)]
    snakes: list[list[(int, int)]]


class SnakeEngine(ABC):
    r"""A game of snake. Contains methods to read the board state and submit moves for one or more snakes."""

    @abstractmethod
    def board(self) -> SnakeGameState:
        pass

    @abstractmethod
    def submit_move(self, snake_id):
        pass


class MultiSnakeEnv(ParallelEnv):
    r"""Takes a SnakeEngine and maps it to a pettingzoo environment.
    Observations are of the form
        "snakes": [id: 0-15]["health": 0-100, "you": T-F]
        "board": [x: 0-w][y:0-h][cell: empty=0, food=1, hazard=2, snake head = 2(id)+3, body= 2(id)+4 => 0 .. 30]
    """

    snake: SnakeEngine

    def __init__(self, eng: SnakeEngine):
        self.snake = eng

        self.possible_agents = ["snake_" + str(r) for r in range(16)]
        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}

    def observation_space(self, agent) -> gymnasium.spaces.Space:
        return spaces.Dict(
            # todo: expand to variable sized board, larger number of snakes
            {
                "snakes": spaces.Tuple(((spaces.Dict(
                    {
                        "health": spaces.Box(0, 100, dtype=int),
                        "you": spaces.Discrete(2)
                    }
                )),)*4),  # There are 4 snakes for now

                #  Board dimensions
                "board": spaces.Box(low=jnp.ndarray([0, 0]), high=jnp.ndarray([10, 10]), dtype=int),
            }
        )

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:
        pass

    def seed(self, seed=None):
        pass

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        pass

    def render(self) -> None | jnp.ndarray | str | List:
        pass

    def state(self) -> jnp.ndarray:
        pass
