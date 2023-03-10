from typing import List, Tuple, Dict, Optional

import jax.numpy as jnp
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ActionDict, ObsDict


class SnakeEngine:
    r"""A game of snake. Contains methods to read the board state and submit moves for one or more snakes."""
    pass


class MultiSnakeEnv(ParallelEnv):
    r"""Takes a SnakeEngine and maps it to a `pettingzoo` environment"""

    snake: SnakeEngine

    def __init__(self, eng: SnakeEngine):
        self.snake = eng

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
