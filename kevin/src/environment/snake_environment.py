import functools
from typing import Optional, Tuple, Dict, List

from gymnasium import spaces
from jax import numpy as jnp
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ObsDict, ActionDict

from kevin.src.engine.snake_engine import SnakeEngine


# noinspection SpellCheckingInspection
class MultiSnakeEnv(ParallelEnv):
    r"""Takes a SnakeEngine and maps it to a pettingzoo environment.
    Observations are of the form
        "snakes": [id: 0-15]["health": 0-100, "you": T-F]
        "turn": [0-inf]
        "board": [x: 0-w][y:0-h][cell: empty=0, food=1, hazard=2, snake head = 2(id)+3, body= 2(id)+4 => 0 .. 30]

    Snake ids are always of the form "snake_i" for i in [0,max snakes]

    """

    game: SnakeEngine

    def __init__(self, eng: SnakeEngine):
        self.game = eng

        self.possible_agents = ["snake_" + str(r) for r in range(eng.player_count)]

        #  For now we train with max players on a board.
        #  todo train with varying number of agents?
        self.agents = ["snake_" + str(r) for r in range(eng.player_count)]

        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> spaces.Space:
        return spaces.Dict(
            {
                "snakes": spaces.Tuple([spaces.Dict(
                    {
                        "health": spaces.Box(0, 100, dtype=jnp.int16),
                        "you": spaces.Discrete(2)
                    })
                    for _ in range(self.game.player_count)]),  # Number of snakes

                "turn": spaces.Box(low=0, high=jnp.inf, dtype=jnp.int16),

                #  Board dimensions
                "board": spaces.Box(low=np.zeros([self.game.width, self.game.height], dtype=int),
                                    high=np.full([self.game.width, self.game.height],

                                                 #  Max value is the max value of a snake body cell
                                                 3 * self.game.player_count + 5,
                                                 dtype=int),
                                    shape=[self.game.width, self.game.height],
                                    dtype=jnp.int16),
            }
        )

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:

        if seed:
            self.game.seed(seed)

        self.game.reset()

        observations = {agent: self.game.get_observation(agent) for agent in self.agents}
        if not return_info:
            return observations

        else:
            infos = {agent: self.game.get_info(agent) for agent in self.agents}
            return infos

    def seed(self, seed=None):
        if seed is not None:
            self.game.seed(seed)

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:
        for agent, action in actions.items():
            self.game.submit_move(agent, action)

        self.game.step()

        observations = {agent: self.game.get_observation(agent) for agent in self.agents}
        rewards = {agent: self.game.get_reward(agent) for agent in self.agents}
        terminations = {agent: self.game.get_terminated(agent) for agent in self.agents}
        truncations = {agent: self.game.get_truncated(agent) for agent in self.agents}
        infos = {agent: self.game.get_info(agent) for agent in self.agents}

        # Remove terminated or truncated agents? This is not well-defined by spec
        for agent in self.agents:
            if terminations[agent]:
                self.agents.remove(agent)
                continue
            if truncations[agent]:
                self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None | jnp.ndarray | str | List:
        return None  # Not supported right now

    def state(self) -> jnp.ndarray:
        return self.game.global_observation()["board"]  # Not very useful
