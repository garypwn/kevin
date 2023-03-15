import functools
from typing import Optional, Tuple, Dict, List, SupportsFloat, Any

from gymnasium import spaces, Env, Space
from gymnasium.core import RenderFrame, ActType, ObsType
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

    metadata = {"render_modes": [], "name": "battlesnake_v0"}

    def __init__(self, eng: SnakeEngine):
        self.game = eng

        self.possible_agents = ["snake_" + str(r) for r in range(eng.player_count)]
        self.reset()

        self.action_spaces = {agent: spaces.Discrete(4) for agent in self.possible_agents}
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}

    def action_space(self, agent):
        return self.action_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> spaces.Space:
        return spaces.Dict(
            {
                "snakes": spaces.Tuple([spaces.Tuple(
                    [
                        spaces.Box(0, 100, dtype=jnp.int16),
                        spaces.Box(0, 1, dtype=jnp.int16),
                    ])
                    for _ in range(self.game.player_count)]),  # Number of snakes

                "turn": spaces.Box(low=0, high=jnp.inf, dtype=jnp.int16),  # Limit 32k turns... should be enough.

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

        if seed is not None:
            self.game.seed(seed)

        self.game.reset()

        #  For now, we train with max players on a board.
        #  todo train with varying number of agents?
        self.agents = ["snake_" + str(r) for r in range(self.game.player_count)]

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
        for agent in self.agents[:]:
            if terminations[agent] or truncations[agent]:
                self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None | jnp.ndarray | str | List:
        return self.game.__str__()

    def state(self) -> jnp.ndarray:
        return self.game.global_observation()["board"]  # Not very useful


class DummyGymEnv(Env):
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    def __init__(self, env):
        r"""
        Coax (and other RL libraries) often take a gym environment to validate action and observation spaces.
        Since the spaces are the same for each snake, we can create a dummy gym environment
        to satisfy these requirements.

        Note that this is not a real gym environment and does not have any methods implemented.
        """

        self.action_space = env.action_space("snake_0")
        self.observation_space = env.observation_space("snake_0")

    action_space: Space
    observation_space: Space
