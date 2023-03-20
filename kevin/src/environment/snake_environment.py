import functools
from typing import Optional, Tuple, Dict, List, SupportsFloat, Any, Callable

from gymnasium import spaces, Env, Space
from gymnasium.core import RenderFrame, ActType, ObsType
from jax import numpy as jnp
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import ObsDict, ActionDict

from kevin.src.engine import utils
from kevin.src.engine.python_engine import PythonGameState
from kevin.src.engine.snake_engine import GameState


# noinspection SpellCheckingInspection
class MultiSnakeEnv(ParallelEnv):
    r"""Takes a SnakeEngine and maps it to a pettingzoo environment.
    Observations are of the form
        "snakes": [id: 0-15]["health": 0-100, "you": T-F]
        "turn": [0-inf]
        "board": [x: 0-w][y:0-h][cell: empty=0, food=1, hazard=2, snake head = 2(id)+3, body= 2(id)+4 => 0 .. 30]

    Snake ids are always of the form "snake_i" for i in [0,max snakes]

    """

    game: GameState
    fancy_render = False

    metadata = {"render_modes": [], "name": "battlesnake_v0"}

    temp_step_result = None
    temp_reset_result = None

    def get_temp_step_result(self):
        return self.temp_step_result

    def get_temp_reset_result(self):
        return self.temp_reset_result

    dummy_gym_environment: Env

    def __init__(self, eng: GameState):
        self.game = eng

        self.possible_agents = ["snake_" + str(r) for r in range(eng.player_count)]
        self.reset()

        self.action_spaces = {agent: spaces.Discrete(3) for agent in self.possible_agents}
        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.agents}

        self.dummy_gym_environment = DummyGymEnv(self.action_spaces["snake_0"], self.observation_spaces["snake_0"],
                                                 self.get_temp_step_result, self.get_temp_reset_result)

    def action_space(self, agent):
        return spaces.Discrete(3)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> spaces.Space:
        viewport = 1 + 2 * max(self.game.width, self.game.height)
        return spaces.Box(0, self.game.width * self.game.height,
                          shape=[self.game.player_count + 3, viewport, viewport], dtype=jnp.int16)

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:

        if seed is not None:
            self.game.seed(seed)

        self.game = self.game.reset(options)

        #  For now, we train with max players on a board.
        #  todo train with varying number of agents?
        self.agents = ["snake_" + str(r) for r in range(self.game.player_count)]

        observations = {agent: self.game.get_observation(agent) for agent in self.agents}

        snake_n = self.agents[0]
        if not return_info:
            self.temp_reset_result = observations[snake_n]
            return observations

        else:
            infos = {agent: self.game.get_info(agent) for agent in self.agents}
            self.temp_reset_result = infos[snake_n]
            return infos

    def seed(self, seed=None):
        if seed is not None:
            self.game.seed(seed)

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:

        self.game = self.game.step(actions)

        observations = {agent: self.game.get_observation(agent) for agent in self.agents}
        rewards = {agent: self.game.get_reward(agent) for agent in self.agents}
        terminations = {agent: self.game.get_terminated(agent) for agent in self.agents}
        truncations = {agent: self.game.get_truncated(agent) for agent in self.agents}
        infos = {agent: self.game.get_info(agent) for agent in self.agents}

        snake_n = self.agents[0]
        self.temp_step_result = (observations[snake_n], rewards[snake_n], terminations[snake_n],
                                 truncations[snake_n], infos[snake_n])

        # Remove terminated or truncated agents? This is not well-defined by spec
        for agent in self.agents[:]:
            if terminations[agent] or truncations[agent]:
                self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None | jnp.ndarray | str | List:
        if self.fancy_render:
            board = utils.fancy_board_from_game(self.game)
            turn = self.game.turn_num
            snakes = {name: snake.health for name, snake in self.game.snakes.items()}

            if isinstance(self.game, PythonGameState) and self.game.branch_name != "":
                branch = self.game.branch_name
                return "\nBranch {}: Turn {}.\n{}\n{}\n".format(branch, turn, snakes, board)

            return "\nTurn {}.\n{}\n{}\n".format(turn, snakes, board)
        return self.game.__str__()

    def state(self) -> jnp.ndarray:
        return self.game.global_observation()["board"]  # Not very useful


class DummyGymEnv(Env):
    action_space: Space
    observation_space: Space
    step_fn: Callable
    reset_fn: Callable

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        To record metrics properly, this must be called after the base environment
        """

        return self.step_fn()

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        """
        To record metrics properly, this must be called after the base environment
        """

        return self.reset_fn()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        raise NotImplementedError

    def __init__(self, action_space, observation_space, step_fn, reset_fn):
        r"""
        Coax (and other RL libraries) often take a gym environment to validate action and observation spaces.
        Since the spaces are the same for each snake, we can create a dummy gym environment
        to satisfy these requirements.

        Note that this is not a real gym environment and does not have any methods implemented.
        """

        self.action_space = action_space
        self.observation_space = observation_space
        self.step_fn = step_fn
        self.reset_fn = reset_fn
