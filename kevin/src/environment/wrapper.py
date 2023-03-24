from gymnasium import spaces
from pettingzoo.utils import BaseParallelWraper
import jax.numpy as jnp
import jax

from kevin.src.environment.snake_environment import DummyGymEnv, MultiSnakeEnv


class FrameStacking(BaseParallelWraper):
    """
    Returns a tuple containing the current and previous observations. If it's turn 0, both entries are duplicates.
    """

    def __init__(self, env: MultiSnakeEnv):
        super().__init__(env)
        self.dummy_gym_environment = DummyGymEnv(self.action_space("snake_0"), self.observation_space("snake_0"))

    def seed(self, seed=None):
        return self.env.seed(seed)

    last: dict | None = None  # The last step's observation

    @property
    def game(self):
        return self.env.game

    @game.setter
    def game(self, game):
        self.env.game = game

    def observation_space(self, agent):
        space = self.env.observation_space(agent)
        x1, x2, x3 = space.shape
        shape = (x1, x2, 2*x3)
        game = self.env.unwrapped.game

        return spaces.Box(0, game.width * game.height, shape, dtype=space.dtype)

    def reset(self, seed=None, return_info=False, options=None):
        result = self.env.reset(seed, return_info, options)
        self.agents = self.env.agents

        self.last = None
        if return_info:
            return result
        return self._wrap_obs(result)

    def step(self, actions):
        obs_dict, rew, done, trunc, info = self.env.step(actions)
        self.agents = self.env.agents

        return self._wrap_obs(obs_dict), rew, done, trunc, info

    def _wrap_obs(self, obs_dict):

        if self.last is None:
            self.last = obs_dict

        result = {name: jnp.concatenate([obs, self.last[name]], 2) for name, obs in obs_dict.items()}
        self.last = obs_dict

        return result

    def winner(self):
        return self.env.winner()


class FlatteningWrapper(BaseParallelWraper):

    def seed(self, seed=None):
        return self.env.seed(seed)

    def observation_space(self, agent):
        return spaces.flatten_space(self.env.observation_space(agent))

    def reset(self, seed=None, return_info=False, options=None):
        result = self.env.reset(seed)
        self.agents = self.env.agents

        if return_info:
            return result
        return self._wrap_obs(result)

    def step(self, actions):
        obs_dict, rew, done, trunc, info = self.env.step(actions)
        self.agents = self.env.agents

        return self._wrap_obs(obs_dict), rew, done, trunc, info

    def _wrap_obs(self, obs_dict):

        for agent, obs in obs_dict.items():
            obs_dict[agent] = spaces.flatten(self.env.observation_space(agent), obs)

        return obs_dict
