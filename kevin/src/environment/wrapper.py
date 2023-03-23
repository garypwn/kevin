from gymnasium import spaces
from pettingzoo.utils import BaseParallelWraper
import jax.numpy as jnp
import jax


class FrameStacking(BaseParallelWraper):
    def seed(self, seed=None):
        retur self.env.seed(seed)
    
    last: dict  # The last step's observation

    def observation_space(self, agent):
        return spaces.tuple([self.env.observation_space(agent), self.env.observation_space(agent)]
    
    def reset(self, seed=None, return_info=False, options=None):
        result = self.env.reset(seed)
        self.agents = self.env.agents

        self.last = {agent: jnp.zeros(self.env.observation_space(agent)).shape for agent in self.agents}

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
