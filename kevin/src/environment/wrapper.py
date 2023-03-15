from gymnasium import spaces
from pettingzoo.utils import BaseParallelWraper


class FlatteningWrapper(BaseParallelWraper):

    def seed(self, seed=None):
        return self.env.seed(seed)

    def observation_space(self, agent):
        return spaces.flatten_space(self.env.observation_space(agent))

    def step(self, actions):
        obs_dict, rew, done, trunc, info = self.env.step(actions)
        self.agents = self.env.agents

        for agent, obs in obs_dict.items():

            obs_dict[agent] = spaces.flatten(self.env.observation_space(agent), obs)

        return obs_dict, rew, done, trunc, info
