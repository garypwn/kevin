import os

import coax
import haiku
import jax.nn
import jax.numpy as jnp

from kevin.src.engine.python_engine import BoardUpdater, PythonStandard4Player
from kevin.src.environment.snake_environment import MultiSnakeEnv, DummyGymEnv
from kevin.src.environment.wrapper import FlatteningWrapper

# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')  # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet

name = 'standard_4p_ppo'

updater = BoardUpdater(11, 11, 4)
game = PythonStandard4Player(updater=updater)
base_env = MultiSnakeEnv(game)
env = FlatteningWrapper(base_env)
gym_env = DummyGymEnv(env)


def func_q(S, is_training):
    seq = haiku.Sequential((
        jnp.log1p,
        haiku.Linear(8, w_init=jnp.zeros), jax.nn.relu,
        haiku.Linear(gym_env.action_space.n, w_init=jnp.zeros)
    ))
    return seq(S)


q = coax.Q(func_q, gym_env)
pi = coax.BoltzmannPolicy(q, temperature=0.1)

# One tracer for each agent
tracers = {agent: coax.reward_tracing.NStep(n=8, gamma=0.9) for agent in env.possible_agents}

# Updater
qlearning = coax.td_learning.QLearning(q)

for i in range(1000000):

    # Episode
    obs = env.reset()

    if i % 800 == 0:
        print("===== Game {} =========================".format(i))
        print(env.render())

    while len(env.agents) > 0:
        a_dict = {}

        # Submit actions
        for agent in env.agents:
            a_dict[agent] = pi(obs[agent])

        obs_next, r_dict, terminations, truncations, infos = env.step(a_dict)

        # Trace rewards
        for agent in env.agents:
            tracers[agent].add(obs[agent], a_dict[agent], r_dict[agent], terminations[agent] or truncations[agent])

        # Update q-learning
        for _, tracer in tracers.items():
            while tracer:
                qlearning.update(tracer.pop())

        if i % 800 == 0:
            print(env.render())
        obs = obs_next
