import os

import coax
import haiku
import jax.numpy as jnp
import supersuit

from kevin.src.engine.python_engine import BoardUpdater, PythonStandard4Player
from kevin.src.environment.snake_environment import MultiSnakeEnv
from kevin.src.environment.wrapper import FlatteningWrapper

# set some env vars
os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')  # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet

name = 'standard_4p_ppo'

updater = BoardUpdater(11, 11, 4)
game = PythonStandard4Player(updater=updater)
base_env = MultiSnakeEnv(game)
gym_env = base_env.gym_environment()
env = FlatteningWrapper(base_env)


def func_pi(S, is_training):
    lin = haiku.Linear(4, w_init=jnp.zeros)
    return lin(S)


def func_v(S, is_training):
    # Value function.

    lin = haiku.Linear(4, w_init=jnp.zeros)
    return lin(S)


v = coax.V(func_v, gym_env)
pi = coax.Policy(func_pi, gym_env)

obs = env.reset()
print(env.render())
while len(env.agents) > 0:
    a_dict = {}

    # Submit actions
    for agent in env.agents:
        a_dict[agent] = pi(obs[agent])

    obs_next, r_dict, terminations, truncations, infos = env.step(a_dict)
    print(env.render())

    # todo: Update policy here

    obs = obs_next
