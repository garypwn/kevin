import os

import coax
import haiku as hk
import jax.nn
import jax.numpy as jnp
import optax
from coax.value_losses import mse, huber

from kevin.src.engine.python_engine import BoardUpdater, PythonStandard4Player
from kevin.src.environment.snake_environment import MultiSnakeEnv, DummyGymEnv
from kevin.src.environment.wrapper import FlatteningWrapper

# set some env vars
os.environ.setdefault('JAX_PLATFORMS', 'gpu, cpu')  # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet

name = 'standard_4p_a2c'

updater = BoardUpdater(11, 11, 4)
game = PythonStandard4Player(updater=updater)
base_env = MultiSnakeEnv(game)
env = FlatteningWrapper(base_env)
gym_env = DummyGymEnv(env)


def func_v(S, is_training):
    val = hk.Sequential((
        jnp.float32,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(8), jax.nn.relu,
        hk.Linear(1, w_init=jnp.zeros), jnp.ravel
    ))
    return val(S)


def func_q(S, is_training):
    seq = hk.Sequential((
        jnp.float32,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(16), jax.nn.relu,
        hk.Linear(gym_env.action_space.n, w_init=jnp.zeros)
    ))
    return seq(S)


def func_pi(S, is_training):
    logits = hk.Sequential((
        jnp.float32,
        hk.Linear(256), jax.nn.relu,
        hk.Linear(128), jax.nn.relu,
        hk.Linear(64), jax.nn.relu,
        hk.Linear(16), jax.nn.relu,
        hk.Linear(gym_env.action_space.n, w_init=jnp.zeros)
    ))
    return {'logits': logits(S)}


# Optimizers
#  optimizer_v = optax.chain(optax.apply_every(k=4), optax.adam(0.001))
optimizer_q = optax.chain(optax.apply_every(k=4), optax.adam(0.001))
optimizer_pi = optax.chain(optax.apply_every(k=4), optax.adam(0.0005))

#  v = coax.V(func_v, gym_env)
q = coax.Q(func_q, gym_env)
pi = coax.Policy(func_pi, gym_env)

# One tracer for each agent
tracers = {agent: coax.reward_tracing.NStep(n=30, gamma=0.9) for agent in env.possible_agents}

# Regularizer
pi_regularizer = coax.regularizers.EntropyRegularizer(pi, 0.007)

# Updaters
vanilla = coax.policy_objectives.VanillaPG(pi, optimizer=optimizer_pi, regularizer=pi_regularizer)
# simple_td = coax.td_learning.SimpleTD(v, loss_function=huber, optimizer=optimizer_v)
sarsa = coax.td_learning.Sarsa(q, loss_function=huber, optimizer=optimizer_q)


# Train
for i in range(1000000):

    render_period = 150

    # Episode
    obs = env.reset(i)
    cum_reward = {agent: 0. for agent in env.possible_agents}

    if i % render_period == 0:
        print("===== Game {} =========================".format(i))
        print(env.render())

    while len(env.agents) > 0:

        # Submit actions and store for later
        a_dict = {}
        for agent in env.agents:
            if i % render_period == 0:
                a_dict[agent] = pi.mode(obs[agent])
            else:
                a_dict[agent] = pi(obs[agent])

        live_agents = env.agents[:]

        obs_next, r_dict, terminations, truncations, _ = env.step(a_dict)

        # Trace rewards
        for agent in live_agents:
            if i % render_period != 0:
                tracers[agent].add(obs[agent], a_dict[agent], r_dict[agent], terminations[agent] or truncations[agent])

            cum_reward[agent] += r_dict[agent]

        # Update q-learning
        for _, tracer in tracers.items():
            while tracer:
                transition_batch = tracer.pop()
                metrics_v, td_error = sarsa.update(transition_batch, return_td_error=True)
                metrics_pi = vanilla.update(transition_batch, td_error)

        if i % render_period == 0:
            print(env.render())
        obs = obs_next

    if i % (render_period // 10) == 0:
        s = {agent: "{:.2f}".format(reward) for agent, reward in cum_reward.items()}
        print("\nEpisode {} cumulative rewards:\n{}".format(i, s))
