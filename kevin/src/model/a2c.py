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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet

name = 'standard_4p_a2c'

updater = BoardUpdater(11, 11, 4)
game = PythonStandard4Player(updater=updater)
env = MultiSnakeEnv(game)
gym_env = DummyGymEnv(env)


def process_obs(x):
    board = jnp.float32(x["board"])
    turn = jnp.float32(x["turn"])
    snakes = jnp.float32(x["snakes"])

    # Process the board
    conv = hk.Sequential((
        hk.Conv2D(24 + 24 * game.player_count, [3, 3]), jax.nn.relu,  # Adjacent objects
        hk.Conv2D(24 * game.player_count, [4, 4]),         # Snakes that can reach each other next turn
        # hk.Conv2D(24 + 16 * game.player_count, [5, 5])(board),  # Objects that can be reached in 2 turns
    ))

    result = jnp.concatenate((turn, snakes, conv(board)), 1)

    mlp = hk.nets.MLP([256, 64, 8])
    return mlp(result)


def func_q(S, is_training):
    seq = hk.Sequential((
        process_obs,
        hk.Linear(gym_env.action_space.n),
    ))
    return jnp.reshape(seq(S), [1, 4])


def func_pi(S, is_training):
    logits = hk.Sequential((
        process_obs,
        hk.Linear(gym_env.action_space.n),
    ))
    return {'logits': jnp.reshape(logits(S), [1, 4])}


# Optimizers
optimizer_q = optax.chain(optax.apply_every(k=4), optax.adam(0.001))
optimizer_pi = optax.chain(optax.apply_every(k=4), optax.adam(0.0005))

q = coax.Q(func_q, gym_env)
pi = coax.Policy(func_pi, gym_env)

# One tracer for each agent
tracers = {agent: coax.reward_tracing.NStep(n=5, gamma=0.9) for agent in env.possible_agents}

# We just need one buffer ...?
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)

# Regularizer
pi_regularizer = coax.regularizers.EntropyRegularizer(pi, 0.001)

# Updaters
vanilla = coax.policy_objectives.VanillaPG(pi, optimizer=optimizer_pi)
sarsa = coax.td_learning.Sarsa(q, loss_function=huber, optimizer=optimizer_q)


render_period = 150

for i in range(100000):
    # Episode of single-player
    obs = env.reset(i // 20, options={"single_player": True})
    cum_reward = 0.

    if i % render_period == 0:
        print("===== Game {} =========================".format(i))
        print(env.render())

    for _ in range(400):  # There's no way it goes past 400 steps... ?
        if i % render_period == 0:
            a = pi.mode(obs["snake_0"])
        else:
            a = pi(obs["snake_0"])

        obs_next, r_dict, terminations, truncations, _ = env.step({"snake_0": a})

        # Trace rewards
        if i % render_period != 0:
            tracers["snake_0"].add(obs["snake_0"], a, r_dict["snake_0"],
                                   terminations["snake_0"] or truncations["snake_0"])

        cum_reward += r_dict["snake_0"]

        # Put transition in buffer
        while tracers["snake_0"]:
            transition_batch = tracers["snake_0"].pop()
            buffer.add(transition_batch)

        # Update
        if len(buffer) == buffer.capacity:
            for _ in range(4 * buffer.capacity // 32):
                transition_batch = buffer.sample(batch_size=32)
                metrics_v, td_error = sarsa.update(transition_batch, return_td_error=True)
                metrics_pi = vanilla.update(transition_batch, td_error)

        if i % render_period == 0:
            print(env.render())
        obs = obs_next

        if terminations["snake_0"] or truncations["snake_0"]:
            break

    if i % (render_period // 10) == 0:
        print("\nEpisode {} cumulative rewards:\n{:.2f}".format(i, cum_reward))

    if cum_reward > -99.5:
        print("=====SINGLE PLAYER WIN! Moving on to multiplayer =======")
        break

for i in range(100000):
    # Episode of multiplayer
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

            # Put transition in buffer
            while tracers[agent]:
                transition_batch = tracers[agent].pop()
                buffer.add(transition_batch)

            # Update
            if len(buffer) == buffer.capacity:
                for _ in range(4 * buffer.capacity // 32):
                    transition_batch = buffer.sample(batch_size=32)
                    metrics_v, td_error = sarsa.update(transition_batch, return_td_error=True)
                    metrics_pi = vanilla.update(transition_batch, td_error)

        if i % render_period == 0:
            print(env.render())
        obs = obs_next

    if i % (render_period // 10) == 0:
        s = {agent: "{:.2f}".format(reward) for agent, reward in cum_reward.items()}
        print("\nEpisode {} cumulative rewards:\n{}".format(i, s))
