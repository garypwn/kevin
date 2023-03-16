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
env.fancy_render = True
gym_env = DummyGymEnv(env)


def process_obs(x):
    board = jnp.float32(x["board"])
    turn = jnp.float32(x["turn"])
    snakes = jnp.float32(x["snakes"])
    flat_board = hk.Flatten()(board)

    # Process the board
    conv = hk.Sequential((
        hk.Conv2D(24 + 24 * game.player_count, [3, 3]), jax.nn.relu,  # Adjacent objects
        hk.Conv2D(24 * game.player_count, [4, 4]),  # Snakes that can reach each other next turn
        # hk.Conv2D(24 + 16 * game.player_count, [5, 5])(board),  # Objects that can be reached in 2 turns
        hk.Flatten()
    ))

    result = jnp.concatenate((turn, snakes, flat_board, conv(board)), 1)

    mlp = hk.nets.MLP([512, 256, 64, 16])
    return mlp(result)


def func_q(S, is_training):
    seq = hk.Sequential((
        process_obs,
        hk.Linear(gym_env.action_space.n),
    ))
    return seq(S)


def func_v(S, is_training):
    value = hk.Sequential((
        process_obs,
        hk.Linear(1),
    ))
    return jnp.ravel(value(S))


def func_pi(S, is_training):
    logits = hk.Sequential((
        process_obs,
        hk.Linear(gym_env.action_space.n),
    ))
    return {'logits': logits(S)}


# Optimizers
optimizer_q = optax.chain(optax.apply_every(k=4), optax.adam(0.0001))
optimizer_pi = optax.chain(optax.apply_every(k=4), optax.adam(0.0001))

# q = coax.Q(func_q, gym_env)
v = coax.V(func_v, gym_env)
pi = coax.Policy(func_pi, gym_env)

pi_behavior = pi.copy()

# One tracer for each agent
tracers = {agent: coax.reward_tracing.NStep(n=1, gamma=0.9) for agent in env.possible_agents}

# We just need one buffer ...?
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=256)

# Regularizer
pi_regularizer = coax.regularizers.EntropyRegularizer(pi, 0.01)

# Updaters
ppo_clip = coax.policy_objectives.PPOClip(pi, optimizer=optimizer_pi, regularizer=pi_regularizer)
simple_td = coax.td_learning.SimpleTD(v, loss_function=huber, optimizer=optimizer_q)

render_period = 25

for i in range(100000):
    # Episode of single-player
    obs = env.reset(i // 20, options={"single_player": True})
    cum_reward = 0.

    if i % render_period == 0:
        print("===== Game {} =========================".format(i))
        print(env.render())

    for _ in range(400):  # There's no way it goes past 400 steps... ?
        if i % render_period == 0:
            a = pi_behavior.mode(obs["snake_0"])
            logp = None
        else:
            a, logp = pi_behavior(obs["snake_0"], return_logp=True)

        obs_next, r_dict, terminations, truncations, _ = env.step({"snake_0": a})

        # Trace rewards
        if i % render_period != 0:
            tracers["snake_0"].add(obs["snake_0"], a, r_dict["snake_0"],
                                   terminations["snake_0"] or truncations["snake_0"], logp)

        cum_reward += r_dict["snake_0"]

        # Update
        while tracers["snake_0"]:
            transition_batch = tracers["snake_0"].pop()
            # buffer.add(transition_batch)
            metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
            metrics_pi = ppo_clip.update(transition_batch, td_error)

        pi_behavior.soft_update(pi, tau=0.1)

        if i % render_period == 0:
            print(env.render())
        obs = obs_next

        if terminations["snake_0"] or truncations["snake_0"]:
            break



    if cum_reward > -98.5:
        print("=====SINGLE PLAYER WIN! Moving on to multiplayer =======")
        break
