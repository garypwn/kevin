import cProfile
import os
import pstats

import coax
import haiku as hk
import jax.nn
import jax.numpy as jnp
import optax
from coax.value_losses import mse, huber

from kevin.src.engine.python_engine import BoardUpdater, PythonGameState
from kevin.src.environment.snake_environment import MultiSnakeEnv, DummyGymEnv
from kevin.src.environment.wrapper import FlatteningWrapper
from kevin.src.model.model import Model, process_obs, resnet

# set some env vars
os.environ.setdefault('JAX_PLATFORMS', 'gpu, cpu')  # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet

name = 'standard_4p_ppo'

updater = BoardUpdater(11, 11, 4)
game = PythonGameState(updater=updater)
env = MultiSnakeEnv(game)
env.fancy_render = True
gym_env = DummyGymEnv(env)

model = Model(resnet, gym_env.action_space)


# Optimizers
optimizer_q = optax.chain(optax.apply_every(k=4), optax.adam(0.0002))
optimizer_pi = optax.chain(optax.apply_every(k=4), optax.adam(0.0001))

# q = coax.Q(func_q, gym_env)
v = coax.V(model.v, gym_env)
pi = coax.Policy(model.pi_logits, gym_env)

pi_behavior = pi.copy()

# One tracer for each agent
tracers = {agent: coax.reward_tracing.NStep(n=6, gamma=0.9) for agent in env.possible_agents}

# We just need one buffer ...?
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=4096)

# Regularizer
pi_regularizer = coax.regularizers.EntropyRegularizer(pi, 0.001)

# Updaters
ppo_clip = coax.policy_objectives.PPOClip(pi, optimizer=optimizer_pi, regularizer=pi_regularizer)
simple_td = coax.td_learning.SimpleTD(v, loss_function=huber, optimizer=optimizer_q)

epoch_num = 0
new_epoch = True
checkpoint_period = 2500

profiler = cProfile.Profile()
for i in range(10000000):

    if i == -1:
        profiler.enable()

    if i == -1:
        profiler.disable()
        profiler.dump_stats("ppo_profile.prof")

    # Episode of single-player
    obs = env.reset(i // 15)
    cum_reward = {agent: 0. for agent in env.possible_agents}

    while len(env.agents) > 0:

        # Get actions
        actions = {}
        logps = {}
        for agent in env.agents:
            actions[agent], logps[agent] = pi_behavior(obs[agent], return_logp=True)

        live_agents = env.agents[:]
        obs_next, rewards, terminations, truncations, _ = env.step(actions)

        # Trace rewards
        for agent in live_agents:
            tracers[agent].add(
                obs[agent],
                actions[agent],
                rewards[agent],
                terminations[agent] or truncations[agent],
                logps[agent]
            )
            cum_reward[agent] += rewards[agent]

        # Update
        for _, tracer in tracers.items():
            while tracer:
                transition_batch = tracer.pop()
                buffer.add(transition_batch)
                metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
                metrics_pi = ppo_clip.update(transition_batch, td_error)

        if len(buffer) == buffer.capacity:
            for _ in range(16 * buffer.capacity // 64):
                transition_batch = buffer.sample(batch_size=64)
                metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
                metrics_pi = ppo_clip.update(transition_batch, td_error)

            buffer.clear()
            pi_behavior.soft_update(pi, tau=0.1)

            new_epoch = True
            epoch_num += 1

        obs = obs_next

    if new_epoch:
        new_epoch = False

        print("===== Game {}. Epoch {} =========================".format(i, epoch_num))
        print(env.render())
        obs = env.reset(i // 15)
        cum_reward = {agent: 0. for agent in env.possible_agents}

        while len(env.agents) > 0:
            # Get actions
            actions = {}
            for agent in env.agents:
                actions[agent] = pi_behavior.mode(obs[agent])

            live_agents = env.agents[:]
            obs_next, rewards, terminations, truncations, _ = env.step(actions)
            for agent in live_agents:
                cum_reward[agent] += rewards[agent]

            print(env.render())
            obs = obs_next

        print("===== End Game {}. Epoch {} =========================".format(i, epoch_num))
        print("Rewards: {}".format({s: "{:.2f}".format(r) for s, r in cum_reward.items()}))
        print("-----------------------------------------------------")

    if i % checkpoint_period == 0 and i != 0:
        print("======Checkpoint {}============================".format(i))
        print("-----------------------------------------------")
        coax.utils.dump([pi, pi_behavior, v, tracers, buffer, pi_regularizer, simple_td, ppo_clip],
                        ".checkpoint/{}_checkpoint_{}.pkl.lz4".format(name, i))

