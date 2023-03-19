import cProfile
import os
from datetime import datetime
from time import sleep

import coax
import optax
from coax.value_losses import mse

from kevin.src.engine.python_engine import BoardUpdater, PythonGameState
from kevin.src.environment.rewinding_environment import RewindingEnv
from kevin.src.environment.snake_environment import MultiSnakeEnv, DummyGymEnv
from kevin.src.model.model import Model, residual_body

# set some env vars
os.environ.setdefault('JAX_PLATFORMS', 'gpu, cpu')  # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


name = 'standard_4p_ppo'

updater = BoardUpdater(11, 11, 4)
game = PythonGameState(updater=updater)
env = RewindingEnv(game)
env.fancy_render = True
gym_env = DummyGymEnv(env)

model = Model(residual_body, gym_env.action_space)


# Optimizers
optimizer_q = optax.chain(optax.apply_every(k=4), optax.adam(0.005))
optimizer_pi = optax.chain(optax.apply_every(k=4), optax.adam(0.005))

# q = coax.Q(func_q, gym_env)
v = coax.V(model.v, gym_env)
pi = coax.Policy(model.pi_logits, gym_env)

pi_behavior = pi.copy()
v_targ = v.copy()

# One tracer for each agent
tracers = {agent: coax.reward_tracing.NStep(n=9, gamma=0.9) for agent in env.possible_agents}

# We just need one buffer ...?
buffer = coax.experience_replay.SimpleReplayBuffer(capacity=2048)

# Regularizer
pi_regularizer = coax.regularizers.EntropyRegularizer(pi, 0.001)

# Updaters
ppo_clip = coax.policy_objectives.PPOClip(pi, optimizer=optimizer_pi, regularizer=pi_regularizer)
simple_td = coax.td_learning.SimpleTD(v, v_targ, optimizer=optimizer_q, loss_function=mse)

epoch_num = 0
new_epoch = True
checkpoint_period = 15

profiler = cProfile.Profile()
verbose = False
for i in range(10000000):

    # Adjust hypers in later epochs
    if epoch_num == 30:

        # Learning rate annealing
        ppo_clip.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(0.001))
        simple_td.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(0.001))

    if epoch_num == 60:

        # Learning rate annealing
        ppo_clip.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(0.0001))
        simple_td.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(0.0001))

    if i == -1:
        profiler.enable()

    if i == -1:
        profiler.disable()
        profiler.dump_stats("ppo_profile.prof")

    # Episode of single-player
    obs = env.reset(i)
    cum_reward = {agent: 0. for agent in env.possible_agents}
    if verbose:
        print(env.render())

    while len(env.agents) > 0:

        # Get actions
        actions = {}
        logps = {}
        for agent in env.agents:
            actions[agent], logps[agent] = pi_behavior(obs[agent], return_logp=True)

        live_agents = env.agents[:]
        obs_next, rewards, terminations, truncations, _ = env.step(actions)

        if verbose:
            print(env.render())
            sleep(0.75)

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
                buffer.add(tracer.pop())

        if len(buffer) >= buffer.capacity:
            for _ in range(4 * buffer.capacity // 32):  # 4 rounds
                transition_batch = buffer.sample(batch_size=32)
                metrics_v, td_error = simple_td.update(transition_batch, return_td_error=True)
                metrics_pi = ppo_clip.update(transition_batch, td_error)

            buffer.clear()
            pi_behavior.soft_update(pi, tau=0.1)
            v_targ.soft_update(v, tau=0.1)

            new_epoch = True
            epoch_num += 1

        obs = obs_next
    if verbose:
        print("===== End Game {}. Epoch {} =========================".format(i, epoch_num))

    if new_epoch:
        new_epoch = False

        # Flush the stack so that we don't get a stinky replay
        env.state_pq = []

        print("===== Game {}. Epoch {} =========================".format(i, epoch_num))
        obs = env.reset(i // 15)
        print(env.render())
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

    if epoch_num % checkpoint_period == 0 and epoch_num != 0:
        print("======Checkpoint {}============================".format(epoch_num))
        print("-----------------------------------------------")
        now = datetime.today().strftime('%Y-%m-%d')
        coax.utils.dump([pi, pi_behavior, v, tracers, buffer, pi_regularizer, simple_td, ppo_clip],
                        ".checkpoint/{}_epoch_{}_{}.pkl.lz4".format(name, epoch_num, now))

