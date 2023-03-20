import os
import random
from datetime import datetime
from time import sleep

import coax
import optax
from coax.value_losses import mse

from kevin.src.engine.python_engine import RotatingBoardUpdater, PythonGameState
from kevin.src.environment.rewinding_environment import RewindingEnv
from kevin.src.model.model import Model, residual_body

# set some env vars
os.environ.setdefault('JAX_PLATFORMS', 'gpu, cpu')  # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


class PPOModel:
    name = 'standard_4p_ppo'

    updater = RotatingBoardUpdater(11, 11, 4)
    game = PythonGameState(updater=updater)
    env = RewindingEnv(game)
    env.fancy_render = True
    gym_env = env.dummy_gym_environment

    def __init__(self):
        self.model = None
        self.pi_regularizer = None
        self.v_targ = None
        self.simple_td = None
        self.ppo_clip = None
        self.buffer = None
        self.v = None
        self.tracers = None
        self.pi = None
        self.pi_behavior = None

    def build(self):

        self.model = Model(residual_body, self.gym_env.action_space)

        # Optimizers
        optimizer_q = optax.chain(optax.apply_every(k=4), optax.adam(0.0001))
        optimizer_pi = optax.chain(optax.apply_every(k=4), optax.adam(0.0001))

        # q = coax.Q(func_q, gym_env)
        self.v = coax.V(self.model.v, self.gym_env)
        self.pi = coax.Policy(self.model.pi_logits, self.gym_env)

        self.pi_behavior = self.pi.copy()
        self.v_targ = self.v.copy()

        # One tracer for each agent
        self.tracers = {agent: coax.reward_tracing.NStep(n=9, gamma=0.9) for agent in self.env.possible_agents}

        # We just need one buffer ...?
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=2048)

        # Regularizer
        self.pi_regularizer = coax.regularizers.EntropyRegularizer(self.pi, 0.001)

        # Updaters
        self.ppo_clip = coax.policy_objectives.PPOClip(self.pi, optimizer=optimizer_pi, regularizer=self.pi_regularizer)
        self.simple_td = coax.td_learning.SimpleTD(self.v, self.v_targ, optimizer=optimizer_q, loss_function=mse)

    epoch_num = 0
    game_num = 0

    new_epoch = True
    checkpoint_period = 15

    verbose = False

    def learn(self):
        for i in range(10000000):

            self.game_num += 1

            # Episode start
            obs = self.env.reset(i)
            self.gym_env.reset()  # This should print logged metrics

            cum_reward = {agent: 0. for agent in self.env.possible_agents}
            if self.verbose:
                print(self.env.render())

            while len(self.env.agents) > 0:

                # Get actions
                actions = {}
                logps = {}
                for agent in self.env.agents:
                    actions[agent], logps[agent] = self.pi_behavior(obs[agent], return_logp=True)

                live_agents = self.env.agents[:]
                obs_next, rewards, terminations, truncations, _ = self.env.step(actions)

                # Record metrics
                self.gym_env.step(actions[live_agents[0]])

                if self.verbose:
                    print(self.env.render())
                    sleep(0.75)

                # Trace rewards
                for agent in live_agents:
                    self.tracers[agent].add(
                        obs[agent],
                        actions[agent],
                        rewards[agent],
                        terminations[agent] or truncations[agent],
                        logps[agent]
                    )
                    cum_reward[agent] += rewards[agent]

                # Update
                for _, tracer in self.tracers.items():
                    while tracer:
                        self.buffer.add(tracer.pop())

                if len(self.buffer) >= self.buffer.capacity:
                    for _ in range(4 * self.buffer.capacity // 32):  # 4 rounds
                        transition_batch = self.buffer.sample(batch_size=32)
                        metrics_v, td_error = self.simple_td.update(transition_batch, return_td_error=True)
                        metrics_pi = self.ppo_clip.update(transition_batch, td_error)

                    self.buffer.clear()
                    self.pi_behavior.soft_update(self.pi, tau=0.1)
                    self.v_targ.soft_update(self.v, tau=0.1)

                    self.new_epoch = True
                    self.epoch_num += 1

                obs = obs_next
            if self.verbose:
                print("===== End Game {}. Epoch {} =========================".format(i, self.epoch_num))

            if self.new_epoch:
                self.new_epoch = False
                self.output_mode_game()

    def output_mode_game(self):

        # Flush the replay stack so that we don't get a stinky replay
        self.env.state_pq = []

        print("===== Game {}. Epoch {} =========================".format(self.game_num, self.epoch_num))
        obs = self.env.reset(random.randint(-1000000, 1000000))
        print(self.env.render())
        cum_reward = {agent: 0. for agent in self.env.possible_agents}

        while len(self.env.agents) > 0:
            # Get actions
            actions = {}
            for agent in self.env.agents:
                actions[agent] = self.pi_behavior.mode(obs[agent])

            live_agents = self.env.agents[:]
            obs_next, rewards, terminations, truncations, _ = self.env.step(actions)
            for agent in live_agents:
                cum_reward[agent] += rewards[agent]

            print(self.env.render())
            obs = obs_next

        print("===== End Game {}. Epoch {} =========================".format(self.game_num, self.epoch_num))
        print("Rewards: {}".format({s: "{:.2f}".format(r) for s, r in cum_reward.items()}))
        print("-----------------------------------------------------")

        if self.epoch_num % self.checkpoint_period == 0 and self.epoch_num != 0:
            print("======Checkpoint {}============================".format(self.epoch_num))
            print("-----------------------------------------------")
            now = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
            coax.utils.dump([self.pi, self.pi_behavior, self.v, self.v_targ, self.tracers,
                             self.buffer, self.pi_regularizer, self.simple_td,
                             self.ppo_clip, self.model],

                            ".checkpoint/{}_epoch_{}_{}.pkl.lz4".format(self.name, self.epoch_num, now))

    def build_from_file(self, path):

        (self.pi, self.pi_behavior, self.v, self.v_targ, self.tracers,
         self.buffer, self.pi_regularizer, self.simple_td,
         self.ppo_clip, self.model) = coax.utils.load(path)


m = PPOModel()
if False:
    m.build()
else:
    m.build_from_file(".checkpoint/standard_4p_ppo_epoch_15_2023-03-19_15:09:18.pkl.lz4")

m.learn()
