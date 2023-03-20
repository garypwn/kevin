import cProfile
import os
import pstats
import random
from datetime import datetime
from time import sleep

import coax
import optax
from coax.value_losses import mse

from kevin.src.engine.board_updater import FixedBoardUpdater
from kevin.src.engine.python_engine import RotatingBoardUpdater, PythonGameState
from kevin.src.environment.rewinding_environment import RewindingEnv
from kevin.src.model.model import Model, residual_body

# set some env vars
os.environ.setdefault('JAX_PLATFORMS', 'gpu, cpu')  # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


class PPOModel:
    name = 'standard_4p_ppo'

    updater = FixedBoardUpdater(11, 11, 4)
    game = PythonGameState(updater=updater)
    env = RewindingEnv(game)
    env.fancy_render = True
    gym_env = env.dummy_gym_environment

    ppo_clip: coax.policy_objectives.PPOClip | None
    simple_td: coax.td_learning.SimpleTD | None

    smooth_update_rate = 0.9

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
        optimizer_q = optax.chain(optax.apply_every(k=4), optax.adam(0.75))
        optimizer_pi = optax.chain(optax.apply_every(k=4), optax.adam(0.75))

        # q = coax.Q(func_q, gym_env)
        self.v = coax.V(self.model.v, self.gym_env)
        self.pi = coax.Policy(self.model.pi_logits, self.gym_env)

        self.pi_behavior = self.pi.copy()
        self.v_targ = self.v.copy()

        # One tracer for each agent
        self.tracers = {agent: coax.reward_tracing.NStep(n=9, gamma=0.9) for agent in self.env.possible_agents}

        # We just need one buffer ...?
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=4096)

        # Updaters
        self.ppo_clip = coax.policy_objectives.PPOClip(self.pi, optimizer=optimizer_pi)
        self.simple_td = coax.td_learning.SimpleTD(self.v, self.v_targ, optimizer=optimizer_q, loss_function=mse)

    epoch_num = 0
    game_num = 0

    new_epoch = True
    checkpoint_period = 15
    render_period = 25

    verbose = False

    def learn(self, loops):
        for i in range(loops):

            # Anneal learning rate
            if self.epoch_num == 4:
                self.change_learning_rate(0.1)
                self.smooth_update_rate = 0.7

            if self.epoch_num == 15:
                self.change_learning_rate(0.01)
                self.smooth_update_rate = 0.35

            if self.epoch_num == 30:
                self.change_learning_rate(0.001)
                self.smooth_update_rate = 0.1

            if self.epoch_num == 50:
                self.change_learning_rate(0.0005)

            if self.epoch_num == 100:
                self.change_learning_rate(0.0001)

            self.game_num += 1

            # Episode start
            obs = self.env.reset(i)
            self.gym_env.reset()  # This should print logged metrics

            cum_reward = {agent: 0. for agent in self.env.possible_agents}

            if self.render_period > 0 and i % self.render_period == -1 % self.render_period:
                self.verbose = True

            if self.verbose:
                print("===== Game {}. Epoch {} =========================".format(self.game_num, self.epoch_num))
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
                    self.pi_behavior.soft_update(self.pi, tau=self.smooth_update_rate)
                    self.v_targ.soft_update(self.v, tau=self.smooth_update_rate)

                    self.new_epoch = True
                    self.epoch_num += 1

                obs = obs_next
            if self.verbose:
                print("===== End Game {}. Epoch {} =========================".format(i, self.epoch_num))
                self.verbose = False

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
                action = self.pi_behavior.mode(obs[agent])
                if self.env.game.meta_factory.all_safe_moves[agent][action] == 0:
                    for i in range(4):
                        action = self.pi_behavior(obs[agent])
                        if self.env.game.meta_factory.all_safe_moves[agent][action] == 8:
                            break
                        if i == 3:
                            print("Warning: Agent {} has tried to make an unsafe move 4 times. Hopefully it knows "
                                  "something we don't.".format(agent))

                actions[agent] = action

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

    def profile(self):
        profiler = cProfile.Profile()

        # Play a few games to warm up
        self.learn(10)

        print("Turning on profiler...")
        profiler.enable()
        self.learn(50)
        profiler.disable()
        now = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        stats = pstats.Stats(profiler)
        stats.dump_stats("./.profiler/{}.prof".format(now))
        print("200 games complete. Turning off profiler.")

    def change_learning_rate(self, rate_pi, rate_v=None):
        if rate_v is None:
            rate_v = rate_pi

        self.ppo_clip.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(rate_pi))
        self.simple_td.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(rate_v))


m = PPOModel()
if True:
    m.build()
else:
    m.build_from_file(".checkpoint/standard_4p_ppo_epoch_15_2023-03-19_15:09:18.pkl.lz4")

m.profile()
m.learn(10000000)
