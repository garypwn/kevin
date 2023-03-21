import cProfile
import os
import pstats
import random
from datetime import datetime

import coax
import optax
import tensorboardX
from coax.value_losses import mse

from kevin.src.engine import utils
from kevin.src.engine.board_updater import FixedBoardUpdater
from kevin.src.engine.python_engine import PythonGameState
from kevin.src.environment.rewinding_environment import RewindingEnv
from kevin.src.model.model import Model, residual_body

# set some env vars
os.environ.setdefault('JAX_PLATFORMS', 'gpu, cpu')  # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.6'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet


class PPOModel:
    name = 'standard_4p_ppo'

    # Set up the tensorboard
    tensorboard = tensorboardX.SummaryWriter(comment=name)

    def record_metrics(self, metrics):
        for name, metric in metrics.items():
            self.tensorboard.add_scalar(str(name), float(metric), global_step=self.episode_number)

    updater = FixedBoardUpdater(11, 11, 4)
    game = PythonGameState(updater=updater)
    env = RewindingEnv(game)
    env.fancy_render = True
    gym_env = env.dummy_gym_environment

    ppo_clip: coax.policy_objectives.PPOClip | None
    simple_td: coax.td_learning.SimpleTD | None

    def __init__(self):
        self.smooth_update_rate = 0.1
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

        self.smooth_update_rate = 0.75
        self.model = Model(residual_body, self.gym_env.action_space)

        # Optimizers
        optimizer_q = optax.chain(optax.apply_every(k=4), optax.adam(0.01))
        optimizer_pi = optax.chain(optax.apply_every(k=4), optax.adam(0.01))

        # q = coax.Q(func_q, gym_env)
        self.v = coax.V(self.model.v, self.gym_env)
        self.pi = coax.Policy(self.model.pi_logits, self.gym_env)

        self.pi_regularizer = coax.regularizers.EntropyRegularizer(self.pi, beta=0.001)

        self.pi_behavior = self.pi.copy()
        self.v_targ = self.v.copy()

        # One tracer for each agent
        self.tracers = {agent: coax.reward_tracing.NStep(n=10, gamma=0.9) for agent in self.env.possible_agents}

        # We just need one buffer ...?
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=4096)

        # Updaters
        self.ppo_clip = coax.policy_objectives.PPOClip(self.pi, optimizer=optimizer_pi, regularizer=self.pi_regularizer)
        self.simple_td = coax.td_learning.SimpleTD(self.v, self.v_targ, optimizer=optimizer_q, loss_function=mse)

    Generation_num = 0
    episode_number = 0

    new_Generation = True
    checkpoint_period = 15
    render_period = 13

    verbose = False

    def smart_random_policy(self, obs, return_logp=False):

        # We get the cheat code safe spots
        meta = obs[1][0]
        moves = [meta[7], meta[11], meta[15], meta[19]]
        safe_moves = []
        for i, move in enumerate(moves):
            if move == 8:
                safe_moves.append(i)

        if len(safe_moves) < 1:
            # No safe move - choose a random one
            possible_moves = [i for i in range(self.gym_env.action_space.n)]
        else:
            # Safe move exists, choose one from there
            possible_moves = safe_moves

        # pi_behavior's probability distribution
        logits = coax.utils.single_to_batch(self.pi_behavior.dist_params(obs))
        dist = self.pi_behavior.proba_dist
        rng = self.pi_behavior.rng  # Borrow this to save time

        # We take a sample from pi_behavior and modify it so that it happened to pick our move
        sample = dist.sample(logits, rng)
        move = random.choice(possible_moves)
        sample = sample.at[0, 0].set(move)

        # Logp is the log of the probability that pi would have given us this sample
        logp = dist.log_proba(logits, sample)
        logp = coax.utils.batch_to_single(logp)

        # Finally, we clip logps that are too big to prevent the optimizer from going crazy.
        if logp < -10:
            logp = -10  # This is still equivalent to 0.005% chance of happening

        if return_logp:
            return move, logp
        else:
            return move

    def learn(self, loops):

        for i in range(loops):

            # Anneal learning rate and other hypers
            if self.Generation_num == 5:
                self.change_learning_rate(0.005)
                self.smooth_update_rate = 0.2

            if self.Generation_num == 15:
                self.change_learning_rate(0.001)
                self.smooth_update_rate = 0.15

            if self.Generation_num == 30:
                self.change_learning_rate(0.0005)
                self.smooth_update_rate = 0.1

            if self.Generation_num == 50:
                self.change_learning_rate(0.0001)

            self.episode_number += 1

            # Episode start
            obs = self.env.reset(i)

            # Possibly set some agents to use different policies
            # For now we have 2 games with no random agent, then 3 games with
            match i % 7:
                case _:
                    random_agent_count = i % 4

            policies = {}
            policy_names = {}
            for j, agent in enumerate(self.env.agents):
                if j < random_agent_count:
                    policies[agent] = self.smart_random_policy
                    policy_names[agent] = "Safe Random"
                else:
                    policies[agent] = self.pi_behavior
                    policy_names[agent] = "Pi Behavior"

            cum_reward = {agent: 0. for agent in self.env.possible_agents}

            if self.render_period > 0 and i % self.render_period == 0:
                self.verbose = True

            if self.verbose:
                print("===== Game {}. Generation {} =========================".format(self.episode_number,
                                                                                      self.Generation_num))
                print(self.env.render())

            live_agents = self.env.agents[:]
            while len(self.env.agents) > 0:

                # Get actions
                actions = {}
                logps = {}
                for agent in self.env.agents:
                    actions[agent], logps[agent] = policies[agent](obs[agent], return_logp=True)

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
                    for _ in range(6 * self.buffer.capacity // 32):  # 6 passes
                        transition_batch = self.buffer.sample(batch_size=32)
                        metrics_v, td_error = self.simple_td.update(transition_batch, return_td_error=True)
                        metrics_pi = self.ppo_clip.update(transition_batch, td_error)
                        self.record_metrics(metrics_pi)
                        self.record_metrics(metrics_v)

                    self.buffer.clear()
                    self.pi_behavior.soft_update(self.pi, tau=self.smooth_update_rate)
                    self.v_targ.soft_update(self.v, tau=self.smooth_update_rate)

                    self.new_Generation = True
                    self.Generation_num += 1

                obs = obs_next
            if self.verbose:
                print("===== End Game {}. Generation {} =========================".format(i, self.Generation_num))
                winner = self.env.game.winner()
                if winner is not None:
                    print("Result: {} {} win!".format(utils.render_symbols[winner]["head"], winner))
                else:
                    print("Result: Draw.")

                for agent, policy in policy_names.items():
                    print("{} {}:\t{}\t\tReward:\t{:.2f}".format(utils.render_symbols[agent]["head"],
                                                                 agent, policy, cum_reward[agent]))
                print("=====================================================\n")
                self.verbose = False

            if self.new_Generation:
                self.new_Generation = False
                # self.output_mode_game()

    def output_mode_game(self):

        # Flush the replay stack so that we don't get a stinky replay
        self.env.state_pq = []

        print("===== Game {}. Generation {} =========================".format(self.episode_number, self.Generation_num))
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

        print("===== End Game {}. Generation {} =========================".format(self.episode_number, self.Generation_num))
        print("Rewards: {}".format({s: "{:.2f}".format(r) for s, r in cum_reward.items()}))
        print("-----------------------------------------------------")

        if self.Generation_num % self.checkpoint_period == 0 and self.Generation_num != 0:
            print("======Checkpoint {}============================".format(self.Generation_num))
            now = datetime.today().strftime('%Y-%m-%d_%H%M')
            filename = ".checkpoint/{}_gen_{}_{}.pkl.lz4".format(self.name, self.Generation_num, now)
            coax.utils.dump([self.pi, self.pi_behavior, self.v, self.v_targ, self.tracers,
                             self.buffer, self.pi_regularizer, self.simple_td,
                             self.ppo_clip, self.model], filename)
            print("Saved as {}".format(filename))
            print("================================================")

    def build_from_file(self, path):

        (self.pi, self.pi_behavior, self.v, self.v_targ, self.tracers,
         self.buffer, self.pi_regularizer, self.simple_td,
         self.ppo_clip, self.model) = coax.utils.load(path)

    def profile(self):
        profiler = cProfile.Profile()

        # Play a few games to warm up
        self.learn(20)

        print("Turning on profiler...")
        profiler.enable()
        self.learn(80)
        profiler.disable()
        now = datetime.today().strftime('%Y-%m-%d_%H%M')
        stats = pstats.Stats(profiler)
        stats.dump_stats("./.profiler/{}.prof".format(now))
        print("80 games complete. Turning off profiler.")

    def change_learning_rate(self, rate_pi, rate_v=None):
        if rate_v is None:
            rate_v = rate_pi

        self.ppo_clip.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(rate_pi))
        self.simple_td.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(rate_v))


m = PPOModel()
if True:
    m.build()
else:
    m.build_from_file("")

m.profile()
m.learn(10000000)
