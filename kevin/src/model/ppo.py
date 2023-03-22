import cProfile
import os
import pstats
import random
import time
from datetime import datetime
from typing import Callable

import coax
import optax
import ray
import tensorboardX
from coax.value_losses import mse

from kevin.src.engine import utils
from kevin.src.engine.board_updater import FixedBoardUpdater
from kevin.src.engine.python_engine import PythonGameState
from kevin.src.environment.rewinding_environment import RewindingEnv
from kevin.src.model.model import Model, residual_body


class PPOModel:
    # set some env vars
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # use most of gpu mem

    version = '0.1'
    name = 'kevin_v{}_{}'.format(version, datetime.today().strftime('%Y-%m-%d_%H%M'))

    # Set up the tensorboard
    tensorboard = tensorboardX.SummaryWriter(logdir="runs/v{}/{}".format(version, name))

    def record_metrics(self, metrics):
        for name, metric in metrics.items():
            self.tensorboard.add_scalar(str(name), float(metric), global_step=self.episode_number)

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

        self.updater = FixedBoardUpdater(11, 11)
        self.game = PythonGameState(updater=self.updater)
        self.env = RewindingEnv(self.game)
        self.gym_env = self.env.dummy_gym_environment

        self.buffer = coax.experience_replay.PrioritizedReplayBuffer(capacity=500000)

        ray.init()
        self.updater_ref = ray.put(self.updater)

    _learning_rate = 0.0001

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate: float):
        if rate == self._learning_rate:
            return

        self._learning_rate = rate
        self.ppo_clip.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(rate))
        self.simple_td.optimizer = optax.chain(optax.apply_every(k=4), optax.adam(rate))

    def build(self):

        self.smooth_update_rate = 0.5
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

        # Updaters
        self.ppo_clip = coax.policy_objectives.PPOClip(self.pi, optimizer=optimizer_pi, regularizer=self.pi_regularizer)
        self.simple_td = coax.td_learning.SimpleTD(self.v, self.v_targ, optimizer=optimizer_q, loss_function=mse)

    generation_num = 0
    episode_number = 0

    checkpoint_period = 1
    render_period = 10.  # How often to render a game in seconds

    verbose = False

    def learn(self, loops):

        num_workers = 10
        games_per_worker = 50

        for i in range(loops):

            futures = []
            renderer = None
            batches_this_gen = 0

            # 128 batches x 256 transitions / 64-128 transitions per game. That's around 1-2k games per gen.
            batches_per_gen = 128
            batch_size = 256
            while True:

                # Anneal learning rate and other hypers
                # Remember 1 gen is anywhere between 1000 and 4000 games
                g = self.generation_num
                if g < 30:  # 60k games
                    self.learning_rate = 0.025
                    self.smooth_update_rate = 0.03

                if 30 <= g < 75:  # 150k games
                    self.learning_rate = 0.0075
                    self.smooth_update_rate = 0.2

                if 75 <= g < 250:  # 500k games
                    self.learning_rate = 0.002
                    self.smooth_update_rate = 0.1

                if 250 <= g < 500:  # 1M games
                    self.learning_rate = 0.00075

                if self.generation_num >= 500:  # 5M games
                    self.learning_rate = 0.0001
                del g

                # These get sent to our workers so that they can display stats properly
                stats = {
                    "episode_num": self.episode_number,
                    "generation_num": self.generation_num
                }

                # Check on our workers
                ready = []
                if len(futures) > 0:
                    ready, futures = ray.wait(futures, timeout=0.1)

                # If render worker is done, we must add another
                if renderer not in futures:
                    renderer = None

                # Set up new workers to play 100 games then die
                pi_lz4 = coax.utils.dumps(self.pi_behavior)
                while len(futures) < num_workers:
                    t = self.render_period if renderer is None else -1
                    w = play_games.remote(pi_lz4, games_per_worker, self.updater_ref, t, stats)
                    futures.append(w)
                    if renderer is None:
                        renderer = w

                # Process results of completed workers
                for future in ready:
                    result = ray.get(future)
                    batches = result['transitions']
                    print("Received {} transition batches from worker.".format(len(batches)))
                    self.log_scoreboard(result['episode_num'], result['gen'], result['score'])
                    trans_added = self.add_transition_batches(batches)
                    print("Processed {} transitions to buffer. "
                          "Buffer {} / {}.".format(trans_added, len(self.buffer), self.buffer.capacity))
                    self.episode_number += games_per_worker

                # Learn
                if len(self.buffer) >= 15000:
                    transition_batch = self.buffer.sample(batch_size=batch_size)
                    metrics_v, td_error = self.simple_td.update(transition_batch, return_td_error=True)
                    metrics_pi = self.ppo_clip.update(transition_batch, td_error)
                    self.record_metrics(metrics_pi)
                    self.record_metrics(metrics_v)

                    self.buffer.update(transition_batch.idx, td_error)
                    batches_this_gen += 1

                if batches_this_gen >= batches_per_gen:
                    self.pi_behavior.soft_update(self.pi, tau=self.smooth_update_rate)
                    self.v_targ.soft_update(self.v, tau=self.smooth_update_rate)

                    self.generation_num += 1
                    batches_this_gen = 0

                    print("Model updated. "
                          "New workers will be dispatched with generation {}.".format(self.generation_num))

                    if self.generation_num % self.checkpoint_period == 0 and self.generation_num != 0:
                        self.checkpoint()

    def add_transition_batches(self, transition_batches):
        ct = 0
        for batch in transition_batches:
            for chunk in coax.utils.chunks_pow2(batch):
                ct += chunk.batch_size
                td_error = self.simple_td.td_error(chunk)
                self.buffer.add(chunk, td_error)

        return ct

    def checkpoint(self):
        print("======Checkpoint {}============================".format(self.generation_num))
        now = datetime.today().strftime('%Y-%m-%d_%H%M')
        filename = ".checkpoint/{}_gen_{}_{}.pkl.lz4".format(self.name, self.generation_num, now)

        coax.utils.dump(
            [self.name, self.episode_number, self.generation_num, self.pi, self.pi_behavior, self.v, self.v_targ,
             self.pi_regularizer, self.simple_td,
             self.ppo_clip, self.model], filename)
        print("Saved as {}".format(filename))
        print("================================================")

    def build_from_file(self, path):

        (self.name, self.episode_number, self.generation_num, self.pi, self.pi_behavior, self.v, self.v_targ,
         self.pi_regularizer, self.simple_td,
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

    def log_scoreboard(self, eps, gen, score):
        self.tensorboard.add_scalar("generation", gen, global_step=eps)

        games = score['games']
        rates_dict = {name: [x / y for x, y in zip(results, games)]
                      for name, results in score.items() if name != 'games'}

        for name, rates in rates_dict.items():
            for i, rate in enumerate(rates):
                self.tensorboard.add_scalar("ScoreVsRandom/{}/{}_opponents".format(name, i), rate, global_step=eps)


@ray.remote
def play_games(pi, num_games, updater_ref, render_period=-1, stats=None):
    os.environ['JAX_PLATFORMS'] = 'gpu, cpu'  # tell JAX to use GPU
    return ExperienceWorker(pi, updater_ref, render_period, stats)(num_games)


class ExperienceWorker:
    verbose = False
    generation_num = 0
    episode_num = 0

    policy: Callable
    random_policy: Callable

    def __init__(self, pi, updater_ref, render_period=-1, stats: dict | None = None):
        """
        Creates a new worker that plays games to gather experience.
        @param pi: The policy function to choose actions in lz4 pickle byte string format (pi_behavior)
        @param render_period: How often to render a game. Usually -1 for all but 1 thread.
        """
        updater = updater_ref
        self.episode_num = stats["episode_num"] if stats is not None else 0
        self.generation_num = stats["generation_num"] if stats is not None else 0

        self.env = RewindingEnv(PythonGameState(updater=updater))
        self.policy = coax.utils.loads(pi)
        self.render_period = render_period

        # One tracer for each agent
        self.tracers = {agent: coax.reward_tracing.NStep(n=2, gamma=0.8) for agent in self.env.possible_agents}

    @property
    def policy(self):
        return self._policy

    @policy.setter
    def policy(self, policy):
        self._policy = policy
        self.random_policy = SmartRandomPolicy(policy, self.env.dummy_gym_environment)

    def __call__(self, n):
        """
        Runs n games and returns a set of transition batches
        @param n: The number of games
        @return: A set containing transition batches.
        """

        seed_start = random.randint(-100000000, 100000000)
        transition_batches = []
        last_render = 0.

        # Record vs random opponents
        vs_random = {'win': [0] * len(self.env.possible_agents),
                     'loss': [0] * len(self.env.possible_agents),
                     'draw': [0] * len(self.env.possible_agents),
                     'games': [0] * len(self.env.possible_agents)}

        for i in range(n):
            # Episode start
            obs = self.env.reset(i + seed_start)

            policies = {}
            policy_names = {}

            # 3 games with random agents for every 25 without
            random_agents = i % 28 if i % 28 in (1, 2, 3) else 0
            for j, agent in enumerate(self.env.agents):
                if j < random_agents:
                    policies[agent] = self.random_policy
                    policy_names[agent] = "Safe Random"
                else:
                    policies[agent] = self.policy
                    policy_names[agent] = "Pi Behavior"

            cum_reward = {agent: 0. for agent in self.env.possible_agents}

            if self.render_period > 0 and time.time() > last_render + self.render_period:
                self.verbose = True

            if self.verbose:
                print("===== Begin Generation {} Game ~{} ========================".format(self.generation_num,
                                                                                           self.episode_num + i))
                print(self.env.render())

            while len(self.env.agents) > 0:

                # Get actions
                actions = {}
                logps = {}
                for agent in self.env.agents:
                    actions[agent], logps[agent] = policies.get(agent)(obs[agent], return_logp=True)

                live_agents = self.env.agents[:]
                obs_next, rewards, terminations, truncations, _ = self.env.step(actions)

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

                obs = obs_next

            # After a game, flush the tracer and add all transitions to the buffer
            for _, tracer in self.tracers.items():
                transition_batches.append(tracer.flush())

            # Record how we did vs the random agent
            winner = self.env.game.winner()
            if winner is None:
                vs_random['draw'][random_agents] += 1
            elif policy_names[winner] == 'Safe Random':
                vs_random['loss'][random_agents] += 1
            else:
                vs_random['win'][random_agents] += 1
            vs_random['games'][random_agents] += 1

            if self.verbose:
                print("===== End Generation {} Game ~{} ({} / {}) =============="
                      .format(self.generation_num, self.episode_num + i, i, n))

                if winner is not None:
                    print("Result:\t{} {} win!".format(utils.render_symbols[winner]["head"], winner))
                else:
                    print("Result: Draw.")

                for agent, policy in policy_names.items():
                    print("{} {}:\t{}\t\tReward:\t{:.2f}".format(utils.render_symbols[agent]["head"],
                                                                 agent, policy, cum_reward[agent]))
                print("========================================================\n")
                self.verbose = False
                last_render = time.time()

        return {
            "transitions": transition_batches,
            "gen": self.generation_num,
            "episode_num": self.episode_num + n,
            "score": vs_random
        }


class SmartRandomPolicy:

    def __init__(self, ref_policy, gym_env):
        self.pi = ref_policy
        self.gym_env = gym_env

    def __call__(self, obs, return_logp=False):

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
        logits = coax.utils.single_to_batch(self.pi.dist_params(obs))
        dist = self.pi.proba_dist
        rng = self.pi.rng  # Borrow this to save time

        # We take a sample from pi_behavior and modify it so that it happened to pick our move
        sample = dist.sample(logits, rng)
        move = random.choice(possible_moves)
        sample = sample.at[0, 0].set(move)

        # Logp is the log of the probability that pi would have given us this sample
        logp = dist.log_proba(logits, sample)
        logp = coax.utils.batch_to_single(logp)

        # Finally, we clip logps that are too big to prevent the optimizer from going crazy.
        if logp < -20:
            logp = -20  # This is still equivalent to 2e-9 chance of happening

        if return_logp:
            return move, logp
        else:
            return move


m = PPOModel()
if True:
    m.build()
else:
    m.build_from_file("")

m.learn(10000000)
