import os
import random
import time
from math import exp
from typing import Callable

import coax
import jax.nn
import ray
import tensorboardX

from kevin.src.engine import utils
from kevin.src.engine.board_updater import FixedBoardUpdater
from kevin.src.engine.python_engine import PythonGameState
from kevin.src.environment.rewinding_environment import RewindingEnv
from kevin.src.environment.wrapper import FrameStacking
from kevin.src.model import func_approximator
from kevin.src.model.func_approximator import FuncApproximator
from kevin.src.model.model import Model
from kevin.src.model.ppo import PPO

RENDER_PERIOD = 10.  # How often to render a game in seconds
CHECKPOINT_PERIOD = 3  # How often to checkpoint in model generations


def make_environment(game):
    return FrameStacking(RewindingEnv(game))


class ParallelLearning:
    # set some env vars
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # tell XLA to be quiet
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'  # use most of gpu mem

    model: Model

    def __init__(self, model: Callable):
        self.version = '0.1'
        self.name = f'kevin_v{self.version}'
        self.tensorboard = tensorboardX.SummaryWriter(logdir="runs/v{}/{}".format(self.version, self.name))

        self.updater = FixedBoardUpdater(11, 11)
        self.game = PythonGameState(updater=self.updater)
        self.env = make_environment(self.game)
        self.gym_env = self.env.dummy_gym_environment

        ray.init()
        self.updater_ref = ray.put(self.updater)

        self.func_approximator = FuncApproximator(func_approximator.residual_body, self.gym_env.action_space)

        self.model = model(self.gym_env, self.func_approximator, self.tensorboard)

    def build(self):
        self.model.build()

    def build_from_file(self, filename):
        self.model.build_from_file(filename)

    def checkpoint(self):
        print("======Checkpoint {}============================".format(self.model.generation))
        filename = self.model.checkpoint(".checkpoint")
        print("Saved as {}".format(filename))
        print("================================================")

    def learn(self, loops):

        num_workers = 12
        games_per_worker = 50

        # Set up the tensorboard

        for i in range(loops):

            futures = []
            renderer = None

            batch_size = 256  # Number of transitions to learn on before rerunning the worker loop
            while True:

                episode_num = self.model.transitions_processed // 100  # Approximate game number

                # These get sent to our workers so that they can display stats properly
                stats = {
                    "episode_num": episode_num,
                    "generation_num": self.model.generation
                }

                # Check on our workers
                ready = []
                if len(futures) > 0:
                    ready, futures = ray.wait(futures, timeout=0.001)

                # If render worker is done, we must add another
                if renderer not in futures:
                    renderer = None

                # Set up new workers to play some games then die
                pi_lz4 = coax.utils.dumps(self.model.policy())
                while len(futures) < num_workers:
                    t = RENDER_PERIOD if renderer is None else -1
                    v_func = None
                    w = play_games.remote(pi_lz4, games_per_worker, self.updater_ref, t, stats, v_func)
                    futures.append(w)
                    if renderer is None:
                        renderer = w

                # Process results of completed workers
                for future in ready:
                    result = ray.get(future)
                    batches = result['transitions']
                    print("Received {} transition batches from worker.".format(len(batches)))
                    self.log_scoreboard(result['episode_num'], result['gen'], result['score'])
                    trans_added = self.model.add_transitions(batches)
                    print("Processed {} transitions to buffer. "
                          "Buffer {} / {}.".format(trans_added, self.model.buffer_len, self.model.buffer_capacity))

                # Learn
                new_gen = self.model.learn(batch_size)
                if new_gen:
                    g = self.model.generation
                    print("Model updated. "
                          "New workers will be dispatched with generation {}.".format(g))

                    if g % CHECKPOINT_PERIOD == 0 and g != 0:
                        self.checkpoint()

    def log_scoreboard(self, eps, gen, score):
        self.tensorboard.add_scalar("generation", gen, global_step=eps)

        games = score['games']

        rates_dict = {name: (v / games) if games != 0 else 0 for name, v in score.items() if name != "games"}

        # rates_dict = {name: [x / y for x, y in zip(results, games)]
        #              for name, results in score.items() if name != 'games'}

        for name, rate in rates_dict.items():
            self.tensorboard.add_scalar("ScoreVsRandom/{}".format(name), rate, global_step=eps)


@ray.remote
def play_games(pi, num_games, updater_ref, render_period=-1, stats=None, v=None):
    os.environ['JAX_PLATFORMS'] = 'gpu, cpu'  # tell JAX to use GPU
    return ExperienceWorker(pi, updater_ref, render_period, stats, v=v)(num_games)


class ExperienceWorker:
    verbose = False
    ultra_verbose = False
    generation_num = 0
    episode_num = 0

    policy: Callable
    random_policy: Callable

    def __init__(self, pi, updater_ref, render_period=-1, stats: dict | None = None, v=None):
        """
        Creates a new worker that plays games to gather experience.
        @param pi: The policy function to choose actions in lz4 pickle byte string format (pi_behavior)
        @param render_period: How often to render a game. Usually -1 for all but 1 thread.
        """
        updater = updater_ref
        self.episode_num = stats["episode_num"] if stats is not None else 0
        self.generation_num = stats["generation_num"] if stats is not None else 0

        self.env = make_environment(PythonGameState(updater=updater))
        self.policy = coax.utils.loads(pi)
        self.value = None if v is None else coax.utils.loads(v)
        self.render_period = render_period

        # One tracer for each agent
        self.tracers = {agent: coax.reward_tracing.NStep(n=10, gamma=0.95) for agent in self.env.possible_agents}

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
        vs_random = {'win': 0,
                     'loss': 0,
                     'draw': 0,
                     'games': 0}

        for i in range(n):
            # Episode start
            obs = self.env.reset(i + seed_start)

            policies = {}
            policy_names = {}

            # 3 games out of every 30 have random agents
            random_agents = i % 30 if i % 30 in (1, 2, 3) else -1
            random_agent_count = 0
            for j, agent in enumerate(self.env.agents):
                if j == random_agents:
                    policies[agent] = self.random_policy
                    policy_names[agent] = "Safe Random"
                    random_agent_count += 1
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
                    if self.ultra_verbose and policies[agent] == self.policy:
                        print(policies.get(agent).dist_params(obs[agent]))

                live_agents = self.env.agents[:]
                obs_next, rewards, terminations, truncations, _ = self.env.step(actions)

                if self.verbose:
                    print(self.env.render())
                    print("Rewards:\t" + "\t".join([f"{utils.render_symbols[a]['head']}: {v:>.2f}"
                                                    for a, v in rewards.items()]))
                    print("Prob:\t\t" + "\t".join([f"{utils.render_symbols[a]['head']}: {exp(v):>.2f}"
                                                   for a, v in logps.items()]))
                    if self.value is not None:
                        vals = {agent: self.value(obs) for agent, obs in obs_next.items()}
                        print("VFunc:\t\t" + "\t".join([f"{utils.render_symbols[a]['head']}: {v:>.2f}"
                                                        for a, v in vals.items()]))

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
            for agent, tracer in self.tracers.items():
                transition_batches.append(tracer.flush())

            # Record how we did vs the random agent
            winner = self.env.game.winner()
            if random_agent_count > 0:
                if winner is None:
                    vs_random['draw'] += 1
                elif policy_names[winner] == 'Safe Random':
                    vs_random['loss'] += 1
                else:
                    vs_random['win'] += 1
                vs_random['games'] += 1

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

        # Pick a uniform random move out of the safe ones
        move = random.choice(possible_moves)

        if return_logp:
            # pi_behavior is a categorical distribution, so we can get the prob of each move from logits
            logits = self.pi.dist_params(obs)
            logps = jax.nn.log_softmax(logits['logits'])
            logp = logps[move]

            # Finally, we clip logps that are too big to prevent grads being computed to inf or nan
            if logp < -10.:
                logp = -10.

            return move, logp
        else:
            return move


m = ParallelLearning(PPO)
if True:
    m.build()
else:
    m.build_from_file(".checkpoint/kevin_v0.1_2023-03-24_1305/kevin_v0.1_2023-03-24_1305_gen_12.pkl.lz4")

m.learn(10000000)
