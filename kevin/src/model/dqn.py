import coax
import gymnasium
import optax
from coax.value_losses import mse
from gymnasium import spaces

from kevin.src.model.func_approximator import FuncApproximator
from kevin.src.model.model import Model


class DeepQ(Model):
    _q_optimizer = None
    _learning_rate = None
    q = None
    pi = None
    q_targ = None
    q_learning = None
    name = None
    soft_update_rate = 0.1
    _transitions_this_gen = 0

    logp_required = True

    def __init__(self, gym_env: gymnasium.Env, func_approximator: FuncApproximator, tensorboard=None):

        super().__init__(gym_env, func_approximator, tensorboard)
        self.buffer = coax.experience_replay.PrioritizedReplayBuffer(capacity=500000, alpha=0.6)

    def build(self, name_prefix=""):
        self.q = coax.Q(self.func_approximator.q, self.gym_env)
        self.pi = coax.BoltzmannPolicy(self.q, temperature=0.2)

        self.q_targ = self.q.copy()

        # Updaters
        self.q_learning = coax.td_learning.QLearning(self.q, q_targ=self.q_targ)

        self.set_name(name_prefix, "dqn")

    def build_from_file(self, filename: str):
        (self.name, self.transitions_processed, self.generation, self.pi,
         self.q, self.q_targ, self.q_learning, self.func_approximator) = coax.utils.load(filename)

    def checkpoint(self, directory_path: str) -> str:
        filename = f"{directory_path}/{self.name}/{self.name}_gen_{self.generation}.pkl.lz4"
        coax.utils.dump(
            [self.name, self.transitions_processed, self.generation, self.pi,
             self.q, self.q_targ, self.q_learning, self.func_approximator], filename)

        return filename

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate: float):
        if rate == self._learning_rate:
            return

        self._learning_rate = rate
        self._q_optimizer = optax.adam(rate)
        self.q_learning.optimizer = self._q_optimizer

    def learn(self, batch_size: int):

        # Anneal learning rate and other hypers
        # Remember 1 gen is anywhere between 1000 and 4000 games
        g = self.generation
        if g < 30:  # 60k games
            self.learning_rate = 3e-5
            self.soft_update_rate = 0.001
            self.buffer.beta = 1e-4

        if 30 <= g < 75:  # 150k games
            self.learning_rate = 3e-5
            self.soft_update_rate = 0.001
            self.buffer.beta = 1e-3

        if 75 <= g < 250:  # 500k games
            self.learning_rate = 3e-5
            self.soft_update_rate = 0.001
            self.buffer.beta = 1e-2

        if 250 <= g < 500:  # 1M games
            self.learning_rate = 3e-5
            self.soft_update_rate = 0.001
            self.buffer.beta = 0.4

        if g >= 500:  # 5M games
            self.learning_rate = 3e-5
            self.soft_update_rate = 0.001
            self.buffer.beta = 0.4
        del g

        mini_batch_size = 128

        if len(self.buffer) >= 5000:
            for _ in range(batch_size // mini_batch_size):
                transition_batch = self.buffer.sample(batch_size=mini_batch_size)
                metrics_v, td_error = self.q_learning.update(transition_batch, return_td_error=True)
                self.record_metrics(metrics_v, self.transitions_processed)
                self.buffer.update(transition_batch.idx, td_error)

            self._transitions_this_gen += batch_size
            self.transitions_processed += batch_size
            self.q_targ.soft_update(self.q, tau=self.soft_update_rate)

        if self._transitions_this_gen >= self.transitions_per_gen:
            self.generation += 1
            self._transitions_this_gen = 0

    def add_transitions(self, batches: list[coax.reward_tracing.TransitionBatch]):
        ct = 0
        for batch in batches:
            for chunk in coax.utils.chunks_pow2(batch):
                ct += chunk.batch_size
                td_error = self.q_learning.td_error(chunk)
                self.buffer.add(chunk, td_error)

        return ct

    @property
    def buffer_len(self):
        return len(self.buffer)

    @property
    def buffer_capacity(self):
        return self.buffer.capacity

    def policy(self) -> coax.BoltzmannPolicy:
        return self.pi
