import coax
import gymnasium
import optax
from coax.value_losses import mse
from gymnasium import spaces

from kevin.src.model.func_approximator import FuncApproximator
from kevin.src.model.model import Model

V_TO_PI_LEARNING_RATIO = 2


class PPO(Model):
    _v_optimizer = None
    _pi_optimizer = None
    _learning_rate = None
    _beta = None
    v = None
    pi = None
    pi_regularizer = None
    pi_behavior = None
    v_targ = None
    ppo_clip = None
    simple_td = None
    name = None
    soft_update_rate = 0.1
    _transitions_this_gen = 0

    logp_required = True

    def __init__(self, gym_env: gymnasium.Env, func_approximator: FuncApproximator, tensorboard=None):

        super().__init__(gym_env, func_approximator, tensorboard)
        self.buffer = coax.experience_replay.SimpleReplayBuffer(capacity=500000)

    def build(self, prefix=""):
        self.v = coax.V(self.func_approximator.v, self.gym_env)
        self.pi = coax.Policy(self.func_approximator.pi_logits, self.gym_env)

        self.pi_regularizer = coax.regularizers.EntropyRegularizer(self.pi, beta=0.001)

        self.pi_behavior = self.pi.copy()
        self.v_targ = self.v.copy()

        # Updaters
        self.ppo_clip = coax.policy_objectives.PPOClip(self.pi, regularizer=self.pi_regularizer)
        self.simple_td = coax.td_learning.SimpleTD(self.v, self.v_targ, loss_function=mse)

        self.set_name(prefix, "ppo")

    def build_from_file(self, filename: str):
        (self.name, self.transitions_processed, self.generation, self.pi, self.pi_behavior,
         self.v, self.v_targ, self.pi_regularizer, self.simple_td,
         self.ppo_clip, self.func_approximator) = coax.utils.load(filename)

    def checkpoint(self, directory_path: str) -> str:
        filename = f"{directory_path}/{self.name}/{self.name}_gen_{self.generation}.pkl.lz4"
        coax.utils.dump(
            [self.name, self.transitions_processed, self.generation, self.pi,
             self.pi_behavior, self.v, self.v_targ, self.pi_regularizer, self.simple_td,
             self.ppo_clip, self.func_approximator], filename)

        return filename

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta == self._beta:
            return

        self._beta = beta
        r = coax.regularizers.EntropyRegularizer(self.pi, self._beta)
        self.ppo_clip = coax.policy_objectives.PPOClip(self.pi, self._pi_optimizer, r)

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate: float):
        if rate == self._learning_rate:
            return

        self._learning_rate = rate
        self._pi_optimizer = optax.adam(rate)
        self._v_optimizer = optax.adam(rate * V_TO_PI_LEARNING_RATIO)
        self.ppo_clip.optimizer = self._pi_optimizer
        self.simple_td.optimizer = self._v_optimizer

    def learn(self, batch_size: int):

        # Anneal learning rate and other hypers
        # Remember 1 gen is anywhere between 1000 and 4000 games
        g = self.generation
        if g < 30:  # 60k games
            self.learning_rate = 0.0005
            self.beta = 0.01
            self.soft_update_rate = 0.01

        if 30 <= g < 75:  # 150k games
            self.learning_rate = 0.0001
            self.beta = 0.005
            self.soft_update_rate = 0.1

        if 75 <= g < 250:  # 500k games
            self.learning_rate = 0.0001
            self.beta = 0.002
            self.soft_update_rate = 0.1

        if 250 <= g < 500:  # 1M games
            self.learning_rate = 0.0001
            self.beta = 0.001
            self.soft_update_rate = 0.1

        if g >= 500:  # 5M games
            self.learning_rate = 0.0001
            self.beta = 0.0005
            self.soft_update_rate = 0.1
        del g

        mini_batch_size = 128

        if len(self.buffer) >= 1000:
            for _ in range(batch_size // mini_batch_size):
                transition_batch = self.buffer.sample(batch_size=mini_batch_size)
                metrics_v, td_error = self.simple_td.update(transition_batch, return_td_error=True)
                metrics_pi = self.ppo_clip.update(transition_batch, td_error)
                self.record_metrics(metrics_pi, self.transitions_processed)
                self.record_metrics(metrics_v, self.transitions_processed)

            self._transitions_this_gen += batch_size
            self.transitions_processed += batch_size

        if self._transitions_this_gen >= self.transitions_per_gen:
            self.pi_behavior.soft_update(self.pi, tau=self.soft_update_rate)
            self.v_targ.soft_update(self.v, tau=self.soft_update_rate)

            self.generation += 1
            self._transitions_this_gen = 0

    def add_transitions(self, batches: list[coax.reward_tracing.TransitionBatch]):
        ct = 0
        for batch in batches:
            for chunk in coax.utils.chunks_pow2(batch):
                ct += chunk.batch_size
                self.buffer.add(chunk)

        return ct

    @property
    def buffer_len(self):
        return len(self.buffer)

    @property
    def buffer_capacity(self):
        return self.buffer.capacity

    def policy(self) -> coax.Policy:
        return self.pi_behavior
