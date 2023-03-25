from abc import ABC, abstractmethod
from datetime import datetime

import coax
import gymnasium
import tensorboardX
from gymnasium import spaces

from kevin.src.model.func_approximator import FuncApproximator


class Model(ABC):
    transitions_per_gen = 128 * 256  # Should be around 1-2k games
    generation = 0
    transitions_processed = 0
    logp_required = False

    def __init__(self, gym_env: gymnasium.Env,
                 func_approximator: FuncApproximator,
                 tensorboard: tensorboardX.SummaryWriter | None = None):
        self.name = "unnamed"
        self.tensorboard = tensorboard
        self.gym_env = gym_env
        self.func_approximator = func_approximator

    def set_name(self, prefix="", model_name=""):
        self.name = f"{prefix}_{model_name}_{datetime.today().strftime('%Y-%m-%d_%H%M')}"

    @abstractmethod
    def learn(self, batch_size: int):
        """
        Updates the model on a number of transitions sampled from those added with add_transitions()
        @param batch_size: The number of transitions to learn form
        @return:
        """
        pass

    @abstractmethod
    def add_transitions(self, batches: list[coax.reward_tracing.TransitionBatch]):
        """
        Adds a list of transition batches to a buffer or queue. When learn() is called, transitions will be taken
        from here.
        @param batches:
        @return: The number of transitions added
        """
        pass

    @property
    def buffer_len(self):
        return None

    @property
    def buffer_capacity(self):
        return None

    @abstractmethod
    def policy(self) -> coax.Policy:
        """
        The policy function that will select actions when training
        @return: The policy
        """
        pass

    @abstractmethod
    def checkpoint(self, directory_path: str) -> str:
        """
        Saves the model
        @param directory_path: the .checkpoint directory
        @return: the name of the file
        """
        pass

    @abstractmethod
    def build_from_file(self, filename: str):
        """
        Builds the model from a file
        @param filename:
        @return:
        """
        pass

    @abstractmethod
    def build(self):
        """
        Builds a new model from scratch
        @return:
        """
        pass

    def record_metrics(self, metrics: dict, time_step):
        for name, metric in metrics.items():
            self.tensorboard.add_scalar(str(name), float(metric), global_step=time_step)
