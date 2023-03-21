from typing import Callable

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
from gymnasium import spaces


class Model:
    body: Callable
    action_space: spaces.Discrete

    def __init__(self, body, space):
        self.body = body
        self.action_space = space

    def pi_logits(self, S, is_training):
        logits = hk.Sequential([
            jnn.relu,
            hk.Flatten(),
            hk.Linear(self.action_space.n, w_init=jnp.zeros)
        ])
        result = self.body(S, is_training)
        conv = hk.Conv2D(2, 1, data_format="NWHC")(result)
        norm = hk.BatchNorm(True, True, 0.999, data_format="NWHC")(conv, is_training)
        return {'logits': logits(norm)}

    def v(self, S, is_training):
        value = hk.Sequential([
            hk.Linear(256),
            jnn.relu,
            hk.Flatten(),
            hk.Linear(1),
            jnp.ravel, jnp.tanh
        ])
        result = self.body(S, is_training)
        return value(result)

    def q(self, S, is_training):
        seq = hk.Sequential([
            hk.Linear(8), jnn.relu,
            hk.Flatten(),
            hk.Linear(self.action_space.n),
        ])
        result = self.body(S, is_training)
        return seq(result)


def residual_body(x, is_training):
    boards = jnp.float32(x)

    class ConvNorm:
        def __init__(self, shape):
            self.shape = [shape, shape]

        def __call__(self, s):
            batch_norm = hk.BatchNorm(True, True, 0.999, data_format="NWHC")
            conv2d = hk.Conv2D(256, self.shape, data_format="NWHC")
            return batch_norm(conv2d(s), is_training)

    class ResCore:

        def __init__(self, kernel_size=3):
            self.kernel_size = kernel_size

        def __call__(self, s):
            convoluted = hk.Sequential([
                ConvNorm(1), jnn.relu,
                ConvNorm(self.kernel_size), jnn.relu,
                ConvNorm(1)
            ])
            return jnp.add(s, convoluted(s))

    conv = hk.Sequential([
        hk.Reshape([-1, 23]),  # Concatenate stacked feature maps
        ConvNorm(3), jnn.relu,
        ResCore(3), jnn.relu,
        ResCore(3), jnn.relu,
        ResCore(3), jnn.relu,
        ResCore(3), jnn.relu,
        ResCore(3), jnn.relu,
    ])

    return conv(boards)
