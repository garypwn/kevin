from typing import Callable

import jax
from gymnasium import spaces
import haiku as hk
import jax.numpy as jnp
import jax.nn as jnn


class Model:
    body: Callable
    action_space: spaces.Discrete

    def __init__(self, body, space):
        self.body = body
        self.action_space = space

    def pi_logits(self, S, is_training):
        logits = hk.Sequential([
            hk.Linear(8), jnn.relu,
            hk.Flatten(),
            hk.Linear(self.action_space.n, w_init=jnp.zeros),
        ])
        result = self.body(S, is_training)
        return {'logits': logits(result)}

    def v(self, S, is_training):
        value = hk.Sequential([
            hk.Linear(8), jnn.relu,
            hk.Flatten(),
            hk.Linear(1, w_init=jnp.zeros),
            jnp.ravel
        ])
        result = self.body(S, is_training)
        return value(result)

    def q(self, S, is_training):
        seq = hk.Sequential([
            hk.Linear(8), jnn.relu,
            hk.Flatten(),
            hk.Linear(self.action_space.n, w_init=jnp.zeros),
        ])
        result = self.body(S, is_training)
        return seq(result)


def linear_body(x, is_training):
    boards = jnp.float32(x["boards"])
    body = hk.Sequential([
        hk.nets.MLP([1024, 1024, 512, 256, 128, 64, 32, 16, 8]),
        hk.Flatten()
    ])
    return body(boards)


def conv_body(x, is_training):
    boards = jnp.float32(x["boards"])

    class Core:
        def __init__(self, shape):
            self.shape = [shape, shape]

        def __call__(self, s):
            batch_norm = hk.BatchNorm(True, True, 0.999, data_format="NHWDC")
            conv3d = hk.Conv2D(256, self.shape, data_format="NHWDC")
            return batch_norm(conv3d(s), is_training)

    conv = hk.Sequential([
        Core(3), jnn.relu,
        Core(3), jnn.relu,
        Core(1), jnn.relu,
        hk.Flatten(),
    ])

    return conv(boards)


def residual_body(x, is_training):
    boards = jnp.float32(x["boards"])

    class ConvNorm:
        def __init__(self, shape):
            self.shape = [shape, shape]

        def __call__(self, s):
            batch_norm = hk.BatchNorm(True, True, 0.99, data_format="NHWDC")
            conv2d = hk.Conv2D(256, self.shape, data_format="NHWDC")
            return batch_norm(conv2d(s), is_training)

    class ResCore:

        def __call__(self, s):
            convoluted = hk.Sequential([
                ConvNorm(1), jnn.relu,
                ConvNorm(3), jnn.relu,
                ConvNorm(1)
            ])
            return jnp.add(convoluted(s), s)

    conv = hk.Sequential([
        ConvNorm(3), jnn.relu,
        ResCore(), jnn.relu,
        ResCore(), jnn.relu,
        ResCore(), jnn.relu,
        ResCore(), jnn.relu,
    ])

    return conv(boards)
