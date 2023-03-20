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
            hk.Linear(self.action_space.n, w_init=hk.initializers.RandomUniform(),
                      b_init=hk.initializers.RandomUniform()),
        ])
        result = self.body(S, is_training)
        conv = hk.Conv2D(2, 1, data_format="NDWHC", w_init=hk.initializers.RandomUniform(),
                         b_init=hk.initializers.RandomUniform())(result)
        norm = hk.BatchNorm(True, True, 0.999, data_format="NDWHC")(conv, is_training)
        return {'logits': logits(norm)}

    def v(self, S, is_training):
        value = hk.Sequential([
            hk.Linear(256, w_init=hk.initializers.RandomUniform(),
                      b_init=hk.initializers.RandomUniform()),
            jnn.relu,
            hk.Flatten(),
            hk.Linear(1, w_init=hk.initializers.RandomUniform(),
                      b_init=hk.initializers.RandomUniform()),
            jnp.ravel, jnp.tanh
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


def residual_body(x, is_training):
    boards = jnp.float32(x)

    class ConvNorm:
        def __init__(self, shape):
            self.shape = [shape, shape]

        def __call__(self, s):
            batch_norm = hk.BatchNorm(True, True, 0.999, data_format="NDWHC")
            conv2d = hk.Conv2D(256, self.shape, data_format="NDWHC", w_init=hk.initializers.RandomUniform(),
                               b_init=hk.initializers.RandomUniform(), )
            return batch_norm(conv2d(s), is_training)

    class ResCore:

        def __call__(self, s):
            convoluted = hk.Sequential([
                ConvNorm(1), jnn.relu,
                ConvNorm(3), jnn.relu,
                ConvNorm(1)
            ])
            return jnp.add(convoluted(s), s)

    top = hk.Sequential([ConvNorm(3), jnn.relu])(boards)

    def skip(s):
        return jnp.add(s, top)

    conv = hk.Sequential([
        ConvNorm(3), jnn.relu,
        ResCore(), jnn.relu,
        ResCore(), jnn.relu,
        ResCore(), jnn.relu,
        ResCore(), skip, jnn.relu,
        ResCore(), jnn.relu,
        ResCore(), jnn.relu,
        ResCore(), jnn.relu,
    ])

    return conv(top)
