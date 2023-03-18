from typing import Callable

import jax
from gymnasium import spaces
import haiku as hk
import jax.numpy as jnp


class Model:
    body: Callable
    action_space: spaces.Discrete

    def __init__(self, body, space):
        self.body = body
        self.action_space = space

    def pi_logits(self, S, is_training):
        result = self.body(S, is_training)
        logits = hk.Sequential((
            hk.Flatten(),
            hk.Linear(16), jax.nn.relu,
            hk.Linear(self.action_space.n, w_init=jnp.zeros),
        ))
        return {'logits': logits(result)}

    def v(self, S, is_training):
        result = self.body(S, is_training)
        value = hk.Sequential((
            hk.Flatten(),
            hk.Linear(16), jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros),
        ))
        return jnp.ravel(value(result))

    def q(self, S, is_training):
        result = self.body(S, is_training)
        seq = hk.Sequential((
            hk.Flatten(),
            hk.Linear(16), jax.nn.relu,
            hk.Linear(self.action_space.n, w_init=jnp.zeros),
        ))
        return seq(result)


def simple_body(x, is_training):
    boards = jnp.float32(x["boards"])

    class Core:
        def __init__(self, shape):
            self.shape = [len(boards), shape, shape]

        def __call__(self, s):
            return hk.BatchNorm(True, True, 0.999)(hk.Conv3D(256, self.shape)(s), is_training)

    conv = hk.Sequential((
        Core(3), jax.nn.relu,
        Core(3), jax.nn.relu,
        Core(1), jax.nn.relu
    ))

    return conv(boards)
