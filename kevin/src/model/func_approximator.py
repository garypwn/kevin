from typing import Callable

import haiku as hk
import jax.nn as jnn
import jax.numpy as jnp
from gymnasium import spaces


class FuncApproximator:
    body: Callable
    action_space: spaces.Discrete

    def __init__(self, body, space):
        self.body = body
        self.action_space = space

    def pi_logits(self, S, is_training):
        logits = hk.Sequential([
            hk.Flatten(),
            hk.Linear(256),
            hk.Linear(self.action_space.n, name="pi_head_output", w_init=jnp.zeros)
        ])
        return {'logits': logits(self.body(S, is_training))}

    def v(self, S, is_training):
        value = hk.Sequential([
            hk.Flatten(),
            hk.Linear(256),
            hk.Linear(1, name="v_head_output", w_init=jnp.zeros),
            jnp.ravel
        ])
        result = self.body(S, is_training)
        return value(result)

    def q(self, S, is_training):
        seq = hk.Sequential([
            hk.Flatten(),
            hk.Linear(256),
            hk.Linear(self.action_space.n, name="q_head_output", w_init=jnp.zeros)
        ])
        return seq(self.body(S, is_training))


def simple_linear_body(x, is_training):
    boards = jnp.float32(jnp.moveaxis(x, 1, 3))
    lin = hk.Sequential([
        hk.Linear(256, name="body_0"), jnn.relu,
        hk.Linear(256, name="body_1"), jnn.relu
    ])
    return lin(boards)


def residual_body(x, is_training):
    class ConvNorm:
        def __init__(self, shape, name=None):
            if name is None:
                self.batch_name = None
                self.conv_name = None
            else:
                self.batch_name = name + "_batchnorm"
                self.conv_name = name + "_conv2d"
            self.shape = [shape, shape]

        def __call__(self, s):
            batch_norm = hk.BatchNorm(True, True, 0.999, data_format="N...C", name=self.batch_name)
            conv2d = hk.Conv2D(256, self.shape, data_format="N...C", name=self.conv_name)
            return batch_norm(conv2d(s), is_training)

    class ResCore:

        def __init__(self, kernel_size=3, name=None):
            if name is None:
                self.conv1_name = None
                self.conv2_name = None
            else:
                self.conv1_name = name + "_first"
                self.conv2_name = name + "_second"

            self.kernel_size = kernel_size

        def __call__(self, s):
            convoluted = hk.Sequential([
                ConvNorm(self.kernel_size, name=self.conv1_name), jnn.relu,
                ConvNorm(self.kernel_size, name=self.conv2_name)
            ])
            return jnp.add(s, convoluted(s))

    conv = hk.Sequential([

        hk.Reshape([-1, 23]),

        # 3x3 -> adjacent things. Things that might happen next turn.
        hk.Conv2D(256, [3, 3], data_format="N...C", name="body_0"), jnn.relu,

        ResCore(3, "res_core_0"), jnn.relu,
        ResCore(3, "res_core_1"), jnn.relu,

    ])

    boards = jnp.float32(jnp.moveaxis(x, 1, 3))
    boards = 10 * jnp.log(1 + jnp.log(1 + boards))  # Transform to make zeros a bigger deal

    return conv(boards)
