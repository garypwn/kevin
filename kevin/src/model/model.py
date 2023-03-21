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
            hk.Linear(self.action_space.n, w_init=jnp.zeros, name="pi_head_output")
        ])
        result = self.body(S, is_training)
        conv = hk.Conv2D(2, (1, 1), data_format="NHWC", name="pi_head_conv")(result)
        norm = hk.BatchNorm(True, True, 0.999, data_format="NHWC", name="pi_head_norm")(conv, is_training)
        return {'logits': logits(norm)}

    def v(self, S, is_training):
        value = hk.Sequential([
            hk.Linear(256, name="v_head_linear"),
            jnn.relu,
            hk.Flatten(),
            hk.Linear(1, name="v_head_output"),
            jnp.tanh, jnp.ravel
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
            batch_norm = hk.BatchNorm(True, True, 0.999, data_format="NHWC", name=self.batch_name)
            conv2d = hk.Conv2D(256, self.shape, data_format="NHWC", name=self.conv_name)
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
        hk.Reshape([23*7, 23], preserve_dims=1),  # Concatenate stacked feature maps
        ConvNorm(3, "top"), jnn.relu,
        ResCore(3, "res_core_0"), jnn.relu,
        ResCore(3, "res_core_1"), jnn.relu,
        ResCore(3, "res_core_2"), jnn.relu,
        ResCore(3, "res_core_3"), jnn.relu,
        ResCore(3, "res_core_4"), jnn.relu,
    ])

    boards = jnp.float32(x)

    return conv(boards)
