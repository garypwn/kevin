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
            jax.nn.relu,
            hk.Linear(self.action_space.n, w_init=jnp.zeros),
        ))
        return {'logits': logits(result)}

    def v(self, S, is_training):
        result = self.body(S, is_training)
        value = hk.Sequential((
            jax.nn.relu,
            hk.Linear(1, w_init=jnp.zeros),
        ))
        return jnp.ravel(value(result))

    def q(self, S, is_training):
        result = self.body(S, is_training)
        seq = hk.Sequential((
            jax.nn.relu,
            hk.Linear(self.action_space.n, w_init=jnp.zeros),
        ))
        return seq(result)


def process_obs(x, is_training):
    board = jnp.float32(x["board"])
    aug_board = 2 * board + 10  # Use this to make padding work better
    turn = jnp.float32(x["turn"])
    snakes = jnp.float32(x["snakes"])

    # Process the board
    conv = hk.Sequential((
        hk.Conv2D(512, 3, padding=hk.pad.create(hk.pad.same, [3, 3], 1, 2)), jax.nn.relu,
        hk.Conv2D(256, 5, padding=hk.pad.create(hk.pad.same, [5, 5], 1, 2)), jax.nn.relu,
        hk.Flatten()
    ))

    # conv3_flat = hk.Sequential((conv3, jax.nn.relu, hk.Flatten()))(board)
    # conv5_flat = hk.Sequential((conv5, jax.nn.relu, hk.Flatten()))(board)

    result = jnp.concatenate((turn, snakes, conv(aug_board)), 1)

    mlp = hk.nets.MLP([2048, 1024, 512, 256, 128, 64, 16])
    return mlp(result)


def resnet(x, is_training):
    board = jnp.float32(x["board"])
    aug_board = 2 * board + 10  # Use this to make padding work better
    turn = jnp.float32(x["turn"])
    snakes = jnp.float32(x["snakes"])

    # Process the board
    conv = hk.nets.ResNet18(128)

    result = jnp.concatenate((turn, snakes, conv(aug_board, is_training)), 1)

    mlp = hk.nets.MLP([64, 16, 8])
    return mlp(result)
