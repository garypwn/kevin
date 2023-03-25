import cProfile
import pstats

import coax
import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test, performance_benchmark
from pettingzoo.utils import parallel_to_aec

from kevin.src.engine.python_engine import PythonGameState, RotatingBoardUpdater
from kevin.src.environment.snake_environment import MultiSnakeEnv
from kevin.src.environment.wrapper import FlatteningWrapper


def test_pettingzoo_api_test():
    game = PythonGameState()
    env = MultiSnakeEnv(game)
    parallel_api_test(env, 1000)


@pytest.mark.parametrize("seed", range(0, 50000, 10013))
def test_pettingzoo_seed_test(seed: int):
    def construct_game():
        game = PythonGameState(seed)
        return MultiSnakeEnv(game)

    parallel_seed_test(construct_game, num_cycles=10, test_kept_state=True)


def test_action_space():
    game = PythonGameState(0)
    env = MultiSnakeEnv(game)
    space = env.action_space("snake_0")
    print(space)
    for i in range(4):
        assert i in space

    assert -1 not in space
    assert 4 not in space


def test_performance_benchmark():
    # Requires manual inspection
    updater = RotatingBoardUpdater(11, 11, 4, False)
    game = PythonGameState(updater=updater)
    env = MultiSnakeEnv(game)
    aec_env = parallel_to_aec(env)
    performance_benchmark(aec_env)


def test_jitted_performance_benchmark():
    # Requires manual inspection
    updater = RotatingBoardUpdater(11, 11, 4, True)
    updater.jitted_board([], []).block_until_ready()
    game = PythonGameState(updater=updater)
    env = MultiSnakeEnv(game)
    aec_env = parallel_to_aec(env)
    performance_benchmark(aec_env)


def test_profile_performance():
    updater = RotatingBoardUpdater(11, 11, 4, True)
    game = PythonGameState(updater=updater)
    env = MultiSnakeEnv(game)
    aec_env = parallel_to_aec(env)

    #  Spool up the jit
    for i in range(10):
        game.seed(i)
        game.reset()
        game.global_observation()

    profiler = cProfile.Profile()
    profiler.enable()
    performance_benchmark(aec_env)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()


def test_flattening_wrapper():
    game = PythonGameState()
    env = MultiSnakeEnv(game)
    wrapped_env = FlatteningWrapper(env)
    print("\n")

    print(env.observation_space("snake_0"))
    print("\n")
    print(wrapped_env.observation_space("snake_0"))

    obs = wrapped_env.reset()
    print(obs)

    for agent in wrapped_env.agents:
        assert obs[agent] in wrapped_env.observation_space(agent)


def test_value_func():
    game = PythonGameState()
    env = MultiSnakeEnv(game)

    import haiku as hk
    import jax.numpy as jnp
    import jax

    def val(S, is_training):
        value = hk.Sequential([
            jnp.float32,
            hk.Flatten(),
            hk.Linear(256), jax.nn.relu,
            hk.Linear(1, name="v_head_output", w_init=jnp.zeros),
            jnp.ravel
        ])
        return value(S)

    v = coax.V(val, env.dummy_gym_environment)
    print(v(env.observation_space('snake_0').sample()))
    print(v(env.observation_space('snake_0').sample()))
    print(v(env.observation_space('snake_0').sample()))
