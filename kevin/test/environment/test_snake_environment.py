import cProfile
import pstats

import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test, performance_benchmark
from pettingzoo.utils import parallel_to_aec

from kevin.src.engine.python_engine import PythonStandard4Player
from kevin.src.environment.snake_environment import MultiSnakeEnv


def test_pettingzoo_api_test():
    game = PythonStandard4Player()
    env = MultiSnakeEnv(game)
    parallel_api_test(env, 1000)


@pytest.mark.parametrize("seed", range(0, 50000, 10013))
def test_pettingzoo_seed_test(seed: int):
    def construct_game():
        game = PythonStandard4Player(seed)
        return MultiSnakeEnv(game)

    parallel_seed_test(construct_game, num_cycles=10, test_kept_state=True)


def test_performance_benchmark():
    # Requires manual inspection
    game = PythonStandard4Player()
    env = MultiSnakeEnv(game)
    aec_env = parallel_to_aec(env)
    performance_benchmark(aec_env)


def test_profile_performance():

    game = PythonStandard4Player()
    env = MultiSnakeEnv(game)
    aec_env = parallel_to_aec(env)

    profiler = cProfile.Profile()
    profiler.enable()
    performance_benchmark(aec_env)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

