from pettingzoo.test import parallel_api_test, parallel_seed_test

from kevin.src.engine.python_engine import PythonStandard4Player
from kevin.src.environment.snake_environment import MultiSnakeEnv


def test_pettingzoo_api_test():
    game = PythonStandard4Player()
    env = MultiSnakeEnv(game)
    parallel_api_test(env, 1000)


def test_pettingzoo_seed_test():
    def construct_game():
        game = PythonStandard4Player()
        return MultiSnakeEnv(game)
    parallel_seed_test(construct_game, num_cycles=10, test_kept_state=True)
