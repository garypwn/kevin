from kevin.src.engine.board_updater import RotatingBoardUpdater, FixedBoardUpdater
from kevin.src.engine.meta_observations import MetaObservationFactory
from kevin.src.engine.python_engine import PythonGameState
from kevin.src.environment.snake_environment import MultiSnakeEnv

updater = FixedBoardUpdater(11, 11)


def make_board():
    game = PythonGameState(0, updater=updater)
    env = MultiSnakeEnv(game)
    env.fancy_render = True
    return env


def test_possible_moves():
    env = make_board()
    game = env.game
    print(env.render())
    meta = MetaObservationFactory(game)
    print(meta.possible_targets)


def test_safe_targets_walls_and_body():
    env = make_board()
    game = env.game
    game.snakes["snake_0"].body = [(0, 5), (0, 4), (1, 4), (1, 5)]
    game.update_board()
    print(env.render())
    meta = MetaObservationFactory(game)
    print(meta.possible_targets["snake_0"])

    m = meta.safe_moves("snake_0")
    print(m)
    assert m == [8, 8, 0, 0]


def test_safe_targets_heads():
    env = make_board()
    game = env.game
    game.snakes["snake_0"].body = [(4, 5), (3, 5), (2, 5)]
    game.snakes["snake_1"].body = [(5, 5), (6, 5)]
    game.snakes["snake_2"].body = [(4, 7), (4, 8), (4, 9)]
    game.snakes["snake_3"].body = [(4, 3), (4, 2)]
    game.update_board()
    print(env.render())
    meta = MetaObservationFactory(game)
    print(meta.possible_targets["snake_0"])

    m = meta.safe_moves("snake_0")
    print(m)

    assert m == [0, 0, 8, 0]


def test_out_of_hp_safe_targets():
    env = make_board()
    game = env.game
    game.snakes["snake_0"].body = [(0, 5), (0, 4), (1, 4), (1, 3)]
    game.snakes["snake_0"].health = 1
    game.food = [(0, 6)]

    game.update_board()
    print(env.render())
    meta = MetaObservationFactory(game)
    print(meta.possible_targets["snake_0"])

    m = meta.safe_moves("snake_0")
    print(m)
    assert m == [8, 0, 0, 0]


def test_meta_observation():
    env = make_board()
    game = env.game
    game.snakes["snake_0"].body = [(4, 5), (3, 5), (2, 5)]
    game.snakes["snake_1"].body = [(5, 5), (6, 5)]
    game.snakes["snake_2"].body = [(4, 7), (4, 8), (4, 9)]
    game.snakes["snake_3"].body = [(4, 3), (4, 2)]
    game.update_board()
    print(env.render())
    meta = MetaObservationFactory(game)

    print(meta("snake_0"))
