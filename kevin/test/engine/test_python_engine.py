import pytest

from kevin.src.engine.python_engine import PythonStandard4Player
import jax.numpy as jnp


def test_print_game():
    game = PythonStandard4Player(99)
    print(game)  # Requires visual inspection


@pytest.mark.parametrize("seed", range(0, 200000, 10007))
def test_spawn_determinism(seed: int):
    games = (PythonStandard4Player(seed), PythonStandard4Player(seed))
    print(games[0])
    assert jnp.array_equal(games[0].board, games[1].board)


@pytest.mark.parametrize("seed", range(50000, 700000, 50023))
def test_no_initial_food_in_corner(seed: int):
    game = PythonStandard4Player(seed)
    print(game)
    for i in range(4):
        assert jnp.rot90(game.board, k=i)[0, 0] == 0


@pytest.mark.parametrize("seed", range(800000, 1700000, 77023))
def test_count_initial_food_and_snakes(seed: int):
    game = PythonStandard4Player(seed)
    print(game)
    food_count = 0
    snake_count = 0
    for row in game.board.tolist():
        for i in row:
            if i == 1:
                food_count += 1
            if i != 0 and i != 1:
                snake_count += 1

    assert food_count == 5
    assert snake_count == 4


def test_snakes_move(seed: int = 0):
    game = PythonStandard4Player(seed)
    print(game)
    for name, move in zip(game.snakes, [0, 3, 1, 2]):
        game.submit_move(name, move)

    for i in range(9):
        game.step()
        print(game)


def test_observations_have_unique_perspective(seed: int = 0):
    #  todo
    pass


def test_same_turn_observations_have_same_board(seed: int = 0):
    #  todo
    pass


def test_elimination_on_wall(seed: int = 0):
    #  todo
    pass


def test_elimination_on_body_collision(seed: int = 0):
    #  todo
    pass


def test_elimination_on_head_collision(seed: int = 0):
    #  todo
    pass


def test_elimination_on_0_hp(seed: int = 0):
    #  todo
    pass


def test_reward_on_victory(seed: int = 0):
    #  todo
    pass


def test_reward_on_defeat(seed: int = 0):
    #  todo
    pass


def test_reward_on_neutral(seed: int = 0):
    #  todo
    pass
