import jax.numpy as jnp

from kevin.src.engine.board_updater import RotatingBoardUpdater, FixedBoardUpdater

test_board = jnp.full([5, 5], 3, dtype=jnp.int16)

updater = FixedBoardUpdater(5, 5, donate=False)

test_viewport = jnp.zeros([updater.viewport_size, updater.viewport_size], dtype=jnp.int16)


def test_pov():
    print(test_board)
    print(updater.snake_pov([(2, 2)], test_board))


def test_walls():
    print(test_board)
    print(updater.walls_pov([(4, 4), (-1, 0)]))


def test_place_snake():
    snake = [(2, 2), (2, 3), (2, 4), (3, 4), (4, 4), (4, 3)]
    board = updater.snake_sub_board(snake, test_board)
    print("\n", board)
    print(updater.snake_pov(snake, board))


