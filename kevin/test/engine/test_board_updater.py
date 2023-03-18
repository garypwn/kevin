import jax.numpy as jnp

from kevin.src.engine.board_updater import BoardUpdater

test_board = jnp.full([5, 5], 3, dtype=jnp.int16)

updater = BoardUpdater(5, 5, donate=False)

test_viewport = jnp.zeros([updater.viewport_size, updater.viewport_size], dtype=jnp.int16)


def test_pov():
    print(test_board)
    print(updater.snake_pov((2, 2), 0, test_board, test_viewport))


def test_walls():
    print(test_board)
    print(updater.walls_pov((0, 0), 2, test_viewport))


def test_place_snake():
    snake = [(2, 2), (2, 3), (2, 4), (3, 4)]
    board = updater.snake_sub_board(snake, test_board)
    print("\n", board)
    print(updater.snake_pov((2, 2), 2, board, test_viewport))


