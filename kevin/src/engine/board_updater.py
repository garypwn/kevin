from typing import Final, Callable

import jax
import jax.numpy as jnp


class BoardUpdater:
    r"""
    A function that computes the array representation of a board. A callable is constructed with a fixed
    board height and width, and a fixed max number of players.
    It is a pure function that accepts all state required to create the board array.

    It also has a max snake length based on the dimensions of the board. If all snakes are within
    the dimensions, it uses the jitted version, and otherwise falls back on the interpreter.

    Also, hazards aren't supported right now.
    """

    width: Final[int]
    height: Final[int]
    batch_size: Final[int]
    viewport_size: Final[int]

    _snake_placer: Final[Callable[[list[tuple[int, int]], int, jax.Array], jax.Array]]

    def snake_sub_board(self, body, board):
        """
        Places a snake on a board filled with zeroes, or continues placing a snake on a board filled
        with a partial snake.

        Snakes are represented with their head having value equal to their length, and their tail having value 1.
        """
        board = jnp.zeros([self.width, self.height], dtype=jnp.int16)
        if body is None or len(body) == 0:
            return board

        for i in range(0, len(body), self.batch_size):
            board = self._place_snake(body, i, board)

        return board

    def _place_snake(self, body, whence, board):

        for i in range(self.batch_size):
            idx = whence + i

            if i >= len(body):
                break

            x, y = body[idx]
            board = board.at[x, y].set(len(body) - idx)

        return board

    _food_placer: Final[Callable[[list[tuple[int, int]], int, jax.Array], jax.Array]]

    def food_sub_board(self, food, board):
        """
        Place food on the board
        """
        board = jnp.zeros([self.width, self.height], dtype=jnp.int16)
        if food is None or len(food) == 0:
            return board

        for i in range(0, len(food), self.batch_size):
            board = self._food_placer(food, i, board)

        return board

    def _place_food(self, body, whence, board):

        for i in range(self.batch_size):
            idx = whence + i

            if i >= len(body):
                break

            x, y = body[idx]
            board = board.at[(x, y)].set(1)

        return board

    snake_pov: Final[Callable[[tuple[int, int], int, jax.Array, jax.Array], jax.Array]]

    def _create_pov(self, head, facing, board, donate_board):
        """
        Place the snake in the middle of the board, facing the snake's current heading
        """
        x, y = head
        donate_board = jnp.zeros([self.viewport_size, self.viewport_size], dtype=jnp.int16)
        o_x = self.width - x
        o_y = self.height - y

        donate_board = jax.lax.dynamic_update_slice(donate_board, board, (o_x, o_y))

        return jnp.rot90(donate_board, facing)

    walls_pov: Final[Callable[[tuple[int, int], int, jax.Array], jax.Array]]

    def _create_walls(self, head, facing, donate_board):
        """
        Place the snake in the middle of the board, facing the snake's current heading
        """

        board = jnp.zeros([self.width, self.height], dtype=jnp.int16)
        donate_board = jnp.ones([self.viewport_size, self.viewport_size], dtype=jnp.int16)
        x, y = head
        o_x = self.width - x
        o_y = self.height - y

        donate_board = jax.lax.dynamic_update_slice(donate_board, board, (o_x, o_y))

        return jnp.rot90(donate_board, facing)

    def __init__(self, width, height, batch_size=5, jit_enabled=True, donate=True):

        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.viewport_size = 1 + 2 * max(width, height)  # Guaranteed to be odd

        if jit_enabled:
            if donate:
                self._snake_placer = jax.jit(self._place_snake, donate_argnums=2)
                self._food_placer = jax.jit(self._place_food, donate_argnums=2)
                self.snake_pov = jax.jit(self._create_pov, static_argnums=1, donate_argnums=3)
                self.walls_pov = jax.jit(self._create_pov, static_argnums=1, donate_argnums=2)
            else:
                self._snake_placer = jax.jit(self._place_snake)
                self._food_placer = jax.jit(self._place_food)
                self.snake_pov = jax.jit(self._create_pov, static_argnums=1)
                self.walls_pov = jax.jit(self._create_walls, static_argnums=1)
        else:
            self._snake_placer = self._place_snake
            self._food_placer = self._place_food
            self.snake_pov = self._create_pov
            self.walls_pov = self._create_walls
