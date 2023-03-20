from typing import Final, Callable

import jax
import jax.numpy as jnp


def translate_action_fixed(action: int, body: list[tuple[int, int]]) -> tuple[int, int]:
    match action:
        case 0:
            target = (0, 1)
        case 1:
            target = (1, 0)
        case 2:
            target = (0, -1)
        case 3:
            target = (-1, 0)
        case _:
            raise ValueError

    return target


def translate_action_rotating(action: int, body: list[tuple[int, int]]) -> tuple[int, int]:
    x, y = body[0]  # head
    hx, hy = (0, 1)  # Heading
    if body[0] != body[1]:
        x1, y1 = body[1]
        hx, hy = (x - x1, y - y1)

    match action:
        case 1:  # Forward
            target = (x + hx, y + hy)
        case 0:  # Left
            target = (x - hy, y + hx)
        case 2:  # Right
            target = (x + hy, y - hx)
        case _:
            raise ValueError

    return target


def get_heading(body: list[tuple[int, int]]) -> int:
    if len(body) < 2:
        return 0

    head = body[0]
    x0, y0 = head
    x1, y1 = body[1]
    heading = (x0 - x1, y0 - y1)
    match heading:
        case (0, 0):
            d = 0
        case (0, 1):
            d = 0
        case (1, 0):
            d = 1
        case (0, -1):
            d = 2
        case (-1, 0):
            d = 3
        case _:
            raise ValueError

    return d


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

    get_target: Callable[[int, list[tuple[int, int]]], tuple[int, int]]

    _snake_placer: Final[Callable[[list[tuple[int, int]], int, int, jax.Array], jax.Array]]

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
            if i + self.batch_size > len(body):
                batch = jnp.array(body[i:])
                cutoff = len(body) % self.batch_size
            else:
                batch = jnp.array(body[i: i + self.batch_size])
                cutoff = self.batch_size + 1
            board = self._snake_placer(batch, len(body) - i, cutoff, board)

        return board

    def _place_snake(self, batch, rev_idx, cutoff, board):

        for i in range(self.batch_size):

            if i >= cutoff:
                break

            x, y = batch[i]
            board = board.at[x, y].max(rev_idx - i)

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
            if i + self.batch_size > len(food):
                batch = jnp.array(food[i:])
                cutoff = len(food) % self.batch_size
            else:
                batch = jnp.array(food[i: i + self.batch_size])
                cutoff = self.batch_size + 1
            board = self._food_placer(batch, cutoff, board)

        return board

    def _place_food(self, batch, cutoff, board):

        for i in range(self.batch_size):

            if i >= cutoff:
                break

            x, y = batch[i]
            board = board.at[x, y].set(1)

        return board

    def __init__(self, width, height, batch_size=5, jit_enabled=True, donate=True):

        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.viewport_size = 1 + 2 * max(width, height)  # Guaranteed to be odd

        if jit_enabled:
            if donate:
                self._snake_placer = jax.jit(self._place_snake, donate_argnums=3, static_argnums=2)
                self._food_placer = jax.jit(self._place_food, donate_argnums=2, static_argnums=1)
                self._board_pov_maker = jax.jit(self._create_pov, static_argnums=1)
                self._walls_pov_maker = jax.jit(self._create_walls, static_argnums=1)
            else:
                self._snake_placer = jax.jit(self._place_snake, static_argnums=2)
                self._food_placer = jax.jit(self._place_food, static_argnums=1)
                self._board_pov_maker = jax.jit(self._create_pov, static_argnums=1)
                self._walls_pov_maker = jax.jit(self._create_walls, static_argnums=1)

        else:
            self._snake_placer = self._place_snake
            self._food_placer = self._place_food
            self._board_pov_maker = self._create_pov
            self._walls_pov_maker = self._create_walls

        self.get_target = translate_action_fixed


class RotatingBoardUpdater(BoardUpdater):

    def __init__(self, width, height, batch_size=5, jit_enabled=True, donate=True):
        super().__init__(width, height, batch_size=batch_size, jit_enabled=jit_enabled, donate=donate)

        if jit_enabled:
            if donate:
                self._board_pov_maker = jax.jit(self._create_pov, static_argnums=1)
                self._walls_pov_maker = jax.jit(self._create_walls, static_argnums=1)
            else:
                self._board_pov_maker = jax.jit(self._create_pov, static_argnums=1)
                self._walls_pov_maker = jax.jit(self._create_walls, static_argnums=1)

        else:
            self._board_pov_maker = self._create_pov
            self._walls_pov_maker = self._create_walls

        self.get_target = translate_action_rotating

    def snake_pov(self, body, board):
        facing = get_heading(body)
        return self._board_pov_maker(body[0], facing, board)

    _board_pov_maker: Final[Callable[[tuple[int, int], int, jax.Array], jax.Array]]

    def _create_pov(self, head, facing, board):
        """
        Place the snake in the middle of the board, facing the snake's current heading
        """
        x, y = head
        donate_board = jnp.zeros([self.viewport_size, self.viewport_size], dtype=jnp.int16)
        o_x = self.width - x
        o_y = self.height - y

        donate_board = jax.lax.dynamic_update_slice(donate_board, board, (o_x, o_y))

        return jnp.rot90(donate_board, facing)

    _walls_pov_maker: Final[Callable[[tuple[int, int], int], jax.Array]]

    def walls_pov(self, body):
        facing = get_heading(body)
        return self._walls_pov_maker(body[0], facing)

    def _create_walls(self, head, facing):
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
