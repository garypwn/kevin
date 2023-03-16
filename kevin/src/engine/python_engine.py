import random
import sys
from math import sqrt, log
from typing import Final, Callable

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from kevin.src.engine.snake_engine import SnakeEngine


class Snake:

    def __getitem__(self, item):
        return self.body[item]

    id: str
    health: int
    body: list[tuple[int, int]]


class PythonStandard4Player(SnakeEngine):
    r""" Attempts to emulate 4 player standard battlesnake logic in python
    """

    #  Game options
    player_count: Final[int] = 4  # todo support more than 4 players
    height: Final[int] = 11
    width: Final[int] = 11

    #  State
    turn_num: int = 0

    rng_key: jax.Array | jrand.PRNGKeyArray
    rng_seed: int

    snakes: dict[str: Snake]

    #  Coords are in the shape (x, y)
    food: list[tuple[int, int]]
    hazards: list[tuple[int, int]] = []

    #  For presenting in observations. Updated every step.
    snakes_array: list[int] = []  # For observations
    boards: dict[str: jax.Array]

    #  Submitted moves
    pending_moves: dict[str: int]

    #  Board updater fn
    updater: Callable[[list[list[tuple[int, int]]], list[tuple[int, int]], jax.Array], jax.Array]

    #  If true, the final snake will not terminate automatically.
    single_player_mode = False

    def _random(self) -> jax.Array | jrand.PRNGKeyArray:
        r"""
        Generates a prng subkey and updates the instance key.
        NEVER use another prng to preserve determinism.
        :return: A new prng subkey
        """
        self.rng_key, subkey = jrand.split(self.rng_key)
        return subkey

    def _is_occupied(self, pt: tuple[int, int]) -> bool:
        r"""
        Checks if a point is occupied.
        :param pt:
        :return:
        """
        if pt in self.food:
            return False
        if pt in self.hazards:
            return False

        for _, snake in self.snakes.items():
            if pt in snake.body:
                return False

    def _random_unoccupied_pt(self):
        r"""
        Places food in a random unoccupied location
        :return:
        """
        x: int
        y: int
        while True:
            point = jrand.randint(self._random(), shape=[2],
                                  minval=jnp.array([0, 0]),
                                  maxval=jnp.array([self.width, self.height]))
            x, y = point[0], point[1]
            if not self._is_occupied((x, y)):
                break
        return x, y

    def __init__(self, seed: int | None = None, updater=None):
        r"""
        Initialize the game with a random seed
        """

        #  Initialize with a random seed
        if seed is None:
            seed = random.randrange(sys.maxsize)

        self.rng_seed = seed

        #  Set up the updater. This allows us to pre-jit the updater and use it in all games.
        if updater is None:
            self.updater = BoardUpdater(self.width, self.height, self.player_count)
        else:
            self.updater = updater

        #  Initialize the boards as empty boards
        self.boards = {"snake_{}".format(i): jnp.zeros([self.width, self.height], dtype=jnp.int16)
                       for i in range(self.player_count)}

    def __getitem__(self, item):
        if item == 0:
            return self.snakes_array
        if item == 1:
            return self.turn_num
        if item == 2:
            return self.boards

        raise IndexError

    def __str__(self):
        turn = "Turn {}.".format(self.turn_num)
        snakes = {"Snake {}: {}".format(i, snake) for i, snake in enumerate(self.snakes_array)}
        jnp.set_printoptions(formatter={"int": lambda i: "{: >2}".format(i)})
        return "\n{}\n{}\n{}\n".format(turn, snakes, self.boards["snake_0"])

    def fancy_str(self):
        turn = "Turn {}.".format(self.turn_num)
        snakes = {"Snake {}: {}".format(i, snake) for i, snake in enumerate(self.snakes_array)}
        board = self.boards["snake_0"]
        l = np.full([self.width, self.height], "__")
        for i, row in enumerate(board):
            for j, col in enumerate(row):
                match board[i, j]:
                    case 0:
                        l[i, j] = "  "
                    case 1:
                        l[i, j] = "\N{tomato}"
                    case 3:
                        l[i, j] = "\N{large green square}"
                    case 4:
                        l[i, j] = "\N{large green circle}"
                    case 5:
                        l[i, j] = "\N{large green circle}"
                    case 6:
                        l[i, j] = "\N{large orange square}"
                    case 7:
                        l[i, j] = "\N{large orange circle}"
                    case 8:
                        l[i, j] = "\N{large orange circle}"
                    case 9:
                        l[i, j] = "\N{large yellow square}"
                    case 10:
                        l[i, j] = "\N{large yellow circle}"
                    case 11:
                        l[i, j] = "\N{large yellow circle}"
                    case 12:
                        l[i, j] = "\N{large blue square}"
                    case 13:
                        l[i, j] = "\N{large blue circle}"
                    case 14:
                        l[i, j] = "\N{large blue circle}"
                    case _:
                        l[i, j] = "  "

        return "\n{}\n{}\n{}\n".format(turn, snakes, l)

    def update_board(self) -> dict[str: jax.Array]:
        """
        Update the board and snake list, as well as empty points
        """

        for name, snake in self.snakes.items():
            num = int(name[6:])
            self.snakes_array[num] = snake.health

        bodies = list([snake.body for _, snake in self.snakes.items()])
        for snake, _ in self.boards.items():
            i = int(snake[6:])
            self.boards[snake] = self.updater(bodies[i:] + bodies[:i], self.food, self.boards[snake])

        return self.boards  # this is just so we can block until ready in tests

    def _eliminated(self, snake_id: str) -> bool:
        r"""
        Check if a snake is elminated. A snake is eliminated if it has 0 length.
        :param snake_id:
        :return: True if the snake is eliminated
        """
        return len(self.snakes[snake_id].body) < 1

    def _move_snakes(self):
        r""" Helper for step() """

        #  Compute next move targets
        def move_to_pt(pt: tuple[int, int], move) -> tuple[int, int]:
            x, y = pt
            match move:
                case 0:  # Up
                    return x, y + 1
                case 1:  # right
                    return x + 1, y

                case 2:  # down
                    return x, y - 1

                case 3:  # left
                    return x - 1, y

            raise ValueError

        # Two snakes can eat the same food. It only disappears after resolving.
        eaten_food: list[tuple[int, int]] = []

        #  Apply move
        for name, snake in self.snakes.items():

            # Dead snakes don't move
            if len(snake.body) == 0:
                continue

            #  Move the head and tail
            old_head = snake.body[0]
            snake.body.insert(0, move_to_pt(old_head, self.pending_moves[name]))
            snake.body.pop()

            head = snake.body[0]
            snake.health -= 1

            if head in self.hazards:
                snake.health -= 15

            if head in self.food:
                snake.health = 100
                snake.body.append(snake.body[len(snake.body) - 1])
                eaten_food.append(head)

        #  Check elimination conditions
        eliminated: set = set()
        for name, snake in self.snakes.items():

            if len(snake.body) < 1:
                #  Snake already eliminated
                continue

            head = snake.body[0]
            x, y = head

            #  Check out of bounds
            if x < 0 or y < 0:
                eliminated.add(name)

            if x >= self.width or y >= self.height:
                eliminated.add(name)

            #  Check out of health
            if snake.health <= 0:
                eliminated.add(name)

            #  Check for collision with snake body (or self)
            for id2, snake2 in self.snakes.items():
                if head in snake2.body[1:]:
                    eliminated.add(name)

            #  Check for head-to-head collisions
            for name2, snake2 in self.snakes.items():

                #  Dead snakes can't be collided with
                if len(snake2.body) < 1:
                    continue

                if name2 == name:
                    continue

                if head == snake2[0]:
                    if len(snake.body) <= len(snake2.body):
                        eliminated.add(name)

        #  Remove eaten food
        for morsel in eaten_food:
            if morsel in self.food:
                self.food.remove(morsel)

        #  Eliminate snakes
        for name in eliminated:
            self.snakes[name].body = []

    def _place_food(self):
        r"""
        Places new food on the board. It seems like default BS keep 1 food at all times, and have a 15% chance
        of spawning new food if there is already food.
        """
        if self.turn_num == 0:
            return

        min_food = 1
        food_chance = 15  # percent
        curr_food = len(self.food)

        if curr_food < min_food:
            self.food.append(self._random_unoccupied_pt())
            return

        roll = jrand.randint(self._random(), [1], minval=0, maxval=100)
        if roll[0] < food_chance:
            self.food.append(self._random_unoccupied_pt())

    def get_observation(self, snake_id: str) -> dict:
        i = int(snake_id[6:])
        ordered_snakes = self.snakes_array[i:] + self.snakes_array[:i]
        return {"turn": self.turn_num, "snakes": jnp.array(ordered_snakes), "board": self.boards[snake_id]}

    def get_terminated(self, snake_id) -> bool:

        # The last snake alive is terminated because the game has ended
        alive_snakes = list(filter(lambda s: not self._eliminated(s), [name for name, _ in self.snakes.items()]))
        if snake_id in alive_snakes and len(alive_snakes) == 1:

            # Except if it's 1-player mode
            if self.single_player_mode:
                return False

            return True

        # Dead snakes are terminated
        return self._eliminated(snake_id)

    def get_truncated(self, snake_id) -> bool:
        r"""
        Always returns false because this environment has no time limit.
        :param snake_id: ignored
        :return: False
        """
        return False

    def global_observation(self) -> dict:
        return {"snakes": self.snakes_array, "turn": self.turn_num, "board": self.boards["snake_0"]}

    def get_reward(self, snake_id) -> float:
        r"""
        Returns 1. if the snake won, -1 if the snake lost, 0 otherwise.
        :param snake_id:
        :return:
        """
        if self._eliminated(snake_id):
            # return -4.  # Losing gives a static penalty
            return -100.

        #  Check if last snake alive
        alive_snakes = list(filter(lambda s: not self._eliminated(s), [name for name, _ in self.snakes.items()]))
        if snake_id in alive_snakes and len(alive_snakes) == 1:

            # If it's single player, return a neutral reward
            if self.single_player_mode:
                return 0.01

            # return log(self.turn_num+1, 2) + 0.15*self.turn_num + 3  # Reward slowly grows the longer the game goes
            return 200.

        # Staying alive gives a small reward which tapers off as the game goes on
        # return 1.05 ** (-self.turn_num - 10) + 0.8 * 1.02 ** (-(self.turn_num - 3) ** 2)

        return 0.01

    def submit_move(self, snake_id, move: int) -> None:
        self.pending_moves[snake_id] = move

    def step(self) -> None:
        self._move_snakes()
        self._place_food()
        self.update_board()
        self.turn_num += 1

    def reset(self, options: dict | None = None) -> None:

        single_player = False

        if options is not None:
            single_player = options["single_player"]

        self.seed(self.rng_seed)  # Reset the prng

        #  Reset all state
        self.turn_num = 0
        self.snakes = {}
        self.pending_moves = {}
        self.snakes_array = [100] * self.player_count
        for i in range(self.player_count):
            name = "snake_" + str(i)
            new_snake = Snake()
            new_snake.id = name
            new_snake.health = 100
            new_snake.body = []
            self.snakes[name] = new_snake
            self.pending_moves[name] = 0

        self.food = []
        self.hazards = []

        #  Place snakes following standard BS conventions of cards -> intercards.
        #  BS normally uses a distribution algorithm for num players > 8. That's a todo.
        xn, xd, xx = 1, (self.width - 1) // 2, self.width - 2
        yn, yd, yx = 1, (self.height - 1) // 2, self.height - 2

        corners = [(xn, yn), (xn, yx), (xx, yn), (xx, yx)]
        cardinals = [(xn, yd), (xd, yn), (xd, yx), (xx, yd)]

        #  Shuffle
        order = jrand.permutation(self._random(), jnp.array([0, 1, 2, 3]))
        corners = [corners[i] for i in order]
        order = jrand.permutation(self._random(), jnp.array([0, 1, 2, 3]))
        cardinals = [cardinals[i] for i in order]

        points = corners + cardinals

        # todo add support for more snakes. Currently goes up to 8.
        if self.player_count > 8:
            raise NotImplementedError("Only supports up to 8 players.")

        for i, (_, snake) in enumerate(self.snakes.items()):
            snake.body += [points[i]] * 5  # Starting length is 5

        #  Place starting food. BS default behaviour is to place a food intercardinal to each snake. Plus one center.
        #  But, do not place food in a corner or adjacent to the center square. todo except on small boards.
        cx, cy = (self.width - 1) // 2, (self.height - 1) // 2
        for _, snake in self.snakes.items():
            hx, hy = snake.body[0]
            tentative = [
                (hx + 1, hy + 1),
                (hx + 1, hy - 1),
                (hx - 1, hy + 1),
                (hx - 1, hy - 1),
            ]

            for x, y in tentative.copy():
                if (x, y) in [
                    (0, 0),
                    (0, self.height - 1),
                    (self.width - 1, 0),
                    (self.width - 1, self.height - 1)
                ]:
                    tentative.remove((x, y))
                    continue

                if self._is_occupied((x, y)):
                    tentative.remove((x, y))
                    continue

                #  Food must be further than snake from center on at least one axis
                def bad(pn, sn, cn):
                    return abs(pn - cn) <= abs(sn - cn)

                if bad(x, hx, cx) and bad(y, hy, cy):
                    tentative.remove((x, y))

            permute = [i for i, _ in enumerate(tentative)]
            order = jrand.permutation(self._random(), jnp.array(permute))
            tentative = [tentative[i] for i in order]

            if len(tentative) != 0:
                self.food.append(tentative[0])

        # Place food in center
        if not self._is_occupied((cx, cy)):
            self.food.append((cx, cy))

        if single_player:
            # Now get rid of the other snakes if it's a single player board
            for name, snake in self.snakes.items():
                if name != "snake_0":
                    snake.health = 0
                    snake.body = []

        self.single_player_mode = single_player

        self.update_board()

    def seed(self, seed) -> None:
        self.rng_seed = seed
        self.rng_key = jrand.PRNGKey(seed)


class BoardUpdater:
    r"""
    A function that computes the array representation of a board. A callable is constructed with a fixed
    board height and width, and a fixed max number of players.
    It is a pure function that accepts all state required to create the board array.

    It also has a max snake length based on the dimensions of the board. If all snakes are within
    the dimensions, it uses the jitted version, and otherwise falls back on the interpreter.

    Also, hazards aren't supported right now.
    """

    jitted_board: Callable[[list[list[tuple[int, int], list[tuple[int, int]]]], jax.Array], jax.Array]
    width: Final[int]
    height: Final[int]
    player_count: Final[int]
    max_snake_len: Final[int]
    max_food: Final[int]
    enable_hazards = False

    def infinite_board(self, snake_bodies, food, board):
        r"""Represents a board as an array"""

        #  Empty the board
        board = jnp.zeros([self.width, self.height], dtype=jnp.int16)

        for x, y in food:
            board = board.at[x, y].set(1)

        for i, snake_body in enumerate(snake_bodies):

            #  Snake names are always of the form snake_i
            head = 3 * i + 3
            body = 3 * i + 4
            tail = 3 * i + 5

            if len(snake_body) == 0:
                #  This snake is dead
                continue

            #  Tail
            for x, y in snake_body[-1:]:
                board = board.at[x, y].set(tail)

            #  Body
            for x, y in snake_body[1:-1]:
                board = board.at[x, y].set(body)

            #  Head
            for x, y in snake_body[:1]:
                board = board.at[x, y].set(head)

        return board

    def finite_board(self, snake_bodies, food, board):
        r"""
        Represents a board up to an array. Compatible with jax jit. Input must be within class attributes.
        """

        #  Empty the board
        board = jnp.zeros([self.width, self.height], dtype=jnp.int16)

        for i in range(self.max_food):
            if i >= len(food):
                break
            x, y = food[i]
            board = board.at[x, y].set(1)

        for i in range(self.player_count):

            if i >= len(snake_bodies):
                break

            if len(snake_bodies[i]) == 0:
                continue

            tx, ty = snake_bodies[i][-1]
            hx, hy = snake_bodies[i][0]
            tail = 3 * i + 5
            body = 3 * i + 4
            head = 3 * i + 3

            board = board.at[tx, ty].set(tail)

            for j in range(1, self.max_snake_len):
                if j >= len(snake_bodies[i]) - 1:
                    break

                x, y = snake_bodies[i][j]
                board = board.at[x, y].set(body)

            board = board.at[hx, hy].set(head)
        return board

    def __init__(self, width, height, max_players, jit_enabled=True):
        self.width = width
        self.height = height
        self.player_count = max_players
        self.max_snake_len = int(1.7 * sqrt(width * height))  # 18 on a standard board
        self.max_food = max_players + 11  # 16 on a standard board with 4 players

        if jit_enabled:
            self.jitted_board = jax.jit(self.finite_board)  # is donating args safe here?
        else:
            self.jitted_board = self.infinite_board

    def __call__(self, snake_bodies, food, board):
        r"""
        Runs the board update function. Requires a list of snake bodies, list of food points,
        as well as a buffer donation for the board.
        """
        if len(food) > self.max_food:
            return self.infinite_board(snake_bodies, food, board)
        for body in snake_bodies:
            if len(body) > self.max_snake_len:
                return self.infinite_board(snake_bodies, food, board)

        return self.jitted_board(snake_bodies, food, board)
