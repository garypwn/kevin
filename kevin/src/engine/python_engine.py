from __future__ import annotations

import copy
import random
import sys
from typing import Final

import jax
import jax.numpy as jnp
import jax.random as jrand
import numpy as np

from kevin.src.engine.board_updater import BoardUpdater
from kevin.src.engine.snake_engine import GameState


class Snake:

    def __getitem__(self, item):
        return self.body[item]

    id: str
    health: int
    body: list[tuple[int, int]]


class PythonGameState(GameState):
    """ Attempts to emulate 4 player standard battlesnake logic in python
    """

    #  Game options
    player_count: Final[int] = 4  # todo support more than 4 players
    height: Final[int] = 11
    width: Final[int] = 11

    #  Board updater fn
    updater: BoardUpdater

    #  State
    turn_num: int = 0

    rng_key: jax.Array | jrand.PRNGKeyArray
    rng_seed: int

    snakes: dict[str: Snake]
    dead_snakes: dict[str: Snake]

    food: list[tuple[int, int]]
    hazards: list[tuple[int, int]] = []

    #  Observations
    snake_boards: dict[str: jax.Array]
    food_board: jax.Array | None

    def _is_occupied(self, pt: tuple[int, int]) -> bool:
        """
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

    def _random_unoccupied_pt(self, prng):
        r"""
        Places food in a random unoccupied location. Deterministic based on prng key.
        :return:
        """
        x: int
        y: int
        prng = jrand.fold_in(prng, 12345)
        while True:
            prng, subkey = jrand.split(prng)
            point = jrand.randint(subkey, shape=[2],
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
        self.seed(seed)

        #  Set up the updater. This allows us to pre-jit the updater and use it in all games.
        if updater is None:
            self.updater = BoardUpdater(self.width, self.height)
        else:
            self.updater = updater

        self.snake_boards = {}
        self.food_board = None

        self._spawn_snakes_and_food()

    def full_copy(self):

        new = PythonGameState(self.rng_seed, self.updater)
        new.turn_num = self.turn_num
        new.rng_key = self.rng_key.copy()
        new.snakes = copy.deepcopy(self.snakes)
        new.dead_snakes = copy.deepcopy(self.dead_snakes)
        new.food = copy.deepcopy(self.food)
        new.hazards = copy.deepcopy(self.hazards)

        return new

    def update_board(self) -> dict[str: jax.Array]:
        """
        Update the board and snake list, as well as empty points
        """

        for name, snake in self.snakes.items():
            num = int(name[6:])
            self.snakes_array[num] = snake.health

        for name, snake in self.snakes.items():
            self.snake_boards[name] = self.updater.snake_sub_board(snake.body, self.snake_boards.get(name))

        self.food_board = self.updater.food_sub_board(self.food, self.food_board)

        return self.food_board  # this is just so we can block until ready in tests

    def _eliminated(self, snake_id: str) -> bool:
        r"""
        Check if a snake is eliminated. A snake is eliminated if it has 0 length.
        :param snake_id:
        :return: True if the snake is eliminated
        """
        return len(self.snakes[snake_id].body) < 1

    def _move_snakes(self, actions: dict):
        r""" Helper for step() """

        # Two snakes can eat the same food. It only disappears after resolving.
        eaten_food: list[tuple[int, int]] = []

        #  Apply move
        for name, snake in self.snakes.items():

            # Dead snakes don't move
            if len(snake.body) == 0:
                continue

            # Figure out heading
            old_head = snake.body[0]
            x0, y0 = old_head
            hx, hy = (0, 1)  # Heading
            if snake.body[0] != snake.body[1]:
                x1, y1 = snake.body[1]
                hx, hy = (x0 - x1, y0 - y1)

            # Compute next move
            match actions[name]:
                case 1:  # Forward
                    target = (x0 + hx, y0 + hy)
                case 0:  # Left
                    target = (x0 - hy, y0 + hx)
                case 2:  # Right
                    target = (x0 + hy, y0 - hx)
                case _:
                    raise ValueError

            #  Move the head and tail

            snake.body.insert(0, target)
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

            # Save the dead snake for later
            cpy = copy.deepcopy(self.snakes[name])
            self.dead_snakes[name] = cpy
            self.snakes[name].body = []

    def _place_food(self, rng):
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
            self.food.append(self._random_unoccupied_pt(rng))
            return

        roll = jrand.randint(rng, [1], minval=0, maxval=100)
        if roll[0] < food_chance:
            self.food.append(self._random_unoccupied_pt(rng))

    def get_observation(self, snake_id: str) -> dict:

        i = int(snake_id[6:])
        ordered_snakes = self.snakes_array[i:] + self.snakes_array[:i]
        nums = [n for n in range(self.player_count)]
        ordering = nums[i:] + nums[:i]

        # Boards should be in order with our snake as number 0
        ordered_snake_boards = [self.snake_boards["snake_{}".format(ordering[n])] for n in nums]

        if self._eliminated(snake_id):
            snake = self.dead_snakes[snake_id]
        else:
            snake = self.snakes[snake_id]

        head = snake[0]
        x0, y0 = head
        x1, y1 = snake[1]
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

        walls = self.updater.walls_pov(head, d)
        food = self.updater.snake_pov(head, d, self.food_board)
        povs = [self.updater.snake_pov(head, d, board)
                for board in ordered_snake_boards]

        boards = jnp.stack([povs[0], walls, food] + povs[1:], 0)
        return {"turn": self.turn_num, "snakes": jnp.array(ordered_snakes), "boards": boards}

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

        # Being long gives a small reward
        length_reward = 0.2 * len(self.snakes[snake_id].body)

        # Neutral reward is based on surviving. Falls off late game.
        neutral_reward = 1. + 15. * 1.1 ** (- self.turn_num) + length_reward

        # Reward is usually 200, except early game.
        victory_reward = 250. - 200. * 1.05 ** (- self.turn_num)

        # Defeat gives a static penalty
        defeat_penalty = -100.

        if self._eliminated(snake_id):
            # return -4.  # Losing gives a static penalty
            return defeat_penalty

        #  Check if last snake alive
        alive_snakes = list(filter(lambda s: not self._eliminated(s), [name for name, _ in self.snakes.items()]))
        if snake_id in alive_snakes and len(alive_snakes) == 1:

            # If it's single player, return a neutral reward
            if self.single_player_mode:
                return neutral_reward

            return neutral_reward  # + victory_reward

        return neutral_reward

    def step(self, actions, options: dict | None = None) -> PythonGameState:
        if options is not None and options.get("save"):
            cpy = self.full_copy()
        else:
            cpy = self
        cpy._move_snakes(actions)
        cpy.rng_key, subkey = jrand.split(cpy.rng_key)
        cpy._place_food(subkey)
        cpy.update_board()
        cpy.turn_num += 1
        return cpy

    def reset(self, options: dict | None = None) -> PythonGameState:
        if options is not None and options.get("save"):
            cpy = self.full_copy()
        else:
            cpy = self

        cpy._spawn_snakes_and_food(options)
        return cpy

    def _spawn_snakes_and_food(self, options: dict | None = None) -> None:

        single_player = False

        if options is not None:
            single_player = options["single_player"]

        self.seed(self.rng_seed)  # Reset the prng

        #  Reset all state
        self.turn_num = 0
        self.snakes = {}
        self.dead_snakes = {}
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
        self.rng_key, subkey_1, subkey_2, subkey_3 = jrand.split(self.rng_key, 4)
        order = jrand.permutation(subkey_1, jnp.array([0, 1, 2, 3]))
        corners = [corners[i] for i in order]
        order = jrand.permutation(subkey_2, jnp.array([0, 1, 2, 3]))
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
            subkey_3, subkey_4 = jrand.split(subkey_3)
            order = jrand.permutation(subkey_4, jnp.array(permute))
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
