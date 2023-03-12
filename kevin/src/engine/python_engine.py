import random
import sys

import jax.random as jrand
import jax.numpy as jnp
import jax

from kevin.src.engine.snake_engine import SnakeEngine


class Snake:
    id: str
    health: int
    body: list[tuple[int, int]]


class PythonStandard4Player(SnakeEngine):
    r""" Attempts to emulate 4 player standard battlesnake logic in python
    """

    turn_num: int

    rng_key: jax.Array | jrand.PRNGKeyArray
    rng_seed: int

    snakes: dict[str: Snake]

    #  Coords are in the shape (x, y)
    food: list[tuple[int, int]]
    hazards: list[tuple[int, int]] = []

    #  For presenting in observations. Updated every step.
    snakes_array: list[dict[str: int]] = []  # For observations
    board: jax.Array

    #  Submitted moves
    pending_moves: dict[str: int]

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

        for _, snake in self.snakes:
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
            point = jrand.uniform(self._random(), shape=[2], dtype=int,
                                  minval=jnp.array([0, 0]),
                                  maxval=jnp.array([self.width(), self.height()]))
            x, y = point.at[0], point.at[1]
            if not self._is_occupied((x, y)):
                break
        return x, y

    def __init__(self):
        r"""
        Initialize the game with a random seed
        """

        #  Initialize with a random seed
        self.seed(random.randrange(sys.maxsize))
        self.reset()

    def player_count(self) -> int:
        return 4

    def height(self) -> int:
        return 11

    def width(self) -> int:
        return 11

    def _update_board(self) -> jax.Array:
        """
        Update the board and snake list, as well as empty points
        """

        #  Empty the board
        board = jnp.zeros([self.width(), self.height()], dtype=int)

        for x, y in self.food:
            board = board.at[x, y].set(1)

        for x, y in self.hazards:
            board = board.at[x, y].set(2)

        for name, snake in self.snakes:

            #  Snake names are always of the form snake_i
            num = int(name[6:])
            head = 2 * num + 3
            body = 2 * num + 4

            self.snakes_array[num] = {
                "health": snake.health,
                "you": 0,  # This changes if observed from a perspective
            }

            if len(snake.body) == 0:
                #  This snake is dead
                continue

            for x, y in snake.body[:1]:
                board = board.at[x, y].set(head)

            for x, y in snake.body[1:]:
                board = board.at[x, y].set(body)

        return board

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
        def compute_next(snake, move):
            h = snake.body[0]
            match move:
                case 0:  # Up
                    return ((x, y + 1) for x, y in h)

                case 1:  # right
                    return ((x + 1, y) for x, y in h)

                case 2:  # down
                    return ((x, y - 1) for x, y in h)

                case 3:  # left
                    return ((x - 1, y) for x, y in h)

        # Two snakes can eat the same food. It only disappears after resolving.
        eaten_food: list[tuple[int, int]] = []

        #  Apply move
        for id, snake in self.snakes:

            #  Move the head and tail
            snake.body.insert(0, compute_next(snake, self.pending_moves[id]))
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
        for id, snake in self.snakes:

            head = snake.body[0]
            x, y = head

            #  Check out of bounds
            if x < 0 or y < 0:
                eliminated.add(id)

            if x >= self.width() or y >= self.height():
                eliminated.add(id)

            #  Check out of health
            if snake.health <= 0:
                eliminated.add(id)

            #  Check for collision with snake body (or self)
            for id2, snake2 in self.snakes:
                if head in snake2.body[1:]:
                    eliminated.add(id)

            #  Check for head-to-head collisions
            for id2, snake2 in self.snakes:
                if id2 == id:
                    continue

                if head == snake2[0]:
                    if len(snake.body) <= len(snake2.body):
                        eliminated.add(id)

        #  Remove eaten food
        for morsel in eaten_food:
            if morsel in self.food:
                self.food.remove(morsel)

        #  Eliminate snakes
        for id in eliminated:
            self.snakes[id].body = []

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

        roll = jrand.uniform(self._random(), dtype=int, minval=0, maxval=1)
        if roll.at[0] < food_chance:
            self.food.append(self._random_unoccupied_pt())

    def get_observation(self, snake_id: str) -> dict:
        num = int(snake_id[6:])

        #  Modify snake array so that this snake is "you"
        snake_copy = self.snakes_array.copy()
        entry_copy = snake_copy[num].copy()
        entry_copy["you"] = 1
        snake_copy[num] = entry_copy

        return {"snakes": snake_copy, "turn": self.turn_num, "board": self.board}

    def get_terminated(self, snake_id) -> bool:

        #  A snake is dead if its body has size 0
        return self._eliminated(snake_id)

    def get_truncated(self, snake_id) -> bool:
        r"""
        Always returns false because this environment has no time limit.
        :param snake_id: ignored
        :return: False
        """
        return False

    def global_observation(self) -> dict:
        return {"snakes": self.snakes_array, "turn": self.turn_num, "board": self.board}

    def get_reward(self, snake_id) -> float:
        r"""
        Returns 1. if the snake won, -1 if the snake lost, 0 otherwise.
        :param snake_id:
        :return:
        """
        if self._eliminated(snake_id):
            return -1.

        #  Check if last snake alive
        alive_snakes = filter(lambda id: not self._eliminated(id), [id for id, _ in self.snakes])
        if snake_id in alive_snakes and len(tuple(alive_snakes)) == 1:
            return 1.

        return 0.

    def submit_move(self, snake_id, move: int) -> None:
        self.pending_moves[snake_id] = move

    def step(self) -> None:
        self._move_snakes()
        self._place_food()
        self.board = self._update_board()

    def reset(self) -> None:

        self.seed(self.rng_seed)  # Reset the prng

        #  Reset all state
        self.turn_num = 0
        self.snakes = {}
        self.pending_moves = {}
        self.snakes_array = [{
                "health": 100,
                "you": 0
            }] * self.player_count()
        for i in range(self.player_count()):
            name = "snake_" + str(i)
            self.snakes[name] = Snake()
            self.snakes[name].id = name
            self.snakes[name].health = 100
            self.snakes[name].body = []
            self.pending_moves[name] = 0

        self.food = []
        self.hazards = []
        self.board = jnp.zeros([self.width(), self.height()], dtype=int)

        #  Place snakes following standard BS conventions of cards -> intercards.
        #  BS normally uses a distribution algorithm for num players > 8. That's a todo.
        xn, xd, xx = 1, (self.width() - 1) / 2, self.width() - 2
        yn, yd, yx = 1, (self.height() - 1) / 2, self.height() - 2

        corners = [(xn, yn), (xn, yx), (xx, yn), (xx, yx)]
        cardinals = [(xn, yd), (xd, yn), (xd, yx), (xx, yd)]

        #  Shuffle
        order = jrand.permutation(self._random(), jnp.array([0, 1, 2, 3]))
        corners = [corners[i] for i in order]
        order = jrand.permutation(self._random(), jnp.array([0, 1, 2, 3]))
        cardinals = [cardinals[i] for i in order]

        points = corners + cardinals

        # todo add support for more snakes. Currently goes up to 8.
        if self.player_count() > 8:
            raise NotImplementedError("Only supports up to 8 players.")

        for i, (_, snake) in enumerate(self.snakes):
            snake.body += [points[i]] * 5  # Starting length is 5

        #  Place starting food. BS default behaviour is to place a food intercardinal to each snake. Plus one center.
        #  But, do not place food in a corner or adjacent to the center square. todo except on small boards.
        cx, cy = int((self.width() - 1) / 2), int((self.height() - 1) / 2)
        for _, snake in self.snakes:
            x, y = snake.body[0]
            tentative = [
                (x + 1, y + 1),
                (x + 1, y - 1),
                (x - 1), (y + 1),
                (x - 1), (y - 1),
            ]

            for x, y in tentative:
                if (x, y) in corners:
                    tentative.remove((x, y))
                    continue

                if self._is_occupied((x, y)):
                    tentative.remove((x, y))
                    continue

                if abs(x - cx) + abs(y - cy) <= 2:  # Manhattan distance from center
                    tentative.remove((x, y))

            order = jrand.permutation(self._random(), jnp.array([0, 1, 2, 3]))
            tentative = [tentative[i] for i in order]

            if len(tentative) != 0:
                self.food.append(tentative[0])

        # Place food in center
        if not self._is_occupied((cx, cy)):
            self.food.append((cx, cy))

        self.board = self._update_board()

    def seed(self, seed) -> None:
        self.rng_seed = seed
        self.rng_key = jrand.PRNGKey(seed)
