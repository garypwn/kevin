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
    seed: int

    snakes: dict[str: Snake]

    #  Coords are in the shape (x, y)
    food: list[tuple[int, int]]
    hazards: list[tuple[int, int]] = []

    #  For presenting in observations. Updated every step.
    snakes_array: list[dict[str: int]]  # For observations
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
        point: tuple[int, int]
        while True:
            point = jrand.uniform(self._random(), shape=[2], dtype=int, minval=[0, 0],
                                  maxval=(self.width(), self.height()))
            if not self._is_occupied(point):
                break
        return point

    def __init__(self):

        #  Initialize with a random seed
        self.seed(random.randrange(sys.maxsize))
        self.reset()

    def player_count(self) -> int:
        return 4

    def height(self) -> int:
        return 11

    def width(self) -> int:
        return 11

    def _update_board(self):
        """
        Update the board and snake list, as well as empty points
        """

        #  Empty the board
        self.board = jnp.zeros([self.width(), self.height()], dtype=int)

        for x, y in self.food:
            self.board[x, y] = 1

        for x, y in self.hazards:
            self.board[x, y] = 2

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
                self.board[x, y] = head

            for x, y in snake.body[1:]:
                self.board[x, y] = body

    def _elminiated(self, snake_id: str) -> bool:
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
            head = snake.body[0]
            match move:
                case 0: #  Up
                    return (x, y+1 for x, y in head)

                case 1: # right
                    return (x+1, y for x, y in head)

                case 2: # down
                    return (x, y-1 for x, y in head)

                case 3: # left
                    return (x-1, y for x, y in head)


        eaten_food: list[tuple[int, int]] #  Two snakes can eat the same food. It only disappears after resolving.

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
                snake.body.append(snake.body[len(snake.body)-1])
                eaten_food.append(head)

        #  Check elimination conditions
        eliminated: set = {}
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

        #  Eliminate snakes
        for id in eliminated:
            self.snakes[id].body = []

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
        return self._elminiated(snake_id)

    def get_truncated(self, snake_id) -> bool:
        r"""
        Always returns false because this environment has no time limit.
        :param snake_id: ignored
        :return: False
        """
        return False

    def global_observation(self) -> dict:
        return {"snakes": self.snakes_array, "turn": self.turn_num, "board": self.board}

    def submit_move(self, snake_id, move: int) -> None:
        self.pending_moves[snake_id] = move

    def step(self) -> None:
        pass

    def reset(self) -> None:

        #  Reset all state
        self.turn_num = 0
        self.snakes = {}
        self.pending_moves = {}
        for i in range(self.player_count()):
            name = "snake_" + i
            self.snakes[name] = Snake()
            self.snakes[name].id = name
            self.snakes[name].health = 100
            self.snakes[name].body = []

            self.snakes_array[i] = {
                "health": 100,
                "you": 0
            }

            self.pending_moves[name] = 0

        self.food = []
        self.hazards = []
        self.board = jnp.zeros([self.width(), self.height()], dtype=int)

        self.seed(self.seed) # Reset the prng

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

        points = corners.append(cardinals)

        # todo add support for more snakes. Currently goes up to 8.
        if self.player_count() > 8:
            raise NotImplementedError("Only supports up to 8 players.")

        for i, (_, snake) in enumerate(self.snakes):
            snake.body.append(points[i])

        #  Place food randomly, one per snake. todo Is this how BS actually behaves?
        for i in range(self.player_count()):
            point = self._random_unoccupied_pt()
            self.food.append(point)

        self._update_board()

    def seed(self, seed) -> None:
        self.seed = seed
        self.rng_key = jrand.PRNGKey(seed)
