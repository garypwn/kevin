from __future__ import annotations

import copy
import random
import sys
from typing import Final

import jax
import jax.numpy as jnp
import jax.random as jrand

from kevin.src.engine.board_updater import RotatingBoardUpdater, BoardUpdater
from kevin.src.engine.snake_engine import GameState


class Snake:

    def __getitem__(self, item):
        return self.body[item]

    id: str
    health: int
    body: list[tuple[int, int]]
    death_location: tuple[int, int] | None = None


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

    food: list[tuple[int, int]]
    hazards: list[tuple[int, int]] = []

    #  Observations
    snake_boards: dict[str: jax.Array]
    food_board: jax.Array | None
    recent_eliminations = set()
    meta_factory: MetaObservationFactory
    snake_ate_food = []

    # Options
    save_replays = False
    replay_flag = False
    _internal_replay_flag = False

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

    def __init__(self, seed: int | None = None, updater=None, fresh=True):
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
            self.updater = RotatingBoardUpdater(self.width, self.height)
        else:
            self.updater = updater

        self.snake_boards = {}
        self.food_board = None

        if fresh:
            self._spawn_snakes_and_food()

        self.replay_flag = False
        self._internal_replay_flag = False
        self.branch_name = ""

    def full_copy(self):

        new = PythonGameState(self.rng_seed, self.updater, fresh=False)
        new.turn_num = self.turn_num
        new.single_player_mode = self.single_player_mode
        new.rng_key = self.rng_key.copy()
        new.snakes = copy.deepcopy(self.snakes)
        new.food = copy.deepcopy(self.food)
        new.hazards = copy.deepcopy(self.hazards)
        new.branch_name = copy.deepcopy(self.branch_name)
        new.replay_flag = False
        new._internal_replay_flag = False
        new.save_replays = self.save_replays
        new.meta_factory = copy.deepcopy(self.meta_factory)

        return new

    def update_board(self) -> dict[str: jax.Array]:
        """
        Update the board and snake list, as well as empty points
        """

        for name, snake in self.snakes.items():
            self.snake_boards[name] = self.updater.snake_sub_board(snake.body, self.snake_boards.get(name))

        self.food_board = self.updater.food_sub_board(self.food, self.food_board)

        # Get the metas
        self.meta_factory = MetaObservationFactory(self)

        return self.food_board  # this is just so we can block until ready in tests

    def _eliminated(self, snake_id: str) -> bool:
        r"""
        Check if a snake is eliminated. A snake is eliminated if it has 0 length.
        :param snake_id:
        :return: True if the snake is eliminated
        """
        return len(self.snakes[snake_id].body) < 1

    def winner(self) -> str | None:
        """
        Returns None if the game is not over or ended in a draw. Otherwise, return the name of the winner.
        """
        alive_snakes = list(filter(lambda s: not self._eliminated(s), [name for name, _ in self.snakes.items()]))
        if len(alive_snakes) != 1:
            return None
        else:
            return alive_snakes[0]

    def _move_snakes(self, actions: dict):
        r""" Helper for step() """

        # Two snakes can eat the same food. It only disappears after resolving.
        eaten_food: list[tuple[int, int]] = []
        self.snake_ate_food = []

        #  Apply move
        for name, snake in self.snakes.items():

            # Dead snakes don't move
            if len(snake.body) == 0:
                continue

            # Figure out heading
            target = self.updater.get_target(actions[name], snake.body)

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
                self.snake_ate_food.append(name)
                eaten_food.append(head)

        #  Check elimination conditions
        self.recent_eliminations = set()
        for name, snake in self.snakes.items():

            if len(snake.body) < 1:
                #  Snake already eliminated
                continue

            head = snake.body[0]
            x, y = head

            #  Check out of bounds
            if x < 0 or y < 0:
                self.recent_eliminations.add(name)

            if x >= self.width or y >= self.height:
                self.recent_eliminations.add(name)

            #  Check out of health
            if snake.health <= 0:
                self.recent_eliminations.add(name)

            #  Check for collision with snake body (or self)
            for id2, snake2 in self.snakes.items():
                if head in snake2.body[1:]:
                    self.recent_eliminations.add(name)

            #  Check for head-to-head collisions
            for name2, snake2 in self.snakes.items():

                #  Dead snakes can't be collided with
                if len(snake2.body) < 1:
                    continue

                if name2 == name:
                    continue

                if head == snake2[0]:
                    if len(snake.body) <= len(snake2.body):
                        self.recent_eliminations.add(name)

        #  Remove eaten food
        for morsel in eaten_food:
            if morsel in self.food:
                self.food.remove(morsel)

        #  Eliminate snakes
        for name in self.recent_eliminations:
            # Save the dead snake for later
            snake = self.snakes[name]
            snake.death_location = snake[0]
            snake.body = []
            snake.health = 0

        if len(self.recent_eliminations) > 0 and self.save_replays and self.turn_num > 12:
            self._internal_replay_flag = True

        roll = jrand.randint(self.rng_key, [1], 0, 100)[0]
        if self.save_replays and self.turn_num > 12 and roll > 95:
            self._internal_replay_flag = True

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

    def get_observation(self, snake_id: str) -> jax.Array:

        i = int(snake_id[6:])
        nums = [n for n in range(self.player_count)]
        ordering = nums[i:] + nums[:i]

        # Boards should be in order with our snake as number 0
        ordered_snake_boards = [self.snake_boards["snake_{}".format(ordering[n])] for n in nums]

        snake = self.snakes[snake_id]
        body = snake.body if snake.death_location is None else [snake.death_location]

        walls = self.updater.walls_pov(body)
        food = self.updater.snake_pov(body, self.food_board)
        povs = [self.updater.snake_pov(body, board)
                for board in ordered_snake_boards]

        meta = self.meta_factory(snake_id)

        boards = jnp.stack([povs[0], meta, walls, food] + povs[1:], 0)
        return boards

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
        return self.get_observation("snake_0")

    def get_reward(self, snake_id) -> float:
        r"""
        Returns 1. if the snake won, -1 if the snake lost, 0 otherwise.
        :param snake_id:
        :return:
        """

        scale = 10.
        live_snakes = len(list(filter(lambda s: not self._eliminated(s), [i for i, _ in self.snakes.items()])))
        live_snakes += len(self.recent_eliminations)
        live_snakes += 4

        # Penalty for being alive. Larger when there are more snakes.
        converging_neutral_reward = 7. / (self.turn_num + 2.)
        # survival_reward = 0.02
        survival_reward = -.0001 * live_snakes

        # Being long gives a small reward
        converging_length_reward = 0.25 * len(self.snakes[snake_id].body) * 0.98 ** self.turn_num
        length_reward = 0.003 * len(self.snakes[snake_id].body)
        length_reward = 0.0

        # Reward for winning is huge
        static_victory_reward = .3 + .3 * self.player_count  # 1.5 for a 4p game

        # Defeat penalty scales with number of live snakes (losing early bad)
        defeat_penalty = -.2 * live_snakes

        # Eating food gives a small static penalty (smaller than killing or winning)
        food_reward = .1 if snake_id in self.snake_ate_food else 0

        #  Check if last snake alive
        alive_snakes = list(filter(lambda s: not self._eliminated(s), [name for name, _ in self.snakes.items()]))

        # Add a reward for other snakes dying (hopefully translates to the urge to kill)
        if len(self.recent_eliminations) > 0 and snake_id not in self.recent_eliminations:
            kill_reward = 0.2 * len(self.recent_eliminations)
        else:
            kill_reward = 0

        if self._eliminated(snake_id):

            # Check for an end of game draw
            if len(alive_snakes) == 0:
                # If defeat is assured, the goal becomes kamikaze
                return defeat_penalty * scale * 0.75

                # return -4.  # Losing gives a static penalty
            return defeat_penalty * scale

        if snake_id in alive_snakes and len(alive_snakes) == 1:

            # If it's single player, return a neutral reward
            if self.single_player_mode:
                return (survival_reward + length_reward + food_reward) * scale

            # Multiplayer victory
            return (static_victory_reward + survival_reward + length_reward + food_reward) * scale

        return (survival_reward + length_reward + food_reward) * scale

    def step(self, actions, options: dict | None = None) -> PythonGameState:

        cpy = self.full_copy()
        cpy._move_snakes(actions)
        cpy.rng_key, subkey = jrand.split(cpy.rng_key)
        cpy._place_food(subkey)

        cpy.turn_num += 1

        self.replay_flag = False
        self._internal_replay_flag = False

        if cpy._internal_replay_flag or (options is not None and options.get("save")):
            cpy._internal_replay_flag = False
            self.replay_flag = True

        else:
            # If we aren't reusing this state, we can donate board buffers to the new one
            cpy.food_board = self.food_board
            cpy.snake_boards = self.snake_boards

        cpy.update_board()

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
        self.pending_moves = {}
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
        self.branch_name = ""
        self._internal_replay_flag = False
        self.replay_flag = False

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


def _snake_order(game: PythonGameState, perspective: str) -> list[int]:
    i = int(perspective[6:])
    nums = [n for n in range(game.player_count)]
    return nums[i:] + nums[:i]


def health_values(game: PythonGameState, perspective: str) -> list[int]:
    """
    Returns the ordered snake hp array, with the perspective snake at the start
    """
    ordering = _snake_order(game, perspective)
    return [game.snakes["snake_{}".format(i)].health for i in ordering]


def get_target_pts(game: PythonGameState, snake: Snake):
    if isinstance(game.updater, RotatingBoardUpdater):
        n = 3
    else:
        n = 4

    # In order target points for 0, 1, 2, 3
    return [game.updater.get_target(i, snake.body) for i in range(n)]


class MetaObservationFactory:
    """Creates meta observations from a game state."""

    possible_targets: dict[str: list[tuple[int, int]]]
    all_safe_moves: dict[str: list[int]]
    alive_snakes: set[str]

    def __init__(self, game: PythonGameState):
        self.game = game
        self.alive_snakes = set()
        for name, snake in game.snakes.items():
            if len(snake.body) > 0:
                self.alive_snakes.add(name)

        self.possible_targets = {name: get_target_pts(game, game.snakes[name]) for name in self.alive_snakes}
        self.all_safe_moves = {name: self.safe_moves(name) for name in self.game.snakes.keys()}

    def __call__(self, pov_name: str):
        ordering = _snake_order(self.game, pov_name)
        turn = self.game.turn_num
        viewport = self.game.updater.viewport_size

        # Try to pack it all into a viewport
        obs = jnp.zeros([viewport, viewport], dtype=jnp.int16)

        # Try a known 11x11 four snake arrangement
        if self.game.player_count == 4 and viewport == 23:

            # For each snake we have 1 hp, 4 move safeties, and 1 turn num for a total of 6.
            # In a 23-length row, we can 4x each for a total of 24. Turn num isn't important, so we cut that down.
            # Then for four snakes, we can make it 5 tall for a total of 20.

            for i, snake_num in enumerate(ordering):
                name = "snake_{}".format(snake_num)

                line = [turn] * 3

                line += [self.game.snakes[name].health] * 4
                for move in self.all_safe_moves[name]:
                    line += [move] * 4

                lines = jnp.array([line] * 5, dtype=jnp.int16)
                obs = jax.lax.dynamic_update_slice(obs, lines, (5 * i, 0))

            return obs

        else:
            raise ValueError

    def safe_moves(self, me_id: str):
        """Returns a list of safe moves [0,1,2] for a snake, with 0 if unsafe, 8 if  guaranteed safe."""

        if self.game.snakes[me_id].death_location is not None:
            # Dead snakes have no safe moves
            return [0, 0, 0, 0]

        targets = self.possible_targets[me_id]
        body_middles = [snake.body[0:-1] for _, snake in self.game.snakes.items()]

        safety_values = [8 for _ in targets]

        for i, target in enumerate(targets):

            # Check body collision
            for middle in body_middles:
                if target in middle:
                    safety_values[i] = 0

            # Check wall collision
            tx, ty = target
            if tx < 0 or tx >= self.game.width:
                safety_values[i] = 0

            if ty < 0 or ty >= self.game.height:
                safety_values[i] = 0

            # Check out of health death
            if self.game.snakes[me_id].health == 1 and target not in self.game.food:
                safety_values[i] = 0

            # Check possible head collisions
            for enemy, e_targets in self.possible_targets.items():
                if enemy == me_id:
                    continue

                if len(self.game.snakes[enemy].body) >= len(self.game.snakes[me_id].body):
                    if target in e_targets:
                        safety_values[i] = 0

        return safety_values
