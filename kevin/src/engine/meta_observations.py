import jax.lax

from kevin.src.engine.board_updater import RotatingBoardUpdater
from kevin.src.engine.python_engine import PythonGameState, Snake
import jax.numpy as jnp


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
        self.all_safe_moves = {name: self.safe_moves(name) for name in self.alive_snakes}

    def __call__(self, pov_name: str):
        ordering = _snake_order(self.game, pov_name)
        turn = self.game.turn_num
        viewport = self.game.updater.viewport_size

        # Try to pack it all into a viewport
        obs = jnp.zeros([viewport, viewport], dtype=jnp.int16)

        # Dead snakes don't care about meta obs
        if pov_name not in self.alive_snakes:
            return obs

        # Try a known 11x11 four snake arrangement
        if self.game.player_count == 4 and viewport == 23:

            # For each snake we have 1 hp, 4 move safeties, and 1 turn num for a total of 6.
            # In a 23-length row, we can 4x each for a total of 24. Turn num isn't important, so we cut that down.
            # Then for four snakes, we can make it 5 tall for a total of 20.

            for i in ordering:
                name = "snake_{}".format(i)
                line = [turn] * 3 + [self.game.snakes[name].health] * 4
                for move in self.all_safe_moves[name]:
                    line += [move] * 4

                lines = jnp.array([line], dtype=jnp.int16)
                for j in range(5):
                    obs = jax.lax.dynamic_update_slice(obs, lines, (5 * i + j, 0))

            return obs

        else:
            raise ValueError

    def safe_moves(self, me_id: str):
        """Returns a list of safe moves [0,1,2] for a snake, with 0 if unsafe, 8 if  guaranteed safe."""

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
