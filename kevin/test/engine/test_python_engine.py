from kevin.src.engine.python_engine import PythonStandard4Player
import jax.numpy as jnp


def board_1() -> PythonStandard4Player:
    game = PythonStandard4Player()
    game.seed(0)
    return game


def test_new_game():
    game = board_1()
    print(game.global_observation()["snakes"])
    print(jnp.rot90(game.global_observation()["board"]))
