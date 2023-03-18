import jax.numpy as jnp

import kevin.src.engine.utils as utils
from kevin.src.engine.python_engine import PythonGameState


def test_observation_from_empty():
    game = {
        "game": None,
        "turn": 0,
        "board": {
            "height": 11,
            "width": 11,
            "food": [],
            "hazards": [],
            "snakes": [],
        },
        "you": {
            "id": "snake_1"
        }
    }

    obs = utils.observation_from_dict(game, 4)

    assert obs["turn"] == 0
    assert len(obs["snakes"]) == 4

    for snake in obs["snakes"]:
        assert snake["health"] == 0
        assert snake["you"] == 0

    assert jnp.array_equal(obs["board"], jnp.zeros([11, 11], dtype=int))
    print(f"\n {obs['board']}")


def test_observation_from_example_board():
    game = {
        "game": None,
        "turn": 5,
        "board": {
            "height": 11,
            "width": 11,
            "food": [
                {"x": 1, "y": 1}, {"x": 1, "y": 2}
            ],
            "hazards": [
                {"x": 2, "y": 1}, {"x": 2, "y": 2}
            ],
            "snakes": [
                {
                    "id": "snake_0",
                    "name": "snake_0",
                    "health": 100,
                    "body": [{"x": 1, "y": 4}, {"x": 2, "y": 4}, {"x": 3, "y": 4}]
                },
                {
                    "id": "snake_1",
                    "name": "snake_1",
                    "health": 50,
                    "body": [{"x": 4, "y": 6}, {"x": 5, "y": 6}, {"x": 6, "y": 6}, {"x": 7, "y": 6}]
                }
            ],
        },
        "you": {
            "id": "snake_1",
            "name": "snake_1",
            "health": 50,
            "body": [{"x": 4, "y": 6}, {"x": 5, "y": 6}, {"x": 6, "y": 6}, {"x": 7, "y": 6}]
        }
    }

    obs = utils.observation_from_dict(game, 4)

    assert obs["turn"] == 5
    assert len(obs["snakes"]) == 4

    assert obs["snakes"][0]["health"] == 100
    assert obs["snakes"][1]["health"] == 50
    assert obs["snakes"][1]["you"] == 1

    for snake in obs["snakes"][2:]:
        assert snake["health"] == 0
        assert snake["you"] == 0

    #  Check there is only one "you"
    assert len([snake for snake in obs["snakes"] if snake["you"] == 1]) == 1

    print(f"\n {obs['board']}")


def test_fancy_print():
    game = PythonGameState()
    print(utils.fancy_board(game.food_board, game.snake_boards))
