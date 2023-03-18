import jax
import jax.numpy as jnp
import numpy as np

from kevin.src.engine.python_engine import PythonGameState
from kevin.src.engine.snake_engine import GameState


def observation_from_dict(d: dict, max_snakes=4) -> dict:
    r"""
    Maps a game state in the dictionary representation provided by the battlesnake engine
    to an observation of the form specified by MultiSnakeEnv.

    :param max_snakes: The maximum number of snakes that can exist on this game board.
    Currently, the model supports 4 snakes.
    :param d: A game state object
    :return: An observation
    """

    width = d["board"]["width"]
    height = d["board"]["height"]

    board = jnp.zeros([width, height], dtype=int)  # Initialize the board with all empty spaces
    snakes = [{
        #  Initialize max_snakes snakes with 0 hp and not you.
        "health": 0,
        "you": 0,
    } for _ in range(max_snakes)]

    # Place food on the board
    for morsel in d["board"]["food"]:
        board = board.at[morsel["x"], morsel["y"]].set(1)

    # Place hazards
    for hazard in d["board"]["hazards"]:
        board = board.at[hazard["x"], hazard["y"]].set(2)

    # Snakes
    for i, snake in enumerate(d["board"]["snakes"]):

        #  Snake metadata
        snakes[i]["health"] = snake["health"]
        if snake["id"] == d["you"]["id"]:
            snakes[i]["you"] = 1

        #  Place on board
        head = 2 * i + 3
        body = 2 * i + 4

        if snake["body"][0]:
            coord = snake["body"][0]
            board = board.at[coord["x"], coord["y"]].set(head)
        else:
            continue  # This snake is empty

        for coord in snake["body"][1:]:
            board = board.at[coord["x"], coord["y"]].set(body)

    return {"snakes": snakes, "turn": d["turn"], "board": board}


def fancy_board_from_game(game: GameState):
    if isinstance(game, PythonGameState):
        return fancy_board(game.food_board, game.snake_boards)
    else:
        raise NotImplementedError


def fancy_board(food_board: jax.Array, snake_boards: dict[str: jax.Array]):
    board = np.full(food_board.shape, "  ")

    symbols = {
        "snake_0": ["\N{Large yellow square}", "\N{Large yellow circle}"],
        "snake_1": ["\N{Large blue square}", "\N{Large blue circle}"],
        "snake_2": ["\N{Large orange square}", "\N{Large orange circle}"],
        "snake_3": ["\N{Large purple square}", "\N{Large purple circle}"],
        "food": ["\N{Cupcake}"]
    }

    # Food
    for i, row in enumerate(food_board):
        for j, c in enumerate(row):
            if c == 1:
                board[i, j] = symbols["food"][0]

    # Snakes
    for snake, snake_board in snake_boards.items():
        m = jnp.amax(snake_board)
        if m == 0:
            continue
        for i, row in enumerate(snake_board):
            for j, c in enumerate(row):
                if c != 0:
                    board[i, j] = symbols[snake][1]
                if c == m:
                    board[i, j] = symbols[snake][0]

    return board
