import jax.numpy as jnp


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
    snakes = (({
        #  Initialize max_snakes snakes with 0 hp and not you.
        "health": 0,
        "you": 0,
    }),) * max_snakes

    # Place food on the board
    for morsel in d["board"]["food"]:
        board[morsel["x"], morsel["y"]] = 1

    # Place hazards
    for hazard in d["board"]["hazards"]:
        board[hazard["x"], hazard["y"]] = 2

    # Snakes
    for i, snake in enumerate(d["board"]["snakes"]):

        #  Snake metadata
        snakes[i]["health"] = snake["health"]
        if snake["id"] == d["you"]["id"]:
            snakes[i]["you"] = 1

        #  Place on board
        head = 2 * i + 3
        body = 2 * i + 4

        if snake[0]:
            coord = snake[0]
            board[coord["x"], coord["y"]] = head
        else:
            continue  # This snake is empty

        for coord in snake[1:]:
            board[coord["x"], coord["y"]] = body

    return {"snakes": snakes, "turn": d["turn"], "board": board}
