import pytest

from kevin.src.engine.python_engine import PythonStandard4Player, Snake
import jax.numpy as jnp


def create_game(seed):
    game = PythonStandard4Player(seed)
    game.reset()
    return game


def test_print_game():
    game = create_game(99)
    print(game)  # Requires visual inspection


@pytest.mark.parametrize("seed", range(0, 200000, 10007))
def test_spawn_determinism(seed: int):
    games = (create_game(seed), create_game(seed))
    print(games[0])
    assert jnp.array_equal(games[0].board, games[1].board)


@pytest.mark.parametrize("seed", range(50000, 700000, 50023))
def test_no_initial_food_in_corner(seed: int):
    game = create_game(seed)
    print(game)
    for i in range(4):
        assert jnp.rot90(game.board, k=i)[0, 0] == 0


@pytest.mark.parametrize("seed", range(800000, 1700000, 77023))
def test_count_initial_food_and_snakes(seed: int):
    game = create_game(seed)
    print(game)
    food_count = 0
    snake_count = 0
    for row in game.board.tolist():
        for i in row:
            if i == 1:
                food_count += 1
            if i != 0 and i != 1:
                snake_count += 1

    assert food_count == 5
    assert snake_count == 4


@pytest.mark.parametrize("seed", range(0, 200000, 10013))
def test_snake_heads_move(seed: int):
    game = create_game(seed)
    print(game)
    for name, move in zip(game.snakes, [0, 3, 1, 2]):
        game.submit_move(name, move)

    game.step()
    print(game)

    values = [0 for _ in range(5 + 3 * game.player_count)]

    for row in game.board.tolist():
        for i in row:
            values[i] += 1

    for i in range(game.player_count):
        # check there is 1 head and 1 body for each snake
        assert values[3 * i + 3] == 1
        assert values[3 * i + 4] == 1


@pytest.mark.parametrize("seed", range(0, 200000, 10013))
def test_observations_have_unique_perspective(seed: int):
    game = create_game(seed)
    for id, snake in game.snakes.items():
        obs = game.get_observation(id)
        yous = 0
        for obs_snake in obs["snakes"]:
            if obs_snake["you"] == 1:
                yous += 1

        assert yous == 1


@pytest.mark.parametrize("seed", range(0, 200000, 10013))
def test_same_turn_observations_have_same_board(seed: int):
    game = create_game(seed)
    board = None
    print(game)
    for id, snake in game.snakes.items():
        obs = game.get_observation(id)
        if board is not None:
            assert jnp.array_equal(obs["board"], board)

        board = obs["board"]


def generate_empty_board(seed: int = 0) -> PythonStandard4Player:
    r"""A board with snake_0 at (5,5)"""
    game = create_game(seed)
    game.food = []
    for _, snake in game.snakes.items():
        snake.body = []

    game.board = game.update_board()
    return game


def add_snake(game: PythonStandard4Player, snake: Snake):
    r"""Adds a snake to the board. Doesn't check if spaces are occupied."""
    game.snakes[snake.id] = snake
    game.board = game.update_board()


def add_food(game: PythonStandard4Player, pts: list[(int, int)]):
    r"""Adds food to the board. Doesn't check if spaces are occupied."""
    game.food = game.food + pts
    game.board = game.update_board()


def test_elimination_on_wall():
    game = generate_empty_board()

    #  Put a snake next to the wall
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(0, 0), (1, 0)]
    add_snake(game, new_snake)

    print(game)
    assert not game._eliminated("snake_0")

    game.submit_move("snake_0", 3)
    game.step()
    print(game)

    assert game._eliminated("snake_0")


def test_elimination_on_single_body_collision():
    game = generate_empty_board()

    #  Put a snake near the middle
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(5, 5), (5, 6), (6, 6), (6, 5), (6, 4)]
    add_snake(game, new_snake)

    print(game)

    game.submit_move("snake_0", 1)
    game.step()

    print(game)
    assert len(game.snakes["snake_0"].body) == 0


def test_elimination_on_double_body_collision():
    game = generate_empty_board()

    #  Put a snake near the middle
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(5, 5), (5, 6), (5, 7)]
    add_snake(game, new_snake)

    #  Put another snake directly next to it
    new_snake = Snake()
    new_snake.id = "snake_1"
    new_snake.health = 100
    new_snake.body = [(4, 5), (4, 6)]
    add_snake(game, new_snake)

    print(game)

    #  Move snakes toward each other. They should both body collide
    game.submit_move("snake_0", 3)
    game.submit_move("snake_1", 1)
    game.step()
    print(game)

    #  Check that both snakes are defeated
    assert game._eliminated("snake_0")
    assert game._eliminated("snake_1")


def test_elimination_on_head_collision_different_sizes(seed: int = 0):
    game = generate_empty_board()

    #  Put a snake near the middle
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(5, 5), (5, 6), (5, 7)]
    add_snake(game, new_snake)

    #  Put another smaller snake nearby
    new_snake = Snake()
    new_snake.id = "snake_1"
    new_snake.health = 100
    new_snake.body = [(3, 5), (3, 6)]
    add_snake(game, new_snake)

    print(game)

    #  Move both snakes to (4,5)
    game.submit_move("snake_0", 3)
    game.submit_move("snake_1", 1)
    game.step()
    print(game)

    #  Check that only snake_1 is dead
    assert len(game.snakes["snake_0"].body) == 3
    assert game._eliminated("snake_1")


def test_elimination_on_head_collision_same_sizes(seed: int = 0):
    game = generate_empty_board()

    #  Put a snake near the middle
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(5, 5), (5, 6)]
    add_snake(game, new_snake)

    #  Put another smaller snake nearby
    new_snake = Snake()
    new_snake.id = "snake_1"
    new_snake.health = 100
    new_snake.body = [(3, 5), (3, 6)]
    add_snake(game, new_snake)

    print(game)

    #  Move both snakes to (4,5)
    game.submit_move("snake_0", 3)
    game.submit_move("snake_1", 1)
    game.step()
    print(game)

    #  Check that both snakes are dead
    assert game._eliminated("snake_0")
    assert game._eliminated("snake_1")


def test_elimination_on_0_hp(seed: int = 0):
    game = generate_empty_board()

    #  Put a low hp snake on the board
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 1
    new_snake.body = [(1, 1), (1, 2)]
    add_snake(game, new_snake)

    print(game)

    # Move the snake, which should kill it
    game.submit_move("snake_0", 1)
    game.step()
    print(game)

    assert game.snakes["snake_0"].health == 0
    assert game._eliminated("snake_0")


def test_reward_on_victory(seed: int = 0):
    game = generate_empty_board()

    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(1, 1), (1, 2)]
    add_snake(game, new_snake)

    print(game)

    reward = game.get_reward("snake_0")
    assert reward == 1.0


def test_reward_on_defeat(seed: int = 0):
    game = generate_empty_board()

    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(1, 1), (1, 2)]
    add_snake(game, new_snake)

    print(game)

    reward = game.get_reward("snake_1")  # Snake 1 is not alive
    assert reward == -1.0


def test_reward_on_neutral(seed: int = 0):
    game = generate_empty_board()

    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(1, 1), (1, 2)]
    add_snake(game, new_snake)

    new_snake = Snake()
    new_snake.id = "snake_1"
    new_snake.health = 100
    new_snake.body = [(2, 2), (3, 2)]
    add_snake(game, new_snake)

    print(game)

    assert game.get_reward("snake_1") == 0
    assert game.get_reward("snake_0") == 0


def test_food_grows_snakes(seed: int = 0):
    game = generate_empty_board()

    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(5, 5), (5, 6)]
    add_snake(game, new_snake)

    add_food(game, [(5, 4)])

    print(game)

    game.submit_move("snake_0", 2)
    game.step()
    print(game)

    assert len(game.snakes["snake_0"].body) == 3


@pytest.mark.parametrize("seed", range(0, 50000, 10013))
def test_food_spawn_determinism(seed: int):

    # two empty boards with the same seed
    games = generate_empty_board(seed), generate_empty_board(seed)
    for i in range(50):
        for game in games:
            game.step()

        print(games[0])
        print(games[1])
        assert games[0].food == games[1].food
