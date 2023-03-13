import pytest

from kevin.src.engine.python_engine import PythonStandard4Player
import jax.numpy as jnp


def test_print_game():
    game = PythonStandard4Player(99)
    print(game)  # Requires visual inspection


@pytest.mark.parametrize("seed", range(0, 200000, 10007))
def test_spawn_determinism(seed: int):
    games = (PythonStandard4Player(seed), PythonStandard4Player(seed))
    print(games[0])
    assert jnp.array_equal(games[0].board, games[1].board)


@pytest.mark.parametrize("seed", range(50000, 700000, 50023))
def test_no_initial_food_in_corner(seed: int):
    game = PythonStandard4Player(seed)
    print(game)
    for i in range(4):
        assert jnp.rot90(game.board, k=i)[0, 0] == 0


@pytest.mark.parametrize("seed", range(800000, 1700000, 77023))
def test_count_initial_food_and_snakes(seed: int):
    game = PythonStandard4Player(seed)
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
def test_snake_heads_move(seed: int = 0):
    game = PythonStandard4Player(seed)
    print(game)
    for name, move in zip(game.snakes, [0, 3, 1, 2]):
        game.submit_move(name, move)

    game.step()
    print(game)

    values = [0 for i in range(3+2*game.num_players())]
    
    for row in game.board.tolist():
        for i in row:
            values[i] += 1
    
    for i in range(game.num_players()):
        # check there are 4 heads and 4 bodies
        assert values[2*i + 2] == game.num_players()
        assert values[2*i + 3] == game.num_players() 



@pytest.mark.parametrize("seed", range(0, 200000, 10013))
def test_observations_have_unique_perspective(seed: int = 0):
    game = PythonStandard4Player(seed)
    for id, snake in game.snakes.items():
        obs = game.get_observation(id)
        yous = 0
        for obs_snake in obs.snakes:
            if obs["you"] == 1
                yous += 1
        
        assert yous == 1


def test_same_turn_observations_have_same_board(seed: int = 0):
    game = PythonStandard4Player(seed)
    board = None
    for id, snake in game.snakes.items():
        obs = game.get_observation(id)
        if board is not None:
            assert array_equal(obs["board"], board)
        
        board = obs["board"]


def generate_empty_board() -> PythonStandard4Player:
    r"""A board with snake_0 at (5,5)"""
    game = PythonStandard4Player(0)
    for id, snake in game.snakes.items():
        snake.body = []

    game.num_players = 0
    game._update_board()
    return game

def add_snake(game: PythonStandard4Player, snake: Snake):
    r"""Adds a snake to the board. Doesn't check if spaces are occupied."""
    game.snakes[snake.id: snake]
    game.num_players += 1
    game._update_board()

def test_elimination_on_wall(seed: int = 0):
    
    game = generate_empty_board()

    #  Put a snake next to the wall
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(0,0), (1,0)]
    add_snake(game, new_snake)

    print(game)
    assert not game._eliminated("snake_0")

    game.submit_move("snake_0, 3")
    game.step()
    print(game)

    assert game._eliminated("snake_0")



def test_elimination_on_body_collision(seed: int = 0):
    game = generate_empty_board()

    #  Put a snake near the middle
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(5,5), (5,6), (6,6), (6,5), (6, 4)]
    add_snake(game, new_snake)

    #  todo maybe it's time to add tails to the observation?
    #  Think about it: there are states where you might know where the head is, But
    #  it isn't obvious where the tail is
    print(game)

    game.submit_move("snake_0", 1)
    game.step()

    print(game)
    assert len(game.snakes["snake_0"].body) == 0


def test_elimination_on_head_collision(seed: int = 0):
    game = generate_empty_board()

    #  Put a snake near the middle
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 100
    new_snake.body = [(5,5), (5,6), (5, 7)]
    add_snake(game, new_snake)

    #  Put another smaller snake nearby
    new_snake = Snake()
    new_snake.id = "snake_1"
    new_snake.health = 100
    new_snake.body = [(3,5), (3,6)]
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

def test_elimination_on_0_hp(seed: int = 0):
    game = generate_empty_board()

    #  Put a low hp snake on the board
    new_snake = Snake()
    new_snake.id = "snake_0"
    new_snake.health = 1
    new_snake.body = [(1,1)]
    add_snake(game, new_snake)

    print(game)

    #Move the snake, which should kill it
    game.submit_move("snake_0", 1)
    print(game)

    assert game.snakes["snake_0"].hp == 0
    assert game._eliminated("snake_0")


def test_reward_on_victory(seed: int = 0):
    #  todo
    pass


def test_reward_on_defeat(seed: int = 0):
    #  todo
    pass


def test_reward_on_neutral(seed: int = 0):
    #  todo
    pass


def food_grows_snakes(seed: int = 0)
    #  todo
    pass