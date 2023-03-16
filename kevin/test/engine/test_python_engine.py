import timeit

import jax
import pytest

from kevin.src.engine.python_engine import PythonStandard4Player, Snake, BoardUpdater
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
    assert jnp.array_equal(games[0].boards["snake_0"], games[1].boards["snake_0"])


@pytest.mark.parametrize("seed", range(50000, 700000, 50023))
def test_no_initial_food_in_corner(seed: int):
    game = create_game(seed)
    print(game)
    for i in range(4):
        assert jnp.rot90(game.boards["snake_0"], k=i)[0, 0] == 0


@pytest.mark.parametrize("seed", range(800000, 1000000, 77023))
def test_count_initial_food_and_snakes(seed: int):
    game = create_game(seed)
    print(game)
    food_count = 0
    snake_count = 0
    for row in game.boards["snake_0"].tolist():
        for i in row:
            if i == 1:
                food_count += 1
            if i != 0 and i != 1:
                snake_count += 1

    assert food_count == 5
    assert snake_count == 4


@pytest.mark.parametrize("seed", range(800000, 1000000, 77023))
def test_count_initial_food_and_snakes_1player(seed: int):
    game = create_game(seed)
    game.reset({"single_player": True})
    print(game)
    food_count = 0
    snake_count = 0
    for row in game.boards["snake_0"].tolist():
        for i in row:
            if i == 1:
                food_count += 1
            if i != 0 and i != 1:
                snake_count += 1

    assert food_count == 5
    assert snake_count == 1


@pytest.mark.parametrize("seed", range(0, 200000, 20013))
def test_snake_heads_move(seed: int):
    game = create_game(seed)
    print(game)
    for name, move in zip(game.snakes, [0, 3, 1, 2]):
        game.submit_move(name, move)

    game.step()
    print(game)

    values = [0 for _ in range(5 + 3 * game.player_count)]

    for row in game.boards["snake_0"].tolist():
        for i in row:
            values[i] += 1

    for i in range(game.player_count):
        # check there is 1 head and 1 body for each snake
        assert values[3 * i + 3] == 1
        assert values[3 * i + 4] == 1


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
    board = game.update_board()["snake_0"].block_until_ready()


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
    print(reward)
    assert reward < 0


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
    r1, r2 = game.get_reward("snake_0"), game.get_reward("snake_1")
    print("Rewards: {}, {}".format(r1, r2))

    assert r1 > 0
    assert r2 > 0


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


def test_jaxpr_board_fn():
    game = create_game(0)
    food = game.food
    snakes = list([snake.body for _, snake in game.snakes.items()])
    updater = BoardUpdater(11, 11, 4)
    print(jax.make_jaxpr(updater.finite_board)(snakes, food))


def test_board_fn_correctness():
    game = generate_empty_board()

    #  Spawn food
    add_food(game, [(8, 8), (9, 9), (10, 10)])

    # Add snakes
    for i in range(4):
        snake = Snake()
        snake.id = "snake_{}".format(i)
        snake.health = 100
        snake.body = [(3, i), (4, i), (5, i), (6, i)]
        add_snake(game, snake)

    food = game.food
    snakes = list([snake.body for _, snake in game.snakes.items()])
    updater = BoardUpdater(11, 11, 4)

    board_i = updater.infinite_board(snakes, food)
    print(board_i)
    board_j = updater.finite_board(snakes, food)
    print(board_j)

    assert jnp.array_equal(board_i, board_j)


def test_jitted_board_fn_correctness():
    game = generate_empty_board()

    #  Spawn food
    add_food(game, [(8, 8), (9, 9), (10, 10)])

    # Add snakes
    for i in range(4):
        snake = Snake()
        snake.id = "snake_{}".format(i)
        snake.health = 100
        snake.body = [(3, i), (4, i), (5, i), (6, i)]
        add_snake(game, snake)

    food = game.food
    snakes = list([snake.body for _, snake in game.snakes.items()])
    updater = BoardUpdater(11, 11, 4)
    jitted_fn = jax.jit(updater.finite_board)

    board_i = updater.infinite_board(snakes, food, game.boards["snake_0"])
    print(board_i)
    board_j = jitted_fn(snakes, food, game.boards["snake_0"])
    print(board_j)

    assert jnp.array_equal(board_i, board_j)


def obs_benchmark(game: PythonStandard4Player):
    def run():
        for i in range(500):
            game.seed(i)
            game.reset()
            game.global_observation()

    return timeit.timeit(stmt=run, number=1)


def measure_obs_interpreter_performance():
    updater = BoardUpdater(11, 11, 4, False)
    game = PythonStandard4Player(0, updater)
    time = obs_benchmark(game)
    print("Interpreter: 500 loops run. Time: {}. Per loop: {}".format(time, time / 500))
    return time / 500


def measure_obs_jit_performance():
    updater = BoardUpdater(11, 11, 4, True)
    game = PythonStandard4Player(0, updater)
    time = obs_benchmark(game)
    print("JIT: 500 loops run. Time: {}. Per loop: {}".format(time, time / 500))
    return time / 500


def test_compare_obs_jit_performance():
    print("\n\n")
    rate_i = measure_obs_interpreter_performance()
    rate_j = measure_obs_jit_performance()
    percent_diff = 100 * rate_i / rate_j
    print("\nThe JIT is {:.0f}% faster than the interpreter.".format(percent_diff))
    assert rate_i > rate_j


def test_board_perspectives():
    game = create_game(0)
    game.snakes_array = [0, 1, 2, 3]
    for name, _ in game.snakes.items():
        print(game.get_observation(name))
