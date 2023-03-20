from kevin.src.engine.board_updater import RotatingBoardUpdater
from kevin.src.engine.python_engine import PythonGameState
from kevin.src.environment.rewinding_environment import RewindingEnv
from kevin.src.environment.snake_environment import MultiSnakeEnv
import jax.numpy as jnp

updater = RotatingBoardUpdater(11, 11)


def make_boards():
    game = PythonGameState(0, updater=updater)
    game_n = PythonGameState(0, updater=updater)
    env = RewindingEnv(game)
    env.fancy_render = True
    normal_env = MultiSnakeEnv(game_n)
    normal_env.fancy_render = True
    return env, normal_env


def test_same_observations():
    env1, env2 = make_boards()
    compare_obs(env1, env2)


def compare_obs(env1, env2):
    print(env1.render())
    print(env2.render())

    for agent in env1.agents:
        o, on = env1.game.get_observation(agent), env2.game.get_observation(agent)
        for board in o:
            print(board, "\n")
        print("\n\n")
        for board in on:
            print(board, "\n")

        print("Obs 1 shape: {}. Obs 2 shape: {}".format(o.shape, on.shape))
        assert jnp.array_equal(o, on)


def test_same_observation_after_moving():
    moves = [1, 2, 0, 1]
    moves2 = [2, 0, 1, 2]

    env1, env2 = make_boards()
    env1.step({"snake_{}".format(i): moves[i] for i in range(4)})
    env2.step({"snake_{}".format(i): moves[i] for i in range(4)})

    compare_obs(env1, env2)

    env1.step({"snake_{}".format(i): moves2[i] for i in range(4)})
    env2.step({"snake_{}".format(i): moves2[i] for i in range(4)})

    compare_obs(env1, env2)


def test_same_observation_when_spawning_rewind():
    env1, env2 = make_boards()
    env1.game.turn_num = 15
    env2.game.turn_num = 15

    # Kill some snakes
    env1.step({"snake_{}".format(i): 1 for i in range(4)})
    env2.step({"snake_{}".format(i): 1 for i in range(4)})
    env1.step({"snake_{}".format(i): 1 for i in range(4)})
    env2.step({"snake_{}".format(i): 1 for i in range(4)})

    compare_obs(env1, env2)
    assert len(env1.state_pq) > 0


def test_same_observation_after_rewind():
    env1, env2 = make_boards()
    env1.game.turn_num = 15
    env2.game.turn_num = 15

    # Kill some snakes
    env1.step({"snake_{}".format(i): 1 for i in range(4)})
    env2.step({"snake_{}".format(i): 1 for i in range(4)})
    env1.step({"snake_{}".format(i): 1 for i in range(4)})
    env1.step({"snake_{}".format(i): 1 for i in range(4)})
    env1.reset()

    compare_obs(env1, env2)
    assert len(env1.state_pq) == 0
