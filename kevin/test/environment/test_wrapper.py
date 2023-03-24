from kevin.src.engine.board_updater import FixedBoardUpdater
from kevin.src.engine.python_engine import PythonGameState
from kevin.src.environment.rewinding_environment import RewindingEnv
from kevin.src.environment.wrapper import FrameStacking
import jax.numpy as jnp

updater = FixedBoardUpdater(11, 11)


def create_env():
    game = PythonGameState(updater=updater)
    env = RewindingEnv(game)
    return env, game


def test_obs_stacking_wrapper():
    env, game = create_env()
    w_env = FrameStacking(env)

    obs_1 = w_env.reset()['snake_0']
    assert jnp.array_equal(obs_1[0], obs_1[1])

    # One step
    obs_2 = w_env.step({f"snake_{i}": i for i in range(4)})[0]['snake_0']
    print(obs_2)

    assert jnp.array_equal(obs_2[1], obs_1[0])
    assert not jnp.array_equal(obs_2[0], obs_1[1])

    # Second step
    obs_3 = w_env.step({f"snake_{i}": i for i in range(4)})[0]['snake_0']
    print(obs_3)

    assert jnp.array_equal(obs_3[1], obs_2[0])
    assert not jnp.array_equal(obs_3[0], obs_2[1])


def test_obs_stacking_space():
    env, game = create_env()
    w_env = FrameStacking(env)
    obs = w_env.reset()['snake_0']
    space = w_env.observation_space('snake_0')
    assert obs in space
    print(space)
    assert w_env.dummy_gym_environment.observation_space == space
