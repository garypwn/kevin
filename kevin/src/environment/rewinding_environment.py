from typing import Optional, Tuple, Dict

from pettingzoo.utils.env import ObsDict, ActionDict

from kevin.src.engine.python_engine import PythonGameState
from kevin.src.engine.snake_engine import GameState
from kevin.src.environment.snake_environment import MultiSnakeEnv


class RewindingEnv(MultiSnakeEnv):
    stack_max = 7
    stale_at = 25
    stale_counter = 0
    stack: list[GameState] = []

    def __init__(self, eng: GameState):
        super().__init__(eng)

        # Set a save replay flag in the game engine
        self.game.save_replays = True

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None) -> ObsDict:

        if seed is not None:
            self.game.seed(seed)

        # After several rewinds, we say the game is stale and start a new one.
        if self.stale_counter >= self.stale_at:
            self.stale_counter = 0
            self.stack = []

        # If there are saved rewinds in the stack, we can use one of those
        if len(self.stack) > 0:

            # Put the most interesting game on the top
            self.stack.sort(
                key=lambda game: game.turn_num * sum([len(snake.body) for _, snake in game.snakes.items()]),
                reverse=True)
            self.game = self.stack.pop()
            self.stale_counter += 1

        else:
            self.game = self.game.reset(options)

        self.agents = [name for name, _ in self.game.snakes.items()]

        observations = {agent: self.game.get_observation(agent) for agent in self.agents}
        if not return_info:
            return observations

        else:
            infos = {agent: self.game.get_info(agent) for agent in self.agents}
            return infos

    def step(self, actions: ActionDict) -> Tuple[
        ObsDict, Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]
    ]:

        new_game = self.game.step(actions)

        # If replays are on, the game will have set a flag if it is worth replaying
        if isinstance(self.game, PythonGameState) and self.game.replay_flag:
            self.stack.append(self.game)
            self.game.replay_flag = False
            self.game.branch_name += "{} - ".format(self.game.turn_num)

        self.game = new_game

        observations = {agent: self.game.get_observation(agent) for agent in self.agents}
        rewards = {agent: self.game.get_reward(agent) for agent in self.agents}
        terminations = {agent: self.game.get_terminated(agent) for agent in self.agents}
        truncations = {agent: self.game.get_truncated(agent) for agent in self.agents}
        infos = {agent: self.game.get_info(agent) for agent in self.agents}

        # Remove terminated or truncated agents? This is not well-defined by spec
        for agent in self.agents[:]:
            if terminations[agent] or truncations[agent]:
                self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos
