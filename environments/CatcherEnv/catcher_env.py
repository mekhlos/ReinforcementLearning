import numpy as np
import pygame

from environments.CatcherEnv import catcher
from environments import environment_interface
from DQN.breakout.env_config import Config1 as env_config


class CatcherEnv(environment_interface.EnvironmentInterface):

    def __init__(self):
        self.game = catcher.Game()
        self.game.display()
        self.prev_state = np.repeat(np.expand_dims(self.game.get_pixels(), 0), 3, 0)

    def get_action_space(self):
        return np.array([0, 1])

    def _step(self, prev_state, action):
        result = self.game.step(action)
        if result == self.game.StepRes.CAUGHT:
            reward = 1
        elif result == self.game.StepRes.DROPPED:
            reward = -0.1
        else:
            reward = -0.001

        new_state = self.game.get_pixels()
        is_done = self.game.game_over or self.game.exit_program

        if is_done:
            reward = -1

        return new_state, reward, is_done

    def _step_wrap(self, prev_state, action):
        frames = []
        reward_sum = 0
        is_done = False
        for i in range(3):
            new_state, reward, is_done = self._step(prev_state[-1], action)
            frames.append(new_state)
            reward_sum += reward

        return np.array(frames), reward_sum, is_done

    def update(self, action):
        frames, reward, is_done = self._step_wrap(self.prev_state, action)
        self.prev_state = frames
        if is_done:
            return frames, reward, True, None
        else:
            return frames, reward, False, None

    def reset(self):
        self.game = catcher.Game()
        self.game.display()
        self.prev_state = np.repeat(np.expand_dims(self.game.get_pixels(), 0), 3, 0)

    def get_state(self):
        if self.prev_state is None:
            raise Exception('prev_state is not initialised!')
        return self.prev_state

    def close(self):
        pygame.quit()

    def display(self):
        pass
