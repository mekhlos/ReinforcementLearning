import numpy as np
import pygame

from environments.BreakoutEnv import breakout
from environments import environment_interface
from DQN.breakout.env_config import Config1 as env_config

block_width = env_config.block_width
block_height = env_config.block_height
blue = (0, 0, 255)


class BreakoutEnv(environment_interface.EnvironmentInterface):
    BLOCK_SIZE = block_height * block_width

    def __init__(self):
        self.game = breakout.Game()
        self.game.display()
        self.prev_state = np.repeat(np.expand_dims(self.game.get_pixels(), 0), 3, 0)

    def get_action_space(self):
        return np.array([0, 1])

    def _step(self, action):
        n_blocks1 = len(self.game.blocks)
        self.game.step(action)
        n_blocks2 = len(self.game.blocks)

        new_state = self.game.get_pixels()

        reward = n_blocks1 - n_blocks2
        is_done = self.game.game_over or self.game.exit_program

        return new_state, reward, is_done

    def _step_wrap(self, action):
        frames = []
        reward_sum = 0
        is_done_final = False

        for i in range(3):
            new_state, reward, is_done = self._step(action)
            reward_sum += reward
            is_done_final |= is_done
            frames.append(new_state)

        return np.array(frames), reward_sum, is_done_final

    def update(self, action):
        frames, reward, is_done = self._step_wrap(action)
        self.prev_state = frames
        if is_done:
            return frames, -1, True, None
        else:
            return frames, reward - 0.001, False, None

    def reset(self):
        self.game = breakout.Game()
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
