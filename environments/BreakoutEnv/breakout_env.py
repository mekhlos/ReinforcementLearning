import numpy as np
import pygame

from environments.BreakoutEnv import breakout
from environments import environment_interface
from DQN.breakout.env_config import Config1 as env_config

block_width = env_config.block_width
block_height = env_config.block_height
blue = (0, 0, 255)


def get_block_segment(pixels):
    row_start = env_config.block_start_from_top
    row_end = row_start + env_config.n_block_rows * (block_height + env_config.block_padding) + 1
    return pixels.T[row_start:row_end]


def count_blue_pixels(pixels):
    return (pixels == 255).sum()


class BreakoutEnv(environment_interface.EnvironmentInterface):
    BLOCK_SIZE = block_height * block_width

    def __init__(self):
        self.game = breakout.Game()
        self.game.display()
        self.prev_state = np.repeat(np.expand_dims(self.game.get_pixels(), 0), 3, 0)

    def get_action_space(self):
        return np.array([0, 1])

    def _step(self, prev_state, action):
        self.game.step(action)
        new_state = self.game.get_pixels()
        reward = count_blue_pixels(get_block_segment(prev_state)) - count_blue_pixels(get_block_segment(new_state))
        reward /= self.BLOCK_SIZE
        assert reward == int(reward)
        is_done = self.game.game_over or self.game.exit_program

        return new_state, reward, is_done

    def _step_wrap(self, prev_state, action):
        frames = []
        reward = 0
        is_done = False
        for i in range(3):
            new_state, reward, is_done = self._step(prev_state[-1], action)
            frames.append(new_state)

        return np.array(frames), reward, is_done

    def update(self, action):
        frames, reward, is_done = self._step_wrap(self.prev_state, action)
        self.prev_state = frames
        if is_done:
            return frames, - 1, True, None
        else:
            return frames, reward / 10 - 0.001, False, None

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
