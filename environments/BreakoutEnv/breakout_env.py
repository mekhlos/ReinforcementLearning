import numpy as np
import pygame

from environments.BreakoutEnv import breakout
from environments import environment_interface

block_width = 23
block_height = 15
blue = (0, 0, 255)


def get_block_segment(pixels):
    return pixels.T[80:80 + 5 * (block_height + 2)]


def count_blue_pixels(pixels):
    return (pixels == 255).sum()


class BreakoutEnv(environment_interface.EnvironmentInterface):
    def __init__(self):
        self.game = breakout.Game()
        self.prev_state = None

    def get_action_space(self):
        return np.array([0, 1])

    def _step(self, prev_state, action):
        self.game.step(action)
        new_state = self.game.get_pixels()
        prev_state = prev_state if prev_state is not None else new_state
        reward = count_blue_pixels(get_block_segment(prev_state) - get_block_segment(new_state))
        is_done = self.game.game_over or self.game.exit_program

        return new_state, reward, is_done

    def _step_wrap(self, prev_state, action):
        frames = []
        reward = 0
        is_done = False
        for i in range(3):
            new_state, reward, is_done = self._step(prev_state, action)
            frames.append(new_state)

        return np.array(frames), reward, is_done

    def update(self, action):
        frames, reward, is_done = self._step_wrap(self.prev_state, action)
        self.prev_state = frames[-1]
        return frames, reward, is_done

    def reset(self):
        self.game = breakout.Game()
        self.prev_state = None

    def get_state(self):
        if self.prev_state is None:
            raise Exception('prev_state is not initialised!')
        return self.prev_state

    def close(self):
        pygame.quit()

    def display(self):
        pass
