import numpy as np
import gym
from environments import environment_interface


class EnvWrapper(environment_interface.EnvironmentInterface):
    def __init__(self, gym_env):
        self.env: gym.Env = gym_env
        self.state = None

    def display(self):
        return self.env.render()

    def get_action_space(self):
        return np.array(range(self.env.action_space.n))

    def update(self, action):
        s, r, d, i = self.env.step(action)
        self.state = s
        return self.get_state(), r, d, i

    def reset(self):
        self.state = self.env.reset()
        return self.get_state()

    def get_state(self):
        if self.state is None:
            raise Exception('Call reset or update first!')

        return np.array(self.state)

    def close(self):
        self.env.close()
