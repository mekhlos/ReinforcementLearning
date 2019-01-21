import rl
import gym
import numpy as np

from networks.tf_network import NetworkManager
from dqn.cartpole.network_architecture import MyNetwork, MyNetworkTrainer
from dqn.memory.replay_memory import ReplayMemory
from dqn.exploration_strategy import ExplorationStrategy
from dqn import q_table, dqn_trainer


class Settings:
    N_EPISODES = 10000
    EPISODE_LENGTH = 5000
    MEMORY_SIZE = 5000
    REPLAY_FREQUENCY = 2
    START_EPSILON = 1
    STOP_EPSILON = 0.01
    INPUT_DIM = 4
    N_ACTIONS = 2
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.97
    ALPHA = 1 / BATCH_SIZE / 100


class EnvWrapper:
    def __init__(self, gym_env: gym.Env):
        self.env = gym_env
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

    def observe_f(self):
        return self.get_state()


if __name__ == '__main__':
    settings = Settings()
    env = EnvWrapper(gym.make('CartPole-v0'))
    exploration_helper = ExplorationStrategy(settings.N_EPISODES * 0.7, settings.START_EPSILON, settings.STOP_EPSILON)
    agent = rl.Agent('test1', env, observe_function=env.observe_f)
    replay_memory = ReplayMemory(settings.MEMORY_SIZE)
    network_manager = NetworkManager(settings.INPUT_DIM, settings.N_ACTIONS, MyNetwork, MyNetworkTrainer)
    q_table = q_table.QTable(16, settings.N_ACTIONS)
    dqn = dqn_trainer.DQNTeacher(agent, replay_memory, env, network_manager, exploration_helper,
                                 dqn_trainer.batch_replay, settings)
    dqn.train()
