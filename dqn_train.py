import sys
import os
import root_file
import rl

gwpath = os.path.join(root_file.ROOT_DIR, 'environments/GridworldEnv')
sys.path.append(gwpath)

import numpy as np

from dqn.brain.tf_network import NetworkManager
from dqn.brain.replay_memory import ReplayMemory
from dqn.exploration_strategy import ExplorationStrategy
from utils import visualiser
from environments.GridworldEnv import gridworld_env
from environments.GridworldEnv.grid_configs import configs
from environments.GridworldEnv.gridworld import Actions
from dqn.brain import q_learning


def batch_replay(network_manager, replay_memory, batch_size, discount_factor):
    state, action, reward, new_state, terminal_flag = replay_memory.sample(batch_size)
    q_values = network_manager.predict(state)
    terminal_ix = np.where(terminal_flag > 0)[0]
    non_terminal_ix = np.where(terminal_flag < 1)[0]
    new_q_values = network_manager.predict(new_state[non_terminal_ix])

    target = q_values.copy()
    target[terminal_ix, action[terminal_ix]] = reward[terminal_ix]
    target[non_terminal_ix, action[non_terminal_ix]] = \
        reward[non_terminal_ix] + discount_factor * new_q_values.max(1)

    loss = network_manager.learn(state, target)
    return loss


def iterative_replay(network_manager, replay_memory, batch_size, discount_factor):
    state, action, reward, new_state, terminal_flag = replay_memory.sample(batch_size)
    q_values = network_manager.predict(state)
    new_q_values = network_manager.predict(new_state)
    targets = q_values.copy()

    for i, (r, a, t, new_q_value) in enumerate(zip(reward, action, terminal_flag, new_q_values)):
        if t:
            targets[i][a] = r
        else:
            targets[i][a] = r + discount_factor * new_q_value.max()

    loss = network_manager.learn(state, targets)
    return loss


def table_replay(q_table, replay_memory, batch_size, discount_factor):
    alpha = 1 / batch_size / 100
    batch = replay_memory.sample(batch_size)

    for i, (s, a, r, s2, t) in enumerate(zip(*batch)):
        old_q = q_table[s, a]
        new_q = q_table.get_q_values_for_state(s2)
        if t:
            target = r
        else:
            target = r + discount_factor * new_q.max()

        update = (1 - alpha) * old_q + alpha * target

        q_table.update(s, a, update)

    return 0


class DQNTeacher:
    def __init__(self, agent: rl.Agent, replay_memory, env, network_manager, exploration_helper, replay_f, settings):
        self.agent = agent
        self.replay_memory = replay_memory
        self.env = env
        self.network_manager = network_manager
        self.exploration_helper = exploration_helper
        self.current_episode = 0
        self.plot_manager = visualiser.DataPlotter()
        self.plot_manager.add_plot('loss', (0, settings.N_EPISODES), (-1, 10), 'loss')
        self.plot_manager.add_plot('reward', (0, settings.N_EPISODES), (-50, 20), 'reward')
        self.settings = settings
        self.loss = []
        self.rewards = []
        self.replay = replay_f
        self.test_q_table = q_learning.QTable(settings.INPUT_DIM // 3, 4)

        # self.plot_manager.add_plot('q_values', (0, settings.N_EPISODES), (0, 100), 'q_values')

    @staticmethod
    def add_noise(x):
        noise = np.random.standard_normal(x.shape) * 0.1
        return x + noise

    def train(self):
        for i in range(self.settings.N_EPISODES):
            self.current_episode = i
            self.loss.append(0)
            self.env.reset()
            state = self.agent.observe()
            total_reward = 0

            for j in range(self.settings.EPISODE_LENGTH):

                q_values = self.network_manager.predict_one(state)

                action = self.exploration_helper.epsilon_greedy(q_values)
                new_state, reward, is_terminal, _ = self.env.update(Actions.get_actions()[action])
                if i % 20 == 0:
                    self.env.display()
                new_state = self.agent.observe()

                self.replay_memory.add(state, action, reward, new_state, is_terminal)
                total_reward += reward

                if len(self.replay_memory) > self.settings.BATCH_SIZE and j % self.settings.REPLAY_FREQUENCY == 0:
                    loss = self.replay(self.network_manager, self.replay_memory, self.settings.BATCH_SIZE,
                                       self.settings.DISCOUNT_FACTOR)
                    self.loss[i] += loss

                if is_terminal:
                    print(f'Finished in {j} steps, reward {reward} and total {total_reward}')
                    break

                state = new_state

                if i % 10 == 0:
                    self.test_q_table.update(state, action, q_values.max())

            self.exploration_helper.update_epsilon()
            self.rewards.append(total_reward)
            print(f'Epsilon: {self.exploration_helper.epsilon}')
            # print(f'Total reward: {total_reward}')
            if i % 20 == 0:
                r = visualiser.moving_average(self.rewards, 100)[-1]
                l = visualiser.moving_average(self.loss, 100)[-1]
                print(f'loss: {l}')
                self.plot_manager.update_plot('reward', i, r)
                self.plot_manager.update_plot('loss', i, l)
                # print(self.test_q_table)


config = configs.config2
M = len(config)
N = len(config[0])


class Settings:
    N_EPISODES = 1000
    EPISODE_LENGTH = 100
    MEMORY_SIZE = 400
    REPLAY_FREQUENCY = 2
    START_EPSILON = 1
    STOP_EPSILON = 0.01
    INPUT_DIM = M * N * 3
    N_ACTIONS = 4
    BATCH_SIZE = 32
    DISCOUNT_FACTOR = 0.99
    ALPHA = 1 / BATCH_SIZE / 100


class Settings2:
    N_EPISODES = 1000
    EPISODE_LENGTH = 300
    MEMORY_SIZE = 600
    REPLAY_FREQUENCY = 3
    START_EPSILON = 1
    STOP_EPSILON = 0.01
    INPUT_DIM = M * N * 3
    N_ACTIONS = 4
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.985
    ALPHA = 1 / BATCH_SIZE / 100


class TableSettings:
    N_EPISODES = 2500
    EPISODE_LENGTH = 70
    MEMORY_SIZE = 400
    REPLAY_FREQUENCY = 4
    START_EPSILON = 1
    STOP_EPSILON = 0.1
    INPUT_DIM = M * N * 3
    N_ACTIONS = 4
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.975
    ALPHA = 1 / BATCH_SIZE / 100


if __name__ == '__main__':
    settings = Settings2()
    env = gridworld_env.GridworldEnv(M, N, grid=configs.to_state(config))
    exploration_helper = ExplorationStrategy(settings.N_EPISODES * 0.7, settings.START_EPSILON, settings.STOP_EPSILON)
    agent = rl.Agent('test1', env)
    replay_memory = ReplayMemory(settings.MEMORY_SIZE)
    network_manager = NetworkManager(settings.INPUT_DIM, settings.N_ACTIONS)
    q_table = q_learning.QTable(16, settings.N_ACTIONS)
    dqn = DQNTeacher(agent, replay_memory, env, network_manager, exploration_helper, iterative_replay, settings)
    # dqn = DQNTeacher(agent, replay_memory, env, q_table, exploration_helper, table_replay, Settings())
    dqn.train()
