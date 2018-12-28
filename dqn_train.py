import sys
import os
import time
import root_file

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


class QValueManager:
    def __init__(self):
        pass

    def get_q_values(self, state):
        pass

    def update_q_values(self, state, value):
        pass


class Agent:
    def __init__(self, agent_id, environment):
        self.agent_id = agent_id
        self.environment = environment

    def observe(self):
        environment_state = self.environment.get_state()
        state = environment_state[[0, 1, 3]]
        return state.flatten()

    def take_action(self, action):
        self.environment.update(action)


def batch_replay(network_manager, batch_size, discount_factor):
    state, action, reward, new_state, terminal_flag = replay_memory.sample(batch_size)
    q_values = network_manager.predict(state)
    non_terminal_flag = 1 - terminal_flag
    new_q_values = network_manager.predict(new_state[non_terminal_flag])

    target = q_values.copy()
    target[range(len(target)), action][terminal_flag] = reward[terminal_flag]
    target[range(len(target)), action][non_terminal_flag] = \
        reward[non_terminal_flag] + discount_factor * new_q_values.max(1)

    loss = network_manager.learn(state, target)
    return loss


def iterative_replay(network_manager, batch_size, discount_factor):
    state, action, reward, new_state, terminal_flag = replay_memory.sample(batch_size)
    q_values = network_manager.predict(state)
    new_q_values = network_manager.predict(new_state)
    targets = q_values.copy()

    for i, (r, t, new_q_value) in enumerate(zip(reward, terminal_flag, new_q_values)):
        if t:
            targets[i] = r
        else:
            targets[i] = r + discount_factor + new_q_value.max()

    loss = network_manager.learn(state, targets)
    return loss


def table_replay(q_table, batch_size, discount_factor):
    alpha = 1 / batch_size / 100
    batch = replay_memory.sample(batch_size)

    for i, (s, a, r, s2, t) in enumerate(zip(*batch)):
        old_q = q_table[s, a]
        new_q = q_table.get_q_values_for_state(s2)
        if t:
            target = r
        else:
            target = r + discount_factor + new_q.max()

        update = (1 - alpha) * old_q + alpha * target

        q_table.update(s, a, update)

    return 0


class DQNTeacher:
    def __init__(self, agent: Agent, replay_memory, env, network_manager, exploration_helper, replay_f, settings):
        self.agent = agent
        self.replay_memory = replay_memory
        self.env = env
        self.network_manager = network_manager
        self.exploration_helper = exploration_helper
        self.current_episode = 0
        self.plot_manager = visualiser.DataPlotter()
        self.plot_manager.add_plot('loss', (0, settings.N_EPISODES), (0, 100), 'loss')
        self.plot_manager.add_plot('reward', (0, settings.N_EPISODES), (-100, 20), 'reward')
        self.settings = settings
        self.loss = []
        self.rewards = []
        self.replay = replay_f

    @staticmethod
    def add_noise(x):
        noise = np.random.standard_normal(x.shape) * 0.1
        return x + noise

    def train(self):
        for i in range(self.settings.N_EPISODES):
            self.current_episode = i
            self.loss.append(0)
            env.reset()
            state = self.agent.observe()
            total_reward = 0

            for j in range(self.settings.EPISODE_LENGTH):

                q_values = self.network_manager.predict_one(state)

                action = self.exploration_helper.epsilon_greedy(q_values)
                new_state, reward, is_terminal, _ = self.env.update(Actions.get_actions()[action])
                self.env.display()
                new_state = self.agent.observe()

                replay_memory.add(state, action, reward, new_state, is_terminal)
                total_reward += reward

                if len(replay_memory) > self.settings.BATCH_SIZE and j % self.settings.REPLAY_FREQUENCY == 0:
                    loss = self.replay(self.network_manager, self.settings.BATCH_SIZE, self.settings.DISCOUNT_FACTOR)
                    self.loss[i] += loss

                if is_terminal:
                    print(f'Finished in {j} steps, reward {reward} and total {total_reward}')
                    break

                state = new_state

            exploration_helper.update_epsilon()
            self.rewards.append(total_reward)
            print(f'Epsilon: {exploration_helper.epsilon}')
            print(f'Total reward: {total_reward}')
            if i % 10 == 0:
                r = visualiser.moving_average(self.rewards, 100)[-1]
                l = visualiser.moving_average(self.loss, 100)[-1]
                self.plot_manager.update_plot('reward', i, r)
                self.plot_manager.update_plot('loss', i, l)


class Settings:
    N_EPISODES = 2500
    EPISODE_LENGTH = 70
    MEMORY_SIZE = 400
    REPLAY_FREQUENCY = 4
    START_EPSILON = 1
    STOP_EPSILON = 0.1
    INPUT_DIM = 3 * 3 * 3
    N_ACTIONS = 4
    BATCH_SIZE = 64
    DISCOUNT_FACTOR = 0.975
    ALPHA = 1 / BATCH_SIZE / 100


if __name__ == '__main__':
    env = gridworld_env.GridworldEnv(3, 3, grid=configs.to_state(configs.config4))
    exploration_helper = ExplorationStrategy(Settings.N_EPISODES * 0.8, Settings.START_EPSILON, Settings.STOP_EPSILON)
    agent = Agent('test1', env)
    replay_memory = ReplayMemory(Settings.MEMORY_SIZE)
    network_manager = NetworkManager(Settings.INPUT_DIM, Settings.N_ACTIONS)
    q_table = q_learning.QTable(9, Settings.N_ACTIONS)
    dqn = DQNTeacher(agent, replay_memory, env, network_manager, exploration_helper, iterative_replay, Settings())
    # dqn = DQNTeacher(agent, replay_memory, env, q_table, exploration_helper, table_replay, Settings())
    dqn.train()
