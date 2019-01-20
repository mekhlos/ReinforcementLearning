import sys
import os
import root_file
import rl

gwpath = os.path.join(root_file.ROOT_DIR, 'environments/GridworldEnv')
sys.path.append(gwpath)

import numpy as np

from utils import visualiser
from environments.GridworldEnv.gridworld import Actions
from dqn import q_table


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
        # self.plot_manager.add_plot('loss', (0, settings.N_EPISODES), (-1, 10), 'loss')
        self.plot_manager.add_plot('reward', (0, settings.N_EPISODES), (-50, 20), 'reward')
        self.settings = settings
        self.loss = []
        self.rewards = []
        self.replay = replay_f
        self.test_q_table = q_table.QTable(settings.INPUT_DIM // 3, 4)

        # self.plot_manager.add_plot('q_values', (0, settings.N_EPISODES), (0, 100), 'q_values')

    @staticmethod
    def add_noise(x):
        noise = np.random.standard_normal(x.shape) * 0.1
        return x + noise

    def train(self):
        for i in range(self.settings.N_EPISODES):
            print(f'Episode {i}')

            self.current_episode = i
            self.loss.append(0)
            self.env.reset()
            state = self.agent.observe()
            total_reward = 0

            for j in range(self.settings.EPISODE_LENGTH):

                q_values = self.network_manager.predict_one(state)

                action = self.exploration_helper.epsilon_greedy(q_values)
                new_state, reward, is_terminal, _ = self.env.update(Actions.get_actions()[action])
                if i % 10 == 0:
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
            if i % 10 == 0:
                r = visualiser.moving_average(self.rewards, 100)[-1]
                l = visualiser.moving_average(self.loss, 100)[-1]
                print(f'loss: {l}')
                self.plot_manager.update_plot('reward', i, r)
                # self.plot_manager.update_plot('loss', i, l)
                # print(self.test_q_table)


if __name__ == '__main__':
    pass
