import sys
import os
import root_file

gwpath = os.path.join(root_file.ROOT_DIR, 'environments/GridworldEnv')
sys.path.append(gwpath)

import numpy as np

from dqn.tf_network import NetworkManager
from dqn.brain.replay_memory import ReplayMemory
from dqn.exploration_strategy import ExplorationStrategy
from utils import visualiser
from environments.GridworldEnv import gridworld_env
from environments.GridworldEnv.grid_configs import configs
from environments.GridworldEnv.gridworld import Actions
from dqn.brain import q_learning

N_EPISODES = 1500
# START_REPLAY = 5000
EPISODE_LENGTH = 50
MEMORY_SIZE = 300
REPLAY_FREQUENCY = 1
START_EPSILON = 1
STOP_EPSILON = 0.01
INPUT_DIM = 3 * 3 * 3
N_ACTIONS = 4
BATCH_SIZE = 50
DISCOUNT_FACTOR = 0.98
ALPHA = 1 / BATCH_SIZE / 100

replay_memory = ReplayMemory(MEMORY_SIZE)
# network = Network(N_ACTIONS)
network_manager = NetworkManager(INPUT_DIM, N_ACTIONS)
q_table = q_learning.QTable(9, N_ACTIONS)
env = gridworld_env.GridworldEnv(3, 3, grid=configs.to_state(configs.config4))

exploration_helper = ExplorationStrategy(N_EPISODES * 0.75, START_EPSILON, STOP_EPSILON)
plotter = visualiser.PlotManager()


def process_state(state):
    state = state[[0, 1, 3]]
    return state.flatten()


def add_noise(x):
    noise = np.random.standard_normal(x.shape) * 0.1
    return x + noise


def replay():
    state, action, reward, new_state, terminal_flag = replay_memory.sample(BATCH_SIZE)
    # q_values = network.predict(state)
    q_values = network_manager.predict(state)
    non_terminal_flag = 1 - terminal_flag
    # new_q_values = network.predict(new_state[non_terminal_flag])
    new_q_values = network_manager.predict(new_state[non_terminal_flag])

    target = add_noise(q_values)
    target[range(len(target)), action][terminal_flag] = reward[terminal_flag]
    target[range(len(target)), action][non_terminal_flag] = \
        reward[non_terminal_flag] + DISCOUNT_FACTOR * new_q_values.max(1)

    # network.learn(state, target)
    network_manager.learn(state, target)


def replay_table():
    for state, action, reward, new_state, is_terminal in zip(*replay_memory.sample(BATCH_SIZE)):
        old_q_value = q_table[state, action]
        if is_terminal:
            new_q_value = reward
        else:
            new_q_value = reward + DISCOUNT_FACTOR + q_table.get_q_values_for_state(new_state).max()

        update = (1 - ALPHA) * old_q_value + ALPHA * new_q_value

        q_table.update(state, action, update)


avg = 0
k = 0
for i in range(N_EPISODES):
    state = env.reset()
    print('Reset')
    state = process_state(state)
    total_reward = 0

    for j in range(EPISODE_LENGTH):
        # q_values = network.predict(state.reshape(1, -1))
        # q_values = network_manager.predict(state.reshape(1, -1))
        q_values = add_noise(q_table.get_q_values_for_state(state.squeeze()))
        # print('q values', q_values, len(q_values))
        action = exploration_helper.epsilon_greedy(q_values)
        new_state, reward, is_terminal, _ = env.take_action(Actions.get_actions()[action])
        env.display()
        new_state = process_state(new_state)

        replay_memory.add(state, action, reward, new_state, is_terminal)
        total_reward += reward

        if len(replay_memory) > BATCH_SIZE and j % REPLAY_FREQUENCY == 0:
            replay_table()

        if is_terminal:
            print(f'Finished in {j} steps, reward {reward} and total {total_reward}')
            break

        state = new_state

    exploration_helper.update_epsilon()

    print(f'Epsilon: {exploration_helper.epsilon}')
    print(f'Total reward: {total_reward}')
    avg = (avg * k + total_reward) / (k + 1)
    k = (k + 1) % 50
    if i % 10 == 0:
        plotter.update(i, avg)
