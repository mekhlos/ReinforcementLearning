import sys
import os
import root_file
import time

gwpath = os.path.join(root_file.ROOT_DIR, 'environments/GridworldEnv')
sys.path.append(gwpath)

import numpy as np

from dqn.network import Network
from dqn.replay_memory import ReplayMemory
from dqn.exploration_strategy import ExplorationStrategy
from utils import logging_helper
from utils import visualiser
from environments.GridworldEnv import gridworld_env
from environments.GridworldEnv.grid_configs import configs
from environments.GridworldEnv.gridworld import Actions

N_EPISODES = 3000
# START_REPLAY = 5000
EPISODE_LENGTH = 30
MEMORY_SIZE = 250
REPLAY_FREQUENCY = 1
START_EPSILON = 0.5
STOP_EPSILON = 0.01
N_ACTIONS = 4
BATCH_SIZE = 50
GAMMA = 0.5

replay_memory = ReplayMemory(MEMORY_SIZE)
network = Network(N_ACTIONS)
env = gridworld_env.GridworldEnv(3, 3, grid=configs.to_state(configs.config4))

exploration_helper = ExplorationStrategy(N_EPISODES * 0.8, START_EPSILON, STOP_EPSILON)
plotter = visualiser.PlotManager()


def process_state(state):
    state = state[[0, 1, 3]]
    return state.flatten()


def add_noise(x):
    noise = np.random.standard_normal(x.shape) * 0.1
    return x + noise


def replay():
    state, action, reward, new_state, terminal_flag = replay_memory.sample(BATCH_SIZE)
    q_values = network.predict(state)
    non_terminal_flag = 1 - terminal_flag
    new_q_values = network.predict(new_state[non_terminal_flag])

    target = add_noise(q_values)
    target[range(len(target)), action][terminal_flag] = reward[terminal_flag]
    target[range(len(target)), action][non_terminal_flag] = reward[non_terminal_flag] + GAMMA * new_q_values.argmax(1)

    network.learn(state, target)

    # print(f'Reward: {reward}')
    # print(f'Action: {action}')
    # print(f'Diff: {target - q_values}')


avg = 0
k = 0
for i in range(N_EPISODES):
    state = env.reset()
    print('Reset')
    state = process_state(state)
    total_reward = 0

    for j in range(EPISODE_LENGTH):
        q_values = network.predict(state.reshape(1, -1))
        # print('q values', q_values, len(q_values))
        action = exploration_helper.epsilon_greedy(q_values)
        new_state, reward, is_terminal, _ = env.take_action(Actions.get_actions()[action])
        env.display()
        new_state = process_state(new_state)

        replay_memory.add(state, action, reward, new_state, is_terminal)
        total_reward += reward

        if len(replay_memory) > BATCH_SIZE and j % REPLAY_FREQUENCY == 0:
            replay()

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
