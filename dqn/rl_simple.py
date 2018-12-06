import sys
import os
import root_file
import time

gwpath = os.path.join(root_file.ROOT_DIR, 'environments/GridworldEnv')
sys.path.append(gwpath)

import numpy as np

from dqn.network import Network
from dqn.replay_memory import ReplayMemory
from environments.GridworldEnv import gridworld_env
from environments.GridworldEnv.grid_configs import configs
from environments.GridworldEnv.gridworld import Actions

N_EPISODES = 10000
MEMORY_SIZE = 1000
REPLAY_FREQUENCY = 10
EPSILON = 0.1
N_ACTIONS = 4
BATCH_SIZE = 100
GAMMA = 0.91

replay_memory = ReplayMemory(MEMORY_SIZE)
network = Network(N_ACTIONS)
env = gridworld_env.GridworldEnv(4, 4, grid=configs.to_state(configs.config3))


def epsilon_greedy(q_values, force_random=False):
    if np.random.random() < EPSILON or force_random:
        return np.random.randint(N_ACTIONS)

    return q_values.argmax()


def process_state(state):
    state = state[[0, 1, 3]]
    return state.flatten()


def replay():
    state, action, reward, new_state, terminal_flag = replay_memory.sample(BATCH_SIZE)
    q_values = network.predict(state)
    non_terminal_flag = 1 - terminal_flag
    new_q_values = network.predict(new_state[non_terminal_flag])

    target = q_values + np.random.random(q_values.shape) * 0.0001
    target[range(len(target)), action][terminal_flag] = reward[terminal_flag]
    target[range(len(target)), action][non_terminal_flag] = reward[non_terminal_flag] + GAMMA * new_q_values.argmax(1)

    network.learn(state, target)


for i in range(N_EPISODES):
    state = env.reset()
    print('Reset')
    print(env._gridworld.player_position)
    state = process_state(state)
    total_reward = 0

    for j in range(100):
        q_values = network.predict(state.reshape(1, -1))

        action = epsilon_greedy(q_values)
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

    print('Total:', total_reward)
