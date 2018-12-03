import numpy as np
import random

from collections import deque
from sklearn.neural_network import MLPClassifier

N_EPISODES = 100
MEMORY_SIZE = 100
REPLAY_FREQUENCY = 10
EPSILON = 0.1
N_ACTIONS = 4
BATCH_SIZE = 10
GAMMA = 0.9


class ReplayMemory:
    def __init__(self, max_length):
        self.states = deque(maxlen=max_length)
        self.actions = deque(maxlen=max_length)
        self.rewards = deque(maxlen=max_length)
        self.new_states = deque(maxlen=max_length)
        self.terminal_flags = deque(maxlen=max_length)

    def sample(self, batch_size):
        ix = np.random.choice(len(self.states), batch_size, replace=False)
        states = np.array([self.states[i] for i in ix])
        actions = np.array([self.actions[i] for i in ix])
        rewards = np.array([self.rewards[i] for i in ix])
        new_states = np.array([self.new_states[i] for i in ix])
        terminal_flags = np.array([self.terminal_flags[i] for i in ix])

        return states, actions, rewards, new_states, terminal_flags

    def add(self, state, action, reward, new_state, terminal_flag):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_states.append(new_state)
        self.terminal_flags.append(terminal_flag)

    def __len__(self):
        return len(self.states)


class Network:
    def __init__(self):
        pass

    def predict(self, x):
        pass

    def learn(self, x, y):
        pass


replay_memory = ReplayMemory(MEMORY_SIZE)
network = Network()
env = Env()


def epsilon_greedy(q_values):
    if np.random.random() < EPSILON:
        return np.random.randint(N_ACTIONS)

    return q_values.argmax()


state = env.reset()
for i in range(N_EPISODES):
    q_values = network.predict(state)
    action = epsilon_greedy(q_values)
    new_state, reward, is_terminal = env.take_action(action)
    replay_memory.add(state, action, reward, new_state, is_terminal)

    if len(replay_memory) > BATCH_SIZE:
        state, action, reward, new_state, terminal_flag = replay_memory.sample(BATCH_SIZE)
        q_values = network.predict(state)
        non_terminal_flag = 1 - terminal_flag
        new_q_values = network.predict(new_state[non_terminal_flag])

        target = np.zeros_like(reward)
        target[terminal_flag] = reward[terminal_flag]
        target[non_terminal_flag] = reward[non_terminal_flag] + GAMMA * new_q_values.argmax(1)

        network.learn(state, target)
