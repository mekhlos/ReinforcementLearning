import numpy as np

from collections import deque


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
