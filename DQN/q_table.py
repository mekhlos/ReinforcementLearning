import numpy as np
import pickle
from pathlib import Path


def pickle_save(content, path):
    with open(path, 'wb') as f:
        pickle.dump(content, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class QTable:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.table = np.zeros((n_states, n_actions))
        self.state_to_ix_map = {}

    def state_to_ix2(self, state):
        assert len(state.shape) == 1
        v = tuple(state[:self.n_states])
        if v not in self.state_to_ix_map:
            self.state_to_ix_map[v] = len(self.state_to_ix_map)

        return self.state_to_ix_map[v]

    def state_to_ix(self, state):
        n = int(self.n_states ** 0.5)
        state = state[:self.n_states].reshape(n, n).T.flatten()
        assert len(state.shape) == 1
        return state.argmax()

    def update(self, state, action, value):
        ix = self.state_to_ix(state)
        self.table[ix, action] = value

    def get(self, state, action):
        return self.table[state, action]

    def __getitem__(self, item):
        ix = self.state_to_ix(item[0])
        return self.table[ix, item[1]]

    def get_q_values_for_state(self, state):
        ix = self.state_to_ix(state)
        return self.table[ix]

    def predict_one(self, state):
        return self.get_q_values_for_state(state)

    def save(self, path):
        pickle_save(self.table, Path(path).joinpath('q_table.pickle').resolve().as_posix())
        pickle_save(self.state_to_ix_map, Path(path).joinpath('state_to_ix_map.pickle').resolve())

    def load(self, path):
        self.table = pickle_load(Path(path).joinpath('q_table.pickle').resolve().as_posix())
        self.state_to_ix_map = pickle_load(Path(path).joinpath('state_to_ix_map.pickle').resolve())

    def __repr__(self):
        return str(self.table)


if __name__ == '__main__':
    q_table = QTable(4, 3)
    q_table.update(np.array([0, 0, 0, 1]), 2, 12.1)
    print(q_table.get(2, 2))
    print(q_table.get(3, 2))
    print(q_table[3, 2])
