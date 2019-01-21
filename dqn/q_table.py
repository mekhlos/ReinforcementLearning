import numpy as np


class QTable:
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.table = np.zeros((n_states, n_actions))
        self.state_to_ix_map = {}

    def state_to_ix(self, state):
        assert len(state.shape) == 1
        v = tuple(state[:self.n_states])
        if v not in self.state_to_ix_map:
            self.state_to_ix_map[v] = len(self.state_to_ix_map)

        return self.state_to_ix_map[v]

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

    def __repr__(self):
        return str(self.table)


if __name__ == '__main__':
    q_table = QTable(4, 3)
    q_table.update(np.array([0, 0, 0, 1]), 2, 12.1)
    print(q_table.get(2, 2))
    print(q_table.get(3, 2))
    print(q_table[3, 2])
