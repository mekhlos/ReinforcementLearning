import numpy as np


class ExplorationStrategy:
    def __init__(self, n_reductions, epsilon_init, final_epsilon):
        self.epsilon = epsilon_init
        self.n_reductions = n_reductions
        self.final_epsilon = final_epsilon

    def update_epsilon(self):
        x = (self.epsilon - self.final_epsilon) / self.n_reductions
        self.epsilon = max(self.epsilon - x, self.final_epsilon)

    def epsilon_greedy(self, q_values, force_random=False):
        if np.random.random() < self.epsilon or force_random:
            return np.random.randint(q_values.shape[-1])

        return q_values.argmax()
