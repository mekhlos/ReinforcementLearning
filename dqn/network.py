import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError


class Network:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(128, n_actions),
            max_iter=1,
            solver='sgd',
            activation='identity',
            # verbose=10,
            learning_rate_init=1e-3,
            warm_start=True
        )

    def predict(self, x):
        try:
            return self.mlp.predict(x)
        except NotFittedError:
            return np.random.standard_normal((len(x), self.n_actions)) * 0.01

    def learn(self, x, y):
        self.mlp.fit(x, y)
