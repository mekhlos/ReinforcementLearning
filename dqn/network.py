import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import NotFittedError


class Network:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.mlp = MLPClassifier(hidden_layer_sizes=(10, n_actions), max_iter=1, alpha=1e-3,
                                 solver='adam', verbose=10, tol=1e-3,
                                 learning_rate_init=.1, warm_start=True)

    def predict(self, x):
        try:
            return self.mlp.predict(x)
        except NotFittedError:
            return np.zeros((len(x), self.n_actions))

    def learn(self, x, y):
        self.mlp.fit(x, y)
