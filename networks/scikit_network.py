import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import NotFittedError


class NetworkManager:
    def __init__(self, input_dim, output_dim):
        self.n_actions = output_dim
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(1024, output_dim),
            max_iter=1,
            solver='adam',
            activation='tanh',
            # verbose=10,
            learning_rate_init=1e-3,
            warm_start=True
        )

    def predict(self, x):
        try:
            return self.mlp.predict(x)
        except NotFittedError:
            return np.random.standard_normal((len(x), self.n_actions)) * 0.01

    def predict_one(self, x):
        return self.predict(x.reshape(1, -1))

    def learn(self, x, y):
        self.mlp.fit(x, y)
        return 0
