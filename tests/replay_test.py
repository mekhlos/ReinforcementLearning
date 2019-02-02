from DQN import trainer
import numpy as np

n = 5
terminal_flag = np.random.random_integers(0, 1, n).astype(int)


class MockMemory:
    def sample(self, n):
        # state, action, reward, new_state, terminal_flag
        return np.ones((n, 7)), \
               np.ones(n, dtype=int) * 2, \
               np.ones(n) * 10, \
               np.ones((n, 7)), \
               terminal_flag


class MockNetwork:
    def learn(self, x, y):
        pass

    def predict(self, x):
        return np.ones((len(x), 4))


trainer.batch_replay(MockNetwork(), MockMemory(), n, 0.9)
trainer.iterative_replay(MockNetwork(), MockMemory(), n, 0.9)
