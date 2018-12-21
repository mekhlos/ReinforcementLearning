import tensorflow as tf
from dqn.brain import tf_network
import numpy as np


def one_hot(x, depth):
    res = np.zeros((x.shape[0], depth))
    res[range(x.shape[0]), x] = 1
    return res


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train[0][::2, ::2])
print(x_train[1][::2, ::2])
print(x_train[2][::2, ::2])

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

y_train = one_hot(y_train, 10)
y_test = one_hot(y_test, 10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

nm = tf_network.NetworkManager(x_train.shape[1], y_train.shape[1])

for i in range(1000):
    ix = np.random.choice(x_train.shape[0], 128, replace=False)
    x_batch = x_train[ix]
    y_batch = y_train[ix]
    loss = nm.learn(x_batch, y_batch)

    if i % 100 == 0:
        print(loss)

    if i % 200 == 0:
        ix = np.random.choice(x_test.shape[0], 128, replace=False)
        x_batch = x_test[ix]
        y_batch = y_test[ix]

        y_pred = nm.predict(x_batch)
        y1 = y_pred.argmax(1)
        y2 = y_batch.argmax(1)

        print(y1[:10])
        print(y2[:10])

        print('%', sum(y1 == y2) / len(y1))
