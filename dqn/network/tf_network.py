import tensorflow as tf
from utils import logging_helper


class Network:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='input_x')
        self.y_pred = self.build_graph()

    def build_graph(self):
        layer1 = tf.layers.dense(self.x, 1024, activation=tf.nn.tanh, trainable=True, use_bias=True, name='l_1')
        layer2 = tf.layers.dropout(layer1, 0.2, training=True, name='l_2')
        # layer3 = tf.layers.dense(layer2, 512, activation=tf.nn.tanh, trainable=True, use_bias=True, name='l_3')
        # layer4 = tf.layers.dropout(layer3, 0.2, training=True, name='l_4')
        layer5 = tf.layers.dense(layer2, self.output_dim, activation=tf.nn.tanh, use_bias=True, trainable=True,
                                 name='l_out')

        return layer5


class NetworkTrainer:
    def __init__(self, session: tf.Session, y_pred, x, output_dim):
        self.session = session
        self.y_pred = y_pred
        self.x = x
        self.y_true = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='output_y')
        self.loss = self.get_loss()
        self.train_op = self.build_optimiser()

    def get_loss(self):
        loss = tf.losses.mean_squared_error(self.y_true, self.y_pred)
        return loss

    def build_optimiser(self):
        train_op = tf.train.RMSPropOptimizer(1e-3).minimize(self.loss)
        return train_op

    def optimise(self, x, y):
        feed_dict = {
            self.x: x,
            self.y_true: y
        }

        self.session.run([self.train_op], feed_dict=feed_dict)

    def compute_loss(self, x, y):
        feed_dict = {
            self.x: x,
            self.y_true: y
        }

        loss = self.session.run(self.loss, feed_dict=feed_dict)
        return loss


class NetworkInterface:
    def __init__(self, session: tf.Session, x, y_pred):
        self.x = x
        self.y_pred = y_pred
        self.session = session

    def predict(self, x):
        feed_dict = {self.x: x}
        res = self.session.run(self.y_pred, feed_dict=feed_dict)
        return res


class TensorflowSaver:
    def __init__(self):
        pass


class NetworkManager:
    def __init__(self, input_dim, output_dim):
        self._logger = logging_helper.get_logger(self.__class__.__name__)
        self.session = tf.Session()
        self.network = Network(input_dim, output_dim)
        self.trainer = NetworkTrainer(self.session, self.network.y_pred, self.network.x, output_dim)
        self.network_interface = NetworkInterface(self.session, self.network.x, self.network.y_pred)
        init = tf.global_variables_initializer()

        self.session.run(init)

    def predict(self, x):
        return self.network_interface.predict(x)

    def predict_one(self, x):
        return self.predict(x.reshape(1, -1))

    def learn(self, x, y):
        self.trainer.optimise(x, y)
        return self.trainer.compute_loss(x, y)


if __name__ == '__main__':
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
