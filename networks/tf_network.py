import tensorflow as tf
from utils import logging_helper


class Network:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='input_x')
        self.y_pred = self.build_graph(self.x, self.output_dim)

    def build_graph(self, x, output_dim):
        raise NotImplementedError('Please implement me!')


class NetworkTrainer:
    def __init__(self, session: tf.Session, y_pred, x, output_dim):
        self.session = session
        self.y_pred = y_pred
        self.x = x
        self.y_true = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='output_y')
        self.loss = self.build_loss(self.y_true, self.y_pred)
        self.train_op = self.build_optimiser(self.loss)

    def build_loss(self, y_true, y_pred):
        raise NotImplementedError('Please implement me!')

    def build_optimiser(self, loss):
        raise NotImplementedError('Please implement me!')

    def optimise(self, x, y):
        feed_dict = {self.x: x, self.y_true: y}

        self.session.run([self.train_op], feed_dict=feed_dict)

    def compute_loss(self, x, y):
        feed_dict = {self.x: x, self.y_true: y}

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
    def __init__(self, input_dim, output_dim, Network, NetworkTrainer):
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
