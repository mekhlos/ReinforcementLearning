import tensorflow as tf
from utils import logging_helper


class Network:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.past_trajectories = tf.placeholder(
            dtype=tf.float32, shape=[None, self.input_dim], name='past_trajectories')
        self.recent_trajectory = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim],
                                                name='recent_trajectory')
        self.current_state = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='current_state')

        e_char = self.build_char_net()
        e_mental = self.build_mental_net(e_char)
        res = self.build_prediction_net(e_char, e_mental)

    def build_char_net(self):
        e_char = tf.layers.dense(self.past_trajectories, 100)
        return e_char

    def build_mental_net(self, e_char):
        input_x = tf.concat([e_char, self.recent_trajectory])
        e_mental = tf.layers.dense(input_x, 100)
        return e_mental

    def build_prediction_net(self, e_char, e_mental):
        input_x = tf.concat([e_char, e_mental, self.current_state])
        res = tf.layers.dense(input_x, 100)
        return res


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
