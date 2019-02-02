import tensorflow as tf
from networks import network
from utils import logging_helper


class MyNetwork(network.Network):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='input_x')
        self.y_pred = self.build_graph()

    def build_graph(self):
        layer1 = tf.layers.dense(self.x, 1024, activation=tf.nn.tanh, trainable=True, use_bias=True, name='l_1')
        layer2 = tf.layers.dropout(layer1, 0.2, training=True, name='l_2')
        layer3 = tf.layers.dense(layer2, self.output_dim, activation=tf.nn.tanh, use_bias=True, trainable=True,
                                 name='l_out')

        return layer3


class MyNetworkTrainer(network.NetworkTrainer):

    def __init__(self, session, network):
        self.y_true = tf.placeholder(dtype=tf.float32, shape=[None, network.output_dim], name='y_true')
        super().__init__(session, network)

    def build_loss(self):
        loss = tf.losses.mean_squared_error(self.y_true, self.network.y_pred)
        return loss

    def build_optimiser(self):
        train_op = tf.train.RMSPropOptimizer(1e-3).minimize(self.loss)
        return train_op

    def build_feed_dict(self, x, y):
        return {
            self.network.x: x,
            self.y_true: y
        }


class MySummaryManager(network.SummaryManager):
    def __init__(self, session, network, trainer, tensorboard_path):
        self.mean_reward = tf.placeholder(tf.float32, name='mean_reward')
        super().__init__(session, network, trainer, tensorboard_path)

    def build_feed_dict(self, x, y, mean_reward):
        return {
            self.network.x: x,
            self.trainer.y_true: y,
            self.mean_reward: mean_reward
        }

    def build_summaries(self):
        tf.summary.scalar('Loss', self.trainer.loss)
        tf.summary.scalar('Reward_mean', self.mean_reward)


class MyNetworkInterface(network.NetworkInterface):
    def build_feed_dict(self, x):
        return {self.network.x: x}

    def get_output_variables(self):
        return self.network.y_pred


class MyNetworkManager:
    def __init__(self, input_dim, output_dim, model_path, tensorboard_path, is_train_mode):
        self._logger = logging_helper.get_logger(self.__class__.__name__)

        self.session = tf.Session()
        self.network = MyNetwork(input_dim, output_dim)
        self.network_interface = MyNetworkInterface(self.session, self.network)

        self.saver = tf.train.Saver()

        if is_train_mode:
            self.trainer = MyNetworkTrainer(self.session, self.network)
            self.summary_manager = MySummaryManager(self.session, self.network, self.trainer, tensorboard_path)
        else:
            self.restore(model_path)

        init = tf.global_variables_initializer()
        self.session.run(init)

    def predict(self, x):
        return self.network_interface.predict(x)

    def predict_one(self, x):
        return self.predict(x.reshape(1, -1))

    def learn(self, x, y):
        return self.trainer.optimise(x, y)

    def save(self, path):
        self.saver.save(self.session, path)
        print('Model saved')

    def restore(self, path):
        self.saver.restore(self.session, path)
        print('Model restored')

    def write_summaries(self, x, y, mean_reward, global_step):
        self.summary_manager.write_summaries(global_step, x, y, mean_reward)
