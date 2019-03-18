import tensorflow as tf
from networks import network
from utils import logging_helper
import numpy as np


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class MyNetwork(network.Network):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        self.input_dim = hyperparams.input_dim
        self.output_dim = hyperparams.output_dim
        self.build_graph()

    def build_graph(self):
        scope = 'global'
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32)

            layer1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='l_1')(self.inputs)
            layer2 = tf.keras.layers.Dropout(0.2, name='l_2')(layer1)
            layer3 = tf.keras.layers.Dense(16, activation=tf.nn.relu, name='l_3')(layer2)
            layer4 = tf.keras.layers.Dropout(0.2, name='l_4')(layer3)

            # Output layers for policy and value estimations
            self.policy = tf.layers.dense(
                layer4,
                self.output_dim,
                activation=tf.nn.softmax,
                kernel_initializer=normalized_columns_initializer(0.01),
            )
            self.value = tf.layers.dense(
                layer4,
                1,
                kernel_initializer=normalized_columns_initializer(1.0),
            )


class MyNetworkTrainer(network.NetworkTrainer):

    def __init__(self, session, network, hyperparams):
        super().__init__(session, network, hyperparams)

    def build_loss(self):
        pass

    def build_optimiser(self):
        pass

    def build_feed_dict(self, state, true_action, discounted_episode_rewards):
        pass


class MySummaryManager(network.SummaryManager):
    def __init__(self, session, network, trainer, tensorboard_path):
        super().__init__(session, network, trainer, tensorboard_path)
        pass

    def build_feed_dict(self, *args, **kwargs):
        pass

    def build_summaries(self):
        pass


class MyNetworkInterface(network.NetworkInterface):
    def build_feed_dict(self, x):
        return {self.network.inputs: x}

    def get_output_variables(self):
        return self.network.policy


class MyNetworkManager(network.NetworkManager):

    def add_network(self):
        return MyNetwork(self.hyperparams)

    def add_trainer(self):
        return MyNetworkTrainer(self.session, self.network, self.hyperparams)

    def add_summary_writer(self):
        return MySummaryManager(
            self.session,
            self.network,
            self.trainer,
            self.tensorboard_path
        )

    def add_network_interface(self):
        return MyNetworkInterface(self.session, self.network)

    def learn(self, *args, **kwargs):
        pass

    def write_summaries(self, *args, **kwargs):
        pass
