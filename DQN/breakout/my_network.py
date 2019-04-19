import tensorflow as tf
from networks import network
from DQN.gridworld import my_network as gridworld_network
from DQN.breakout.env_config import Config1 as env_config


class MyNetwork(network.Network):

    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.input_dim = hyperparams.input_dim
        self.output_dim = hyperparams.output_dim

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='input_x')
        self.image_in = tf.reshape(self.x, shape=[-1, env_config.height, env_config.width, 3])
        self.y_pred = self.build_graph()

    def build_graph(self):
        # layer1 = tf.keras.layers.Dense(self.hyperparams.n_hidden_layer1, activation=tf.nn.relu, name='l_1')(self.x)
        # layer2 = tf.keras.layers.Dropout(0.2, name='l_2')(layer1)
        # layer3 = tf.keras.layers.Dense(self.hyperparams.n_hidden_layer2, activation=tf.nn.relu, name='l_3')(layer2)
        # layer4 = tf.keras.layers.Dropout(0.2, name='l_4')(layer3)
        # layer5 = tf.keras.layers.Dense(self.output_dim, activation=tf.identity, name='l_out')(layer4)
        pool_size = 2
        pool_stride = 2

        conv0 = tf.layers.average_pooling2d(
            self.image_in,
            pool_size=pool_size,
            strides=pool_stride
        )

        conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[8, 8],
            strides=[4, 4],
            padding='valid',
            activation=tf.nn.elu
        )(conv0)

        conv1b = tf.layers.average_pooling2d(
            conv1,
            pool_size=pool_size,
            strides=pool_stride
        )

        conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[4, 4],
            strides=[2, 2],
            padding='valid',
            activation=tf.nn.elu
        )(conv1b)

        conv2b = tf.layers.average_pooling2d(
            conv2,
            pool_size=pool_size,
            strides=pool_stride
        )

        hidden = tf.keras.layers.Dense(
            units=2,
            activation=tf.nn.elu
        )(tf.layers.flatten(conv2b))

        return hidden


class MyNetworkManager(gridworld_network.MyNetworkManager):

    def add_network(self):
        return MyNetwork(self.hyperparams)
