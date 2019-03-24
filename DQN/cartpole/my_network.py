import tensorflow as tf
from networks import network
from DQN.gridworld import my_network as gridworld_network


class MyNetwork(network.Network):

    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.input_dim = hyperparams.input_dim
        self.output_dim = hyperparams.output_dim

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='input_x')
        self.y_pred = self.build_graph()

    def build_graph(self):
        layer1 = tf.keras.layers.Dense(self.hyperparams.n_hidden_layer1, activation=tf.nn.relu, name='l_1')(self.x)
        layer2 = tf.keras.layers.Dropout(0.2, name='l_2')(layer1)
        layer3 = tf.keras.layers.Dense(self.hyperparams.n_hidden_layer2, activation=tf.nn.relu, name='l_3')(layer2)
        layer4 = tf.keras.layers.Dropout(0.2, name='l_4')(layer3)
        layer5 = tf.keras.layers.Dense(self.output_dim, activation=tf.identity, name='l_out')(layer4)

        return layer5


class MyNetworkManager(gridworld_network.MyNetworkManager):

    def add_network(self):
        return MyNetwork(self.hyperparams)
