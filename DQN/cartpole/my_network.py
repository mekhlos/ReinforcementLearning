import tensorflow as tf
from networks import network
from DQN.gridworld1 import my_network as gridworld_network
from utils import logging_helper


class MyNetwork(network.Network):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='input_x')
        self.y_pred = self.build_graph()

    def build_graph(self):
        layer1 = tf.layers.dense(self.x, 64, activation=tf.nn.relu, name='l_1')
        layer2 = tf.layers.dropout(layer1, 0.2, name='l_2')
        layer3 = tf.layers.dense(layer2, 16, activation=tf.nn.relu, name='l_3')
        layer4 = tf.layers.dropout(layer3, 0.2, name='l_4')
        layer5 = tf.layers.dense(layer4, self.output_dim, activation=tf.identity, name='l_out')

        return layer5


class MyNetworkManager:
    def __init__(self, input_dim, output_dim, model_path, tensorboard_path, is_train_mode):
        self._logger = logging_helper.get_logger(self.__class__.__name__)

        self.session = tf.Session()
        self.network = MyNetwork(input_dim, output_dim)
        self.network_interface = gridworld_network.MyNetworkInterface(self.session, self.network)

        self.saver = tf.train.Saver()

        if is_train_mode:
            self.trainer = gridworld_network.MyNetworkTrainer(self.session, self.network)
            self.summary_manager = gridworld_network.MySummaryManager(
                self.session, self.network, self.trainer, tensorboard_path)
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

    def write_summaries(self, global_step, x, y, mean_reward):
        self.summary_manager.write_summaries(global_step, x, y, mean_reward)
