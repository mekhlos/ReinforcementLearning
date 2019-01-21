import tensorflow as tf
from networks import tf_network


class MyNetwork(tf_network.Network):

    def build_graph(self, x, output_dim):
        layer1 = tf.layers.dense(x, 1024, activation=tf.nn.tanh, trainable=True, use_bias=True, name='l_1')
        layer2 = tf.layers.dropout(layer1, 0.2, training=True, name='l_2')
        layer3 = tf.layers.dense(layer2, output_dim, activation=tf.nn.tanh, use_bias=True, trainable=True, name='l_out')

        return layer3


class MyNetworkTrainer(tf_network.NetworkTrainer):

    def build_loss(self, y_true, y_pred):
        loss = tf.losses.mean_squared_error(y_true, y_pred)
        return loss

    def build_optimiser(self, loss):
        train_op = tf.train.RMSPropOptimizer(1e-3).minimize(loss)
        return train_op
