import tensorflow as tf
from networks import tf_network


class MyNetwork(tf_network.Network):
    def build_graph(self, x, output_dim):
        layer1 = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=True, use_bias=True, name='l_1')
        layer2 = tf.layers.dropout(layer1, 0.2, training=True, name='l_2')
        layer3 = tf.layers.dense(layer2, 16, activation=tf.nn.relu, trainable=True, use_bias=True, name='l_3')
        layer4 = tf.layers.dropout(layer3, 0.2, training=True, name='l_4')
        layer5 = tf.layers.dense(layer4, output_dim, activation=tf.identity, use_bias=True, trainable=True,
                                 name='l_out')

        return layer5


class MyNetworkTrainer(tf_network.NetworkTrainer):

    def build_loss(self, y_true, y_pred):
        loss = tf.losses.mean_squared_error(y_true, y_pred)
        return loss

    def build_optimiser(self, loss):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        return train_op
