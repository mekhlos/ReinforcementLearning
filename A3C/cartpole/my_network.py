import tensorflow as tf
from networks import network
import numpy as np


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class MyNetwork(network.Network):
    def __init__(self, network_name, hyperparams):
        super().__init__(hyperparams)

        self.network_name = network_name
        self.input_dim = hyperparams.input_dim
        self.output_dim = hyperparams.output_dim
        with tf.variable_scope(self.network_name):
            self.state = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32)
            self.build_graph()
            self.policy, self.value = self.build_graph()

    def build_graph(self):
        with tf.variable_scope(self.network_name):
            layer1 = tf.keras.layers.Dense(64, activation=tf.nn.relu, name='l_1')(self.state)
            layer2 = tf.keras.layers.Dropout(0.2, name='l_2')(layer1)
            layer3 = tf.keras.layers.Dense(16, activation=tf.nn.relu, name='l_3')(layer2)
            layer4 = tf.keras.layers.Dropout(0.2, name='l_4')(layer3)

            policy = tf.layers.dense(
                layer4,
                self.output_dim,
                activation=tf.nn.softmax,
                kernel_initializer=normalized_columns_initializer(0.01),
            )

            value = tf.layers.dense(
                layer4,
                1,
                kernel_initializer=normalized_columns_initializer(1.0),
            )

            return policy, value


class MyNetworkTrainer(network.NetworkTrainer):

    def __init__(self, session, network, hyperparams):
        # Only the worker network need ops for loss functions and gradient updating.
        if network.network_name != 'global':
            self.true_action = tf.placeholder(shape=[None, hyperparams.output_dim], dtype=tf.int32)
            self.true_value = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            self.responsible_outputs = tf.reduce_sum(self.network.policy * self.true_action, [1])

        super().__init__(session, network, hyperparams)

    def build_loss(self):
        # Loss functions
        # self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
        value_loss = tf.losses.mean_squared_error(self.true_value, tf.reshape(self.network.value, [-1]))
        entropy = - tf.reduce_sum(self.network.policy * tf.log(self.network.policy))
        policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
        loss = 0.5 * value_loss + policy_loss - entropy * 0.01
        return loss

    def build_optimiser(self):
        with tf.name_scope('train'):
            optimiser = tf.train.AdamOptimizer(self.hyperparams.learning_rate)

            # Get gradients from local network using local losses
            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.network.network_name)
            gradients = tf.gradients(self.loss, local_vars)
            var_norms = tf.global_norm(local_vars)
            grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)

            # Apply local gradients to global network
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            train_op = optimiser.apply_gradients(zip(grads, global_vars))

        return train_op

    def build_feed_dict(self, state, true_action, discounted_episode_rewards):
        pass


class MySummaryManager(network.SummaryManager):
    def __init__(self, session, network, trainer, tensorboard_path):
        self.mean_reward = tf.placeholder(tf.float32, name='mean_reward')
        super().__init__(session, network, trainer, tensorboard_path)

    def build_feed_dict(self, state, true_action, mean_reward):
        return {
            self.network.state: state,
            self.trainer.true_action: true_action,
            self.mean_reward: mean_reward
        }

    def build_summaries(self):
        tf.summary.scalar('Loss', self.trainer.loss)
        tf.summary.scalar('Reward_mean', self.mean_reward)


class MyNetworkInterface(network.NetworkInterface):
    def build_feed_dict(self, x):
        return {self.network.state: x}

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
