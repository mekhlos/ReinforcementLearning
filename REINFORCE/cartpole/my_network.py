import tensorflow as tf
from networks import network
from utils import logging_helper


class MyNetwork(network.Network):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

        self.input_dim = hyperparams.input_dim
        self.output_dim = hyperparams.output_dim

        with tf.name_scope('placeholders'):
            self.state = tf.placeholder(tf.float32, [None, self.input_dim], name='states')

        self.predicted_action, self.predicted_action_probas = self.build_graph()

    def build_graph(self):
        with tf.name_scope('fc1'):
            fc1 = tf.layers.dense(
                inputs=self.state,
                units=self.hyperparams.n_hidden_layer1,
                activation=tf.nn.relu,
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        with tf.name_scope('fc2'):
            fc2 = tf.layers.dense(
                inputs=fc1,
                units=self.hyperparams.n_hidden_layer2,
                activation=tf.nn.relu,
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(
                inputs=fc2,
                units=self.output_dim,
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        with tf.name_scope('softmax'):
            action_distribution = tf.nn.softmax(fc3)

        return fc3, action_distribution


class MyNetworkTrainer(network.NetworkTrainer):

    def __init__(self, session, network, hyperparams):
        self.true_action = tf.placeholder(tf.int32, [None, hyperparams.output_dim], name='actions')
        self.discounted_episode_rewards = tf.placeholder(tf.float32, [None, ], name='discounted_episode_rewards')

        super().__init__(session, network, hyperparams)

    def build_loss(self):
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.network.predicted_action,
                labels=self.true_action
            )

            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards)

        return loss

    def build_optimiser(self):
        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(self.hyperparams.learning_rate).minimize(self.loss)

        return train_op

    def build_feed_dict(self, state, true_action, discounted_episode_rewards):
        return {
            self.network.state: state,
            self.true_action: true_action,
            self.discounted_episode_rewards: discounted_episode_rewards
        }


class MySummaryManager(network.SummaryManager):
    def __init__(self, session, network, trainer, tensorboard_path):
        self.mean_reward = tf.placeholder(tf.float32, name='mean_reward')
        super().__init__(session, network, trainer, tensorboard_path)

    def build_feed_dict(self, state, true_action, discounted_episode_rewards, mean_reward):
        return {
            self.network.state: state,
            self.trainer.true_action: true_action,
            self.trainer.discounted_episode_rewards: discounted_episode_rewards,
            self.mean_reward: mean_reward
        }

    def build_summaries(self):
        tf.summary.scalar('Loss', self.trainer.loss)
        tf.summary.scalar('Reward_mean', self.mean_reward)


class MyNetworkInterface(network.NetworkInterface):
    def build_feed_dict(self, x):
        return {self.network.state: x}

    def get_output_variables(self):
        return self.network.predicted_action_probas


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

    def learn(self, state, true_action, discounted_episode_rewards):
        return self.trainer.optimise(state, true_action, discounted_episode_rewards)

    def write_summaries(self, global_step, state, true_action, discounted_episode_rewards, mean_reward):
        self.summary_manager.write_summaries(global_step, state, true_action, discounted_episode_rewards, mean_reward)
