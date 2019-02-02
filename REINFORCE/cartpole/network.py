import tensorflow as tf
from utils import logging_helper


class Network:
    def __init__(self, input_dim, output_dim):
        self.state_size = input_dim
        self.action_size = output_dim
        self.output_dim = output_dim

        with tf.name_scope('placeholders'):
            self.state = tf.placeholder(tf.float32, [None, input_dim], name='states')
            self.true_action = tf.placeholder(tf.int32, [None, output_dim], name='actions')
            self.discounted_episode_rewards = tf.placeholder(tf.float32, [None, ], name='discounted_episode_rewards')
            # Add this placeholder for having this variable in tensorboard
            self.mean_reward = tf.placeholder(tf.float32, name='mean_reward')

        self.predicted_action, self.predicted_action_probas = self.build_graph()

    def build_graph(self):
        with tf.name_scope('fc1'):
            fc1 = tf.layers.dense(
                inputs=self.state,
                units=10,
                activation=tf.nn.relu,
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        with tf.name_scope('fc2'):
            fc2 = tf.layers.dense(
                inputs=fc1,
                units=2,
                activation=tf.nn.relu,
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        with tf.name_scope('fc3'):
            fc3 = tf.layers.dense(
                inputs=fc2,
                units=self.action_size,
                # activation=None,
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )

        with tf.name_scope('softmax'):
            action_distribution = tf.nn.softmax(fc3)

        return fc3, action_distribution


class NetworkTrainer:
    def __init__(self, session: tf.Session, network, hyperparams):
        self.session = session
        self.network: Network = network
        self.hyperparams = hyperparams
        self.loss = self.build_loss()
        self.train_op = self.build_optimiser()

    def build_loss(self):
        with tf.name_scope('loss'):
            # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the
            # softmax function If you have single-class labels, where an object can only belong to one class,
            # you might now consider using tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to
            # convert your labels to a dense one-hot array.
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.network.predicted_action,
                labels=self.network.true_action
            )

            loss = tf.reduce_mean(neg_log_prob * self.network.discounted_episode_rewards)

        return loss

    def build_optimiser(self):
        with tf.name_scope('train'):
            train_opt = tf.train.AdamOptimizer(self.hyperparams.LEARNING_RATE).minimize(self.loss)

        return train_opt

    def optimise(self, state, true_action, discounted_episode_rewards):
        feed_dict = {
            self.network.state: state,
            self.network.true_action: true_action,
            self.network.discounted_episode_rewards: discounted_episode_rewards
        }

        _, loss = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def compute_loss(self, state, true_action, discounted_episode_rewards):
        feed_dict = {
            self.network.state: state,
            self.network.true_action: true_action,
            self.network.discounted_episode_rewards: discounted_episode_rewards
        }

        loss = self.session.run(self.loss, feed_dict=feed_dict)
        return loss


class NetworkInterface:
    def __init__(self, session: tf.Session, network):
        self.network = network
        self.session = session
        self.saver = tf.train.Saver()

    def predict(self, x):
        feed_dict = {self.network.state: x}
        res = self.session.run(self.network.predicted_action_probas, feed_dict=feed_dict)
        return res

    def predict_one(self, x):
        return self.predict(x.reshape(1, -1))

    def restore(self, path):
        self.saver.restore(self.session, path)


class NetworkManager:
    def __init__(self, input_dim, output_dim, Network, NetworkTrainer, hyperparams):
        self._logger = logging_helper.get_logger(self.__class__.__name__)
        self.session = tf.Session()

        self.network = Network(input_dim, output_dim)
        self.trainer = NetworkTrainer(self.session, self.network, hyperparams)
        self.network_interface = NetworkInterface(self.session, self.network)

        self.saver = tf.train.Saver()

        # Setup TensorBoard Writer
        self.writer = tf.summary.FileWriter('./tensorboard/pg/1')
        tf.summary.scalar('Loss', self.trainer.loss)
        tf.summary.scalar('Reward_mean', self.network.mean_reward)
        self.write_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        self.session.run(init)

    def predict(self, x):
        return self.network_interface.predict(x)

    def predict_one(self, x):
        return self.predict(x.reshape(1, -1))

    def learn(self, state, true_action, discounted_episode_rewards):
        return self.trainer.optimise(state, true_action, discounted_episode_rewards)

    def save(self, path):
        self.saver.save(self.session, path)
        print('Model saved')

    def restore(self, path):
        self.saver.restore(self.session, path)

    def write_summaries(self, episode_ix, state, true_action, discounted_episode_rewards, mean_reward):
        summary = self.session.run(self.write_op, feed_dict={
            self.network.state: state,
            self.network.true_action: true_action,
            self.network.discounted_episode_rewards: discounted_episode_rewards,
            self.network.mean_reward: mean_reward
        })

        self.writer.add_summary(summary, episode_ix)
        self.writer.flush()


if __name__ == '__main__':
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
