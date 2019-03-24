import tensorflow as tf
from networks import network


class MyNetwork(network.Network):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)
        self.input_dim = hyperparams.input_dim
        self.output_dim = hyperparams.output_dim

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_dim], name='input_x')
        self.y_pred = self.build_graph()

    def build_graph(self):
        layer1 = tf.layers.dense(self.x, self.hyperparams.n_hidden_layer1, activation=tf.nn.tanh, name='layer1')
        layer2 = tf.layers.dropout(layer1, 0.2, name='layer2')
        layer3 = tf.layers.dense(layer2, self.output_dim, activation=tf.nn.tanh, name='layer3')

        return layer3


class MyNetworkTrainer(network.NetworkTrainer):

    def __init__(self, session, network, hyperparams):
        self.y_true = tf.placeholder(dtype=tf.float32, shape=[None, hyperparams.output_dim], name='y_true')
        super().__init__(session, network, hyperparams)

    def build_loss(self):
        loss = tf.losses.mean_squared_error(self.y_true, self.network.y_pred)
        return loss

    def build_optimiser(self):
        train_op = tf.train.RMSPropOptimizer(self.hyperparams.learning_rate).minimize(self.loss)
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

    def learn(self, x, y):
        return self.trainer.optimise(x, y)

    def write_summaries(self, global_step, x, y, mean_reward):
        self.summary_manager.write_summaries(global_step, x, y, mean_reward)
