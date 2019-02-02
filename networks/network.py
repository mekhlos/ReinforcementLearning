import tensorflow as tf
from utils import logging_helper


class Network:
    def build_graph(self):
        raise NotImplementedError('Please implement me!')

    def get_placeholder_dict(self):
        raise NotImplementedError('Please implement me!')

    def get_input_names(self):
        raise NotImplementedError('Please implement me!')


class NetworkTrainer:
    def __init__(self, session, network):
        self.session: tf.Session = session
        self.network: Network = network
        self.loss = self.build_loss()
        self.train_op = self.build_optimiser()

    def build_loss(self):
        raise NotImplementedError('Please implement me!')

    def build_optimiser(self):
        raise NotImplementedError('Please implement me!')

    def build_feed_dict(self, *args, **kwargs):
        raise NotImplementedError('Please implement me!')

    def optimise(self, *args, **kwargs):
        feed_dict = self.build_feed_dict(*args, **kwargs)
        _, loss = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def compute_loss(self, *args, **kwargs):
        feed_dict = self.build_feed_dict(*args, **kwargs)
        return self.session.run(self.loss, feed_dict=feed_dict)


class SummaryManager:
    def __init__(self, session, network, trainer, tensorboard_path):
        self.session: tf.Session = session
        self.network: Network = network
        self.trainer: NetworkTrainer = trainer
        self.writer = tf.summary.FileWriter(tensorboard_path)
        self.summaries = self.build_summaries()
        self.write_op = tf.summary.merge_all()

    def build_summaries(self):
        raise NotImplementedError('Please implement me!')

    def build_feed_dict(self, *args, **kwargs):
        raise NotImplementedError('Please implement me!')

    def write_summaries(self, global_step, *args, **kwargs):
        feed_dict = self.build_feed_dict(*args, **kwargs)
        summary = self.session.run(self.write_op, feed_dict=feed_dict)
        self.writer.add_summary(summary, global_step)
        self.writer.flush()


class NetworkInterface:
    def __init__(self, session, network):
        self.network: Network = network
        self.session: tf.Session = session
        self.saver = tf.train.Saver()

    def build_feed_dict(self, *args, **kwargs):
        raise NotImplementedError('Please implement me!')

    def get_output_variables(self):
        raise NotImplementedError('Please implement me!')

    def predict(self, *args, **kwargs):
        feed_dict = self.build_feed_dict(*args, **kwargs)
        output_variables = self.get_output_variables()
        return self.session.run(output_variables, feed_dict=feed_dict)

    def predict_one(self, x):
        return self.predict(x.reshape(1, -1))

    def restore(self, path):
        self.saver.restore(self.session, path)

# class NetworkManager:
#     def __init__(self, model_path, tensorboard_path, is_train_mode):
#         self._logger = logging_helper.get_logger(self.__class__.__name__)
#         self.saver = tf.train.Saver()
#
#         self.network = MyNetwork(input_dim, output_dim)
#         self.network_interface = MyNetworkInterface(self.session, self.network)
#         self.session = tf.Session()
#
#         if is_train_mode:
#             self.trainer = MyNetworkTrainer(self.session, self.network, hyperparams)
#             self.summary_manager = MySummaryManager(self.session, self.network, self.trainer, tensorboard_path)
#         else:
#             self.restore(model_path)
#
#         init = tf.global_variables_initializer()
#         self.session.run(init)
#
#     def predict(self, *args, **kwargs):
#         raise NotImplementedError('Please implement me!')
#
#     def predict_one(self, *args, **kwargs):
#         raise NotImplementedError('Please implement me!')
#
#     def learn(self, *args, **kwargsy):
#         raise NotImplementedError('Please implement me!')
#
#     def save(self, path):
#         self.saver.save(self.session, path)
#         print('Model saved')
#
#     def restore(self, path):
#         self.saver.restore(self.session, path)
#         print('Model restored')
#
#     def write_summaries(self, *args, **kwargs):
#         raise NotImplementedError('Please implement me!')
