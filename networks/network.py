import tensorflow as tf
from utils import logging_helper


class Network:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams

    def build_graph(self):
        raise NotImplementedError('Please implement me!')

    def get_placeholder_dict(self):
        raise NotImplementedError('Please implement me!')

    def get_input_names(self):
        raise NotImplementedError('Please implement me!')


class NetworkTrainer:
    def __init__(self, session, network, hyperparams):
        self.session: tf.Session = session
        self.hyperparams = hyperparams
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


class NetworkManager:
    def __init__(self, model_path, tensorboard_path, is_train_mode, hyperparams):
        self._logger = logging_helper.get_logger(self.__class__.__name__)

        self.session = tf.Session()
        self.hyperparams = hyperparams
        self.tensorboard_path = tensorboard_path

        self.network = self.add_network()
        self.network_interface = self.add_network_interface()

        self.saver = tf.train.Saver()

        if is_train_mode:
            self.trainer = self.add_trainer()
            self.summary_manager = self.add_summary_writer()
        else:
            self.restore(model_path)

        init = tf.global_variables_initializer()
        self.session.run(init)

    def add_network(self):
        raise NotImplementedError('Please implement me!')

    def add_trainer(self):
        raise NotImplementedError('Please implement me!')

    def add_summary_writer(self):
        raise NotImplementedError('Please implement me!')

    def add_network_interface(self):
        raise NotImplementedError('Please implement me!')

    def predict(self, x):
        return self.network_interface.predict(x)

    def predict_one(self, x):
        return self.predict(x.reshape(1, -1))

    def learn(self, *args, **kwargs):
        raise NotImplementedError('Please implement me!')

    def save(self, path):
        self.saver.save(self.session, path)
        print('Model saved')

    def restore(self, path):
        self.saver.restore(self.session, path)
        print('Model restored')

    def write_summaries(self, *args, **kwargs):
        raise NotImplementedError('Please implement me!')
