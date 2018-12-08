import tensorflow as tf


class Network:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.x = tf.placeholder(dtype=tf.int64, shape=self.input_shape, name='input_x')
        self.graph = self.build_graph()

    def build_graph(self):
        n_classes = self.output_shape[1]

        layer1 = tf.layers.dense(self.x, 128, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, n_classes, activation=tf.identity)

        return layer2


class Optimiser:
    def __init__(self, graph, x, session: tf.Session, output_shape):
        self.graph = graph
        self.x = x
        self.y = tf.placeholder(dtype=tf.int64, shape=output_shape, name='output_y')
        self.loss = self.get_loss()
        self.train_op = self.build_optimiser()
        self.session = session

    def get_loss(self):
        loss = tf.losses.mean_squared_error(self.y, self.graph)
        return loss

    def build_optimiser(self):
        train_op = tf.train.AdamOptimizer().minimize(self.loss)
        return train_op

    def optimise(self, x, y):
        feed_dict = {
            self.x: x,
            self.y: y
        }

        self.session.run([self.train_op], feed_dict=feed_dict)

    def compute_loss(self, x, y):
        feed_dict = {
            self.x: x,
            self.y: y
        }

        loss = self.session.run(self.loss, feed_dict=feed_dict)
        return loss


class NetworkInterface:
    def __init__(self, graph, x, y, session: tf.Session):
        self.graph = graph
        self.x = x
        self.y = y
        self.session = session

    def predict(self, x):
        feed_dict = {self.x: x}
        res = self.session.run(self.y, feed_dict=feed_dict)
        return res


if __name__ == '__main__':
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
