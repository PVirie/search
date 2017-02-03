import tensorflow as tf
import numpy as np
import util


class Layers:

    def __init__(self, layers, activations):
        self.fs = activations
        self.Ws = []
        self.Bs = []
        for i in xrange(1, len(layers)):
            self.Ws.append(tf.Variable(util.xavier_init(layers[i - 1], layers[i]), dtype=tf.float32))
            self.Bs.append(tf.Variable(np.zeros((layers[i])), dtype=tf.float32))

    def __call__(self, input):
        output = input
        for i in xrange(0, len(self.Ws)):
            output = self.fs[i](tf.matmul(output, self.Ws[i]) + self.Bs[i])
        return output


if __name__ == "__main__":
    with tf.Session() as sess:
        layer = Layers([5, 2, 1], [tf.sigmoid, tf.sigmoid])
        y = layer(tf.constant([[0, 1, 0, 1, 0], [1, 1, 1, 1, 1]], dtype=np.float32))
        z = tf.constant([[0.2], [0.8]], dtype=np.float32)
        loss = -tf.reduce_sum(z * tf.log(y) + (1 - z) * tf.log(1 - y))
        opt = tf.train.AdamOptimizer(0.1).minimize(loss)
        sess.run(tf.global_variables_initializer())

        for i in xrange(100):
            print sess.run((opt, loss), feed_dict={})
        print sess.run(y)
