import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant=0.1):
    """ Xavier initialization of network weights"""
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class Layers:

    def __init__(self, layers, activations):
        self.fs = activations
        self.Ws = []
        self.Bs = []
        self.input_bias = tf.Variable(np.zeros((layers[0])), dtype=tf.float32)
        for i in xrange(1, len(layers)):
            self.Ws.append(tf.Variable(xavier_init(layers[i - 1], layers[i]), dtype=tf.float32))
            self.Bs.append(tf.Variable(np.zeros((layers[i])), dtype=tf.float32))

    def __call__(self, input):
        output = input
        for i in xrange(0, len(self.Ws)):
            output = self.fs[i](tf.matmul(output, self.Ws[i]) + self.Bs[i])
        return output
