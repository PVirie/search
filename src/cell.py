import tensorflow as tf
import layers
import numpy as np
import util


class AffineCell(tf.contrib.rnn.RNNCell):

    def __init__(self):
        self.lstm_state_size = 100
        self.state_size_ = (self.lstm_state_size, self.lstm_state_size)
        self.output_size_ = (6, 6)

        self.layers = layers.Layers([7, self.lstm_state_size], [tf.tanh])
        self.projection_layers = layers.Layers([self.lstm_state_size, 6 + 6], [tf.tanh])

        # internal lstm produces mean + var
        self.lstm = tf.contrib.rnn.LSTMCell(self.lstm_state_size, initializer=tf.random_normal_initializer(0, 0.01), forget_bias=1.0, state_is_tuple=True)
        # self.rnn = tf.contrib.rnn.MultiRNNCell([self.lstm] * 2, state_is_tuple=True)
        self.rnn = self.lstm

    @property
    def state_size(self):
        return self.state_size_

    @property
    def output_size(self):
        return self.output_size_

    def init_state(self, batch_size):
        state = self.rnn.zero_state(batch_size, dtype=tf.float32)
        return state

    def __call__(self, inputs, state, scope=None):

        lstm_output, lstm_out_state = self.rnn(self.layers(inputs), state)
        projected_output = self.projection_layers(lstm_output)
        out_mean = tf.slice(projected_output, [0, 0], [-1, 6]) * 0.2 + tf.slice(inputs, [0, 1], [-1, 6])
        out_var = tf.slice(projected_output, [0, 6], [-1, 6]) * 0.2 + tf.constant([[0.201, 0.201, 0.201, 0.201, 0.201, 0.201]], dtype=tf.float32)

        return (out_mean, out_var), lstm_out_state


if __name__ == "__main__":
    with tf.Session() as sess:
        cell = AffineCell()
        init_state = cell.init_state(2)
        rnn_inputs = tf.random_normal([20, 2, 7], 0.0, 1.0, dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, time_major=True, initial_state=init_state, parallel_iterations=1, swap_memory=True)

        sess.run(tf.global_variables_initializer())
        out = sess.run(rnn_outputs, feed_dict={})
        print out
        print len(out), out[0].shape
