import tensorflow as tf
import numpy as np


class AffineCell(tf.nn.rnn_cell.RNNCell):
    """Spatial compare and transformer"""
    """Output = previous_output + alpha*gradient_input + sigma"""

    def __init__(self):
        with tf.variable_scope(type(self).__name__):  # "AffineCell"
            self.lstm_state_size = 1 + 6 + 6
            self.state_size_ = (6, (self.lstm_state_size, self.lstm_state_size))
            self.output_size_ = 6

            # internal lstm produces sigma + alpha
            self.lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_state_size, forget_bias=1.0, state_is_tuple=True, num_proj=6 + 1)

    @property
    def state_size(self):
        return self.state_size_

    @property
    def output_size(self):
        return self.output_size_

    def init_state(self, batch_size):
        thetas = tf.constant(np.tile(np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32), (batch_size, 1)), dtype=tf.float32)
        state = self.lstm.zero_state(batch_size, dtype=tf.float32)
        return thetas, state

    def __call__(self, inputs, state, scope=None):
        # split inputs, value, grad, step
        grad = tf.slice(inputs, [0, 1], [-1, 6])

        # split state
        previous = state[0]
        lstm_states = state[1]

        lstm_output, lstm_out_state = self.lstm(inputs, lstm_states)

        sigma = tf.slice(lstm_output, [0, 0], [-1, 6])
        alpha = tf.slice(lstm_output, [0, 6], [-1, 1])

        # The activation of lstm is in range [-1, 1], which I tries to expand...
        output = previous + alpha * grad + sigma * 2
        return output, (output, lstm_out_state)


if __name__ == "__main__":
    with tf.Session() as sess:
        cell = AffineCell()
        init_state = cell.init_state(2)
        rnn_inputs = tf.constant(np.random.rand(5, 2, 1 + 6 + 6), dtype=tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, time_major=True, initial_state=init_state, parallel_iterations=1, swap_memory=True)

        sess.run(tf.global_variables_initializer())
        print sess.run(rnn_outputs, feed_dict={})
