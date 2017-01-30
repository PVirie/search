import tensorflow as tf
import numpy as np
import cv2
import gen
import third_parties.spatial_transformer as transformer


class ElfCell(tf.nn.rnn_cell.RNNCell):
    """Spatial compare and transformer"""
    """all sizes = (width, height)"""

    def __init__(self, template_size, example_size, kernel_size):
        with tf.variable_scope(type(self).__name__):  # "ElfCell"
            self.template_size = template_size
            self.kernel_size = kernel_size
            self.example_size = example_size
            self.feature_map_numel = np.prod(self.example_size)
            self.state_size_ = (6, (self.feature_map_numel, self.feature_map_numel))
            # thetas and alpha
            # self.output_size_ = 6 + 1
            self.output_size_ = 6

            self.lstm = tf.nn.rnn_cell.LSTMCell(self.feature_map_numel, forget_bias=1.0, state_is_tuple=True, num_proj=6 + self.output_size_)

    @property
    def state_size(self):
        return self.state_size_

    @property
    def output_size(self):
        return self.output_size_

    def init_state(self, batch_size):
        thetas = tf.constant(np.tile(np.asarray([0, 0, 0, 0, 0, 0], dtype=np.float32), (batch_size, 1)), dtype=tf.float32)
        state = self.lstm.zero_state(batch_size, dtype=tf.float32)
        return thetas, state

    def __call__(self, inputs, state, scope=None):
        # split inputs
        template_numel = np.prod(self.template_size)
        salients = tf.reshape(tf.slice(inputs, [0, 0], [-1, template_numel]), [-1, self.template_size[0], self.template_size[1], 1])
        examples = tf.reshape(tf.slice(inputs, [0, template_numel], [-1, self.feature_map_numel]), [-1, self.example_size[0], self.example_size[1], 1])

        # split state, theta [cos, -sin, tx, sin, cos, ty]
        selective_thetas = state[0] + tf.constant([1, 0, 0, 0, 1, 0], dtype=tf.float32)
        lstm_states = state[1]

        # since tensorflow does not support batched filters, I work around the problem by flipping the filters and the inputs and use depthwise instead.
        transposed_examples = tf.transpose(examples, perm=[3, 1, 2, 0])
        # tx, ty = (0, 0) means center
        kernels = transformer.transformer(salients, selective_thetas, self.kernel_size)
        transposed_kernel = tf.transpose(kernels, perm=[1, 2, 0, 3])
        conv_out = tf.nn.depthwise_conv2d(transposed_examples, transposed_kernel, [1, 1, 1, 1], "SAME")
        feature_maps = tf.reshape(tf.transpose(conv_out, [3, 1, 2, 0]), [-1, self.feature_map_numel])
        lstm_output, lstm_out_state = self.lstm(feature_maps, lstm_states)
        output = tf.slice(lstm_output, [0, 6], [-1, self.output_size])
        out_state = (tf.slice(lstm_output, [0, 0], [-1, 6]), lstm_out_state)

        return output, out_state


if __name__ == "__main__":
    with tf.Session() as sess:
        t, g = gen.gen_batch(2, (10, 10), (50, 50))
        cell = ElfCell((10, 10), (50, 50), (10, 10))
        init_state = cell.init_state(2)
        cat = np.concatenate([np.reshape(t, (2, -1)), np.reshape(g, (2, -1))], axis=1)
        rnn_inputs = tf.tile(tf.reshape(tf.constant(cat, dtype=tf.float32), [1, 2, -1]), [5, 1, 1])
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, time_major=True, initial_state=init_state, parallel_iterations=1, swap_memory=True)

        sess.run(tf.global_variables_initializer())
        print sess.run(rnn_outputs, feed_dict={})

        cv2.imshow("tem0", t[0])
        cv2.imshow("tem1", t[1])
        cv2.imshow("gen0", g[0])
        cv2.imshow("gen1", g[1])
        cv2.waitKey(0)
