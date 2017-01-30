import tensorflow as tf
import cell
import numpy as np
import inv_transformer as putter


def expand_last_dim(t):
    return np.reshape(t, (t.shape[0], t.shape[1], t.shape[2], 1))


class Network():

    def __init__(self, batches, template_size, canvas_size, iterations=10):
        self.sess = tf.Session()
        self.template_numel = template_size[0] * template_size[1]
        self.canvas_numel = canvas_size[0] * canvas_size[1]
        self.gpu_templates = tf.placeholder(tf.float32, [None, template_size[0], template_size[1], 1])
        self.gpu_examples = tf.placeholder(tf.float32, [None, canvas_size[0], canvas_size[1], 1])
        self.blur = tf.placeholder(tf.float32)

        with tf.variable_scope("elf"):
            self.cell = cell.ElfCell(template_size, canvas_size, template_size)
            self.init_state = self.cell.init_state(batches)
            cat = tf.concat(1, [tf.reshape(self.gpu_templates, [batches, self.template_numel]), tf.reshape(self.gpu_examples, [batches, self.canvas_numel])])
            rnn_inputs = tf.tile(tf.reshape(cat, [1, batches, -1]), [iterations, 1, 1])
            rnn_outputs, final_state = tf.nn.dynamic_rnn(self.cell, rnn_inputs, time_major=True, initial_state=self.init_state, parallel_iterations=1, swap_memory=True)

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="elf")

        pivot = tf.constant([1, 0, 0, 0, 1, 0], dtype=tf.float32)
        self.outputs = tf.reshape(tf.slice(rnn_outputs, [iterations - 1, 0, 0], [1, -1, -1]), [batches, -1]) + pivot
        self.gen = putter.Invert_Transformer(self.gpu_templates, self.outputs, template_size, canvas_size, self.blur)
        # self.overall_cost = tf.reduce_sum(-tf.mul(self.gpu_examples, tf.log(self.gen)))
        self.overall_cost = tf.reduce_sum(tf.squared_difference(self.gpu_examples, self.gen))

        self.training_op = tf.train.AdamOptimizer(0.001).minimize(self.overall_cost, var_list=scope)
        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

        # Launch the graph.
        self.sess.run(tf.global_variables_initializer())

    def train(self, templates, examples, session_name, batch_size, max_iteration, continue_from_last=False):
        if continue_from_last:
            self.load_session(session_name)

        for step in xrange(max_iteration):
            sum_loss = 0.0
            total_batches = templates.shape[0] / batch_size
            blur = 100.0 * (1 - step * 1.0 / max_iteration) + 5
            for b in xrange(total_batches):
                tb = templates[(b * batch_size):((b + 1) * batch_size), ...]
                eb = examples[(b * batch_size):((b + 1) * batch_size), ...]
                _, loss = self.sess.run((self.training_op, self.overall_cost), feed_dict={self.gpu_templates: expand_last_dim(tb), self.gpu_examples: expand_last_dim(eb), self.blur: blur})
                sum_loss += loss
            print sum_loss / total_batches
            if step % 100 == 0:
                self.saver.save(self.sess, session_name)

        self.saver.save(self.sess, session_name)

    def load_session(self, session_name):
        print "loading from last save..."
        self.saver.restore(self.sess, session_name)

    def load_last(self, directory):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))

    def draw(self, templates, examples):
        drawn = self.sess.run(self.gen, feed_dict={self.gpu_templates: expand_last_dim(templates), self.gpu_examples: expand_last_dim(examples), self.blur: 1.0})
        return drawn
