import tensorflow as tf
import cell
import math
import numpy as np
import inv_transformer as putter
import third_parties.spatial_transformer as transformer


def expand_last_dim(t):
    return np.reshape(t, (t.shape[0], t.shape[1], t.shape[2], 1))


class Network():

    def __init__(self, batches, template_size, canvas_size, total_steps=10):
        self.sess = tf.Session()
        self.template_numel = template_size[0] * template_size[1]
        self.canvas_numel = canvas_size[0] * canvas_size[1]
        self.gpu_templates = tf.placeholder(tf.float32, [None, template_size[0], template_size[1], 1])
        self.gpu_examples = tf.placeholder(tf.float32, [None, canvas_size[0], canvas_size[1], 1])
        self.blur = tf.placeholder(tf.float32)
        temp = self.gpu_templates + 1e-6
        norm_template = temp / tf.reduce_sum(temp, axis=[1, 2, 3], keep_dims=True)

        sqr_step = int(math.sqrt(total_steps))
        sx = template_size[0] * 1.0 / canvas_size[0]
        sy = template_size[1] * 1.0 / canvas_size[1]
        dx = 2 * (1 - sx) / sqr_step
        dy = 2 * (1 - sy) / sqr_step

        print sx, sy, dx, dy

        with tf.variable_scope("model"):
            self.cell = cell.AffineCell()
            state = self.cell.init_state(batches)
            input = self.cell.init_input(batches)
            self.total_match = 0
            self.matches = []
            for i in xrange(sqr_step):
                for j in xrange(sqr_step):
                    if i > 0 or j > 0:
                        tf.get_variable_scope().reuse_variables()
                    output, state = self.cell(tf.stop_gradient(input), state)
                    gen = transformer.transformer(self.gpu_examples, output, template_size) + 1e-6
                    norm_gen = gen / (tf.reduce_sum(gen, axis=[1, 2, 3], keep_dims=True))
                    # match = tf.exp(tf.reduce_mean(-tf.squared_difference(self.gpu_templates, gen), axis=[1, 2, 3]) / 0.1)
                    match = tf.reduce_mean(norm_template * tf.log(norm_gen) + norm_gen * tf.log(norm_template), axis=[1, 2, 3])
                    step = tf.tile(tf.constant([[1.0, 0, sx + dx * j, 0, 1.0, sy + dy * i]], dtype=tf.float32), [batches, 1])
                    input = tf.concat(1, [tf.reshape(match, [-1, 1]), step])
                    self.matches.append(match)
                    self.total_match = self.total_match + tf.reduce_sum(match)

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")

        self.training_op = tf.train.AdamOptimizer(0.001).minimize(-self.total_match, var_list=scope)
        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

        self.outputs = output
        self.gen = putter.Invert_Transformer(self.gpu_templates, self.outputs, template_size, canvas_size, self.blur)

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
                _, loss = self.sess.run((self.training_op, self.total_match), feed_dict={self.gpu_templates: expand_last_dim(tb), self.gpu_examples: expand_last_dim(eb), self.blur: blur})
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
        drawn, output = self.sess.run((self.gen, self.outputs), feed_dict={self.gpu_templates: expand_last_dim(templates), self.gpu_examples: expand_last_dim(examples), self.blur: 1.0})
        return drawn, output

    def debug(self, templates, examples):
        matches = self.sess.run((self.matches), feed_dict={self.gpu_templates: expand_last_dim(templates), self.gpu_examples: expand_last_dim(examples), self.blur: 1.0})
        return matches
