import tensorflow as tf
import cell
import math
import numpy as np
import inv_transformer as putter
import third_parties.spatial_transformer as transformer


def expand_last_dim(t):
    return np.reshape(t, (t.shape[0], t.shape[1], t.shape[2], 1))


def independent_normal_distribution(xs, means, variances):
    return tf.exp(-tf.reduce_sum(tf.squared_difference(xs, means) / (2 * variances), axis=[1])) / (tf.sqrt(tf.reduce_prod(2 * math.pi * variances, axis=[1])))


def compute_pixel_match(templates, template_sum, examples, thetas, size):
    gen = transformer.transformer(examples, thetas, size)
    gen_sum = tf.reduce_sum(gen, axis=[1, 2, 3]) + 1e-8
    match = tf.reduce_sum(templates * gen, axis=[1, 2, 3]) / (template_sum * gen_sum)
    return match


class Network():

    def __init__(self, batches, template_size, canvas_size, total_steps=10):
        self.sess = tf.Session()
        self.template_numel = template_size[0] * template_size[1]
        self.canvas_numel = canvas_size[0] * canvas_size[1]
        self.gpu_true = tf.placeholder(tf.float32, [None, 6])
        self.gpu_templates = tf.placeholder(tf.float32, [None, template_size[0], template_size[1], 1])
        self.gpu_examples = tf.placeholder(tf.float32, [None, canvas_size[0], canvas_size[1], 1])
        self.template_sum = tf.reduce_sum(self.gpu_templates, axis=[1, 2, 3])
        self.blur = tf.placeholder(tf.float32)

        with tf.variable_scope("model"):
            self.cell = cell.AffineCell()
            state = self.cell.init_state(batches)
            means = tf.constant([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
            variances = tf.constant([[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]], dtype=np.float32)
            weights = tf.constant([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
            total_matches = 0
            total_likelihood = 0
            for i in xrange(total_steps):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                output = means + tf.random_normal([batches, 6], 0.0, 1.0, dtype=tf.float32) * variances
                match = compute_pixel_match(self.gpu_templates, self.template_sum, self.gpu_examples, output, template_size)
                match_means = compute_pixel_match(self.gpu_templates, self.template_sum, self.gpu_examples, means, template_size)
                # match = 1 - tf.reduce_mean(tf.squared_difference(output, self.gpu_true) * weights, axis=[1])
                # match_means = 1 - tf.reduce_mean(tf.squared_difference(means, self.gpu_true) * weights, axis=[1])
                likelihood = tf.stop_gradient(tf.maximum(match - match_means, 0.0)) * independent_normal_distribution(tf.stop_gradient(output), means, variances)
                input = tf.concat([tf.reshape(tf.stop_gradient(match), [-1, 1]), output], 1)

                output_tuple, state = self.cell(input, state)
                means = output_tuple[0]
                variances = output_tuple[1]

                # total_matches = total_matches + tf.reduce_sum(match) * float(i) / total_steps
                total_matches = tf.reduce_sum(match)
                total_likelihood = total_likelihood + tf.reduce_sum(likelihood) * float(i) / total_steps
                # total_likelihood = tf.reduce_sum(likelihood)

            self.total_matches = total_matches
            self.total_likelihood = total_likelihood

        scope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="model")
        print "Total variable", [v.name for v in scope]

        self.training_op = tf.train.AdamOptimizer(0.001).minimize(-self.total_likelihood, var_list=scope)
        # self.training_op = tf.train.AdamOptimizer(0.001).minimize(-self.total_matches, var_list=scope)
        self.saver = tf.train.Saver(var_list=scope, keep_checkpoint_every_n_hours=1)

        self.outputs = output
        self.gen = putter.Invert_Transformer(self.gpu_templates, self.outputs, template_size, canvas_size, self.blur)

        # Launch the graph.
        self.sess.run(tf.global_variables_initializer())

    def train(self, templates, examples, true_values, session_name, batch_size, max_iteration, continue_from_last=False):
        if continue_from_last:
            self.load_session(session_name)

        for step in xrange(max_iteration):
            sum_loss = 0.0
            total_batches = templates.shape[0] / batch_size
            blur = 100.0 * (1 - step * 1.0 / max_iteration) + 5
            for b in xrange(total_batches):
                tb = templates[(b * batch_size):((b + 1) * batch_size), ...]
                eb = examples[(b * batch_size):((b + 1) * batch_size), ...]
                vb = true_values[(b * batch_size):((b + 1) * batch_size), ...]
                _, loss, out = self.sess.run((self.training_op, self.total_matches, self.outputs), feed_dict={self.gpu_templates: expand_last_dim(tb), self.gpu_examples: expand_last_dim(eb), self.gpu_true: vb, self.blur: blur})
                sum_loss += loss
            print out[0]
            print sum_loss / total_batches
            if step % 100 == 0:
                self.saver.save(self.sess, session_name)

        self.saver.save(self.sess, session_name)

    def load_session(self, session_name):
        print "loading from last save..."
        self.saver.restore(self.sess, session_name)

    def load_last(self, directory):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(directory))

    def draw(self, templates, examples, true_values):
        drawn, values = self.sess.run((self.gen, self.outputs), feed_dict={self.gpu_templates: expand_last_dim(templates), self.gpu_examples: expand_last_dim(examples), self.gpu_true: true_values, self.blur: 1.0})
        return drawn, values


if __name__ == "__main__":
    with tf.Session() as sess:
        xs = tf.random_normal([20, 6], 0.0, 1.0, dtype=tf.float32)
        means = tf.zeros([20, 6], dtype=tf.float32)
        variances = tf.ones([20, 6], dtype=tf.float32)
        sess.run(tf.global_variables_initializer())
        print sess.run(independent_normal_distribution(xs, means, variances))
