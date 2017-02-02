import tensorflow as tf
import numpy as np
import math

# During training, we might want to discard the slowly progress module with a more meaningful sampler from external;
# this script does just that.
# We want to flow the gradient from the cost to this function and bridge the gradient to the module
# by making the module produces the sampled value. I. e., d(cost)/d(module) = (1-cost) * d(sampler - module)/d(module)


# def external_attention(internal, external, alpha):
#     # alpha controls how much we focus on external \in [0, 1]
#     f = external * alpha + internal * (1 - alpha)
#     return internal + tf.stop_gradient(f - internal)
#     # return f

def external_attention_gradient(es, xs, eval):
    er = eval(es)
    xr = eval(xs)
    rs = tf.concat(concat_dim=0, values=[er, xr])

    loss = tf.reduce_sum(tf.squared_difference(e, x), axis=[1], keep_dims=True)
    loss = tf.concat(concat_dim=0, values=[loss, -a])

    g_lx = tf.gradients(loss, params, rs)
    return optimizer.apply_gradients(zip(g_lx, params))


if __name__ == "__main__":

    with tf.Session() as sess:
        x = tf.Variable([[10.0, 10.0, 10.0, 10.0]], dtype=tf.float32, name="X")
        e = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        t = tf.constant([[-0.5, 0.5, 1.0, -1.0]], dtype=tf.float32)

        r = tf.exp(-tf.reduce_sum(tf.squared_difference(e, t) / 2, axis=[1], keep_dims=True))
        a = tf.exp(-tf.reduce_sum(tf.squared_difference(x, t) / 2, axis=[1], keep_dims=True))
        r = tf.concat(concat_dim=0, values=[r, a])

        loss = tf.reduce_sum(tf.squared_difference(e, x) / 2, axis=[1], keep_dims=True)
        loss = tf.concat(concat_dim=0, values=[loss, -a])
        opt = tf.train.AdamOptimizer(0.1)

        params = tf.trainable_variables()
        print ' '.join(str(p.name) for p in params)
        g_lx = tf.gradients(loss, params, r)

        # by default, most optimizer perform negative gradient update (descending)
        training_op = opt.apply_gradients(zip(g_lx, params))
        # training_op = opt.minimize(loss)

        sess.run(tf.global_variables_initializer())

        for i in xrange(1000):
            sess.run((training_op), feed_dict={e: np.random.rand(5, 4) * 10 - 5})
        print sess.run(x)
