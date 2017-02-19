import tensorflow as tf
import cv2
import numpy as np
import util
import gen
import third_parties.spatial_transformer as transformer


def Invert_Transformer(U, theta, kernel_size, out_size, blur=100, name='InvertTransformer', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(tf.expand_dims(tf.ones(shape=[n_repeats, ]), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, grid, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.reshape(tf.slice(grid, [0, 0, 0], [-1, 1, -1]), [-1])
            y = tf.reshape(tf.slice(grid, [0, 1, 0], [-1, 1, -1]), [-1])
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-width/hgith/2, width/height/2] to [0, width/height]
            x = x + (width_f) / 2.0
            y = y + (height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            # prevent out of bound
            x0 = tf.clip_by_value(x0, 0, max_x)
            x1 = tf.clip_by_value(x1, 0, max_x)
            y0 = tf.clip_by_value(y0, 0, max_y)
            y1 = tf.clip_by_value(y1, 0, max_y)

            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, [-1, channels])
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
            wb = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
            wc = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
            wd = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

            x_t, y_t = tf.meshgrid(tf.linspace(-width / 2.0, width / 2.0, width), tf.linspace(-height / 2.0, height / 2.0, height))
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            grid = tf.concat([x_t_flat, y_t_flat], 0)
            return grid

    def _rotate(theta, input_dim, out_size):
        with tf.variable_scope('_rotate'):
            num_batch = tf.shape(input_dim)[0]
            num_channels = tf.shape(input_dim)[3]
            # R.shape = [numbatch*2, 2]
            R_t = tf.slice(tf.reshape(theta, (-1, 3)), [0, 0], [-1, 2])
            R = tf.reshape(tf.transpose(tf.reshape(R_t, [-1, 2, 2]), [0, 2, 1]), [-1, 2])

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            # Transform A x (x_t, y_t)^T -> (x_s, y_s) of shape [batch*2, -1]
            T_g = tf.reshape(tf.matmul(R, grid), [num_batch, 2, -1])
            input_transformed = _interpolate(input_dim, T_g, out_size)

            output = tf.reshape(input_transformed, [num_batch, out_height, out_width, num_channels])
            return output

    def create_dimension_expander_indicator(kernel, dx, three_sigmas):
        with tf.variable_scope('_expander'):
            v = 2.0 * three_sigmas * three_sigmas / 9.0
            t = tf.exp(-tf.squared_difference(dx, kernel) / v)
            return tf.div(t, tf.reduce_sum(t, axis=2, keep_dims=True))

    def _translate(theta, input_dim, input_size, out_size, blur):
        with tf.variable_scope('_translate'):
            num_batch = tf.shape(input_dim)[0]
            num_channels = tf.shape(input_dim)[3]
            # T.shape = [numbatch, 2]
            Tx = tf.reshape((tf.slice(theta, [0, 2], [-1, 1]) + 1) * (out_size[0] / 2), (-1, 1, 1))
            Ty = tf.reshape((tf.slice(theta, [0, 5], [-1, 1]) + 1) * (out_size[1] / 2), (-1, 1, 1))

            row_expander = tf.constant(util.make_dimension_expander_kernel(out_size[0], input_size[0]), dtype=tf.float32)
            col_expander = tf.constant(util.make_dimension_expander_kernel(out_size[1], input_size[1]), dtype=tf.float32)

            Fy = create_dimension_expander_indicator(row_expander, Ty, blur)
            Fx = create_dimension_expander_indicator(col_expander, Tx, blur)

            batch_channeled = tf.reshape(tf.transpose(input_dim, [0, 3, 1, 2]), [-1, input_size[0], input_size[1]])
            input_transformed = tf.matmul(tf.matmul(Fy, batch_channeled), Fx, transpose_a=False, transpose_b=True)
            output = tf.transpose(tf.reshape(input_transformed, [num_batch, num_channels, out_size[0], out_size[1]]), [0, 2, 3, 1])
            return output

    with tf.variable_scope(name):
        Red = _rotate(theta, U, kernel_size)
        output = _translate(theta, Red, kernel_size, out_size, blur)
        return output


if __name__ == "__main__":
    with tf.Session() as sess:
        inp = util.batch_resize([gen.data[0], gen.data[1]], (40, 40))
        imgs = tf.reshape(tf.constant(inp, dtype=tf.float32), [2, 40, 40, 1])
        thetas = tf.constant(np.asarray([[1, 0, 0, 0, 1, 0], [0.5, -0.85, -0.5, 0.85, 0.5, -0.25]]), dtype=tf.float32)
        transformed = Invert_Transformer(imgs, thetas, (60, 60), (200, 200), 1.0)

        thetas2 = tf.constant(np.asarray([[1, 0, 0, 0, 1, 0], [0.5, -0.85, -0.5, 0.85, 0.5, -0.25]]), dtype=tf.float32)
        backward = transformer.transformer(transformed, thetas2, (40, 40))
        sess.run(tf.global_variables_initializer())
        bak, out = sess.run((backward, transformed), feed_dict={})

        cv2.imshow("tem0", inp[0])
        cv2.imshow("tem1", inp[1])
        cv2.imshow("gen0", out[0])
        cv2.imshow("gen1", out[1])
        cv2.imshow("bak0", bak[0])
        cv2.imshow("bak1", bak[1])
        cv2.waitKey(0)
