import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

"""We use PIL for representing images."""


def convert_255_2_1(img):
    return img.astype(np.float32) / 255.0


def convert_1_2_255(img):
    _ = img * 255
    return _.astype(np.uint8)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def image_to_FC1(img):
    return rgb2gray(convert_255_2_1(img))


def image_to_invFC1(img):
    return 1.0 - rgb2gray(convert_255_2_1(img))


def plot_1D(mat):
    plt.plot(xrange(mat.shape[0]), mat, 'ro')
    plt.ion()
    plt.show()


def cvtColorGrey2RGB(mat):
    last_dim = len(mat.shape)
    return np.repeat(np.expand_dims(mat, last_dim), 3, last_dim)


def batch_resize(imgs, size):
    # scale = (width, height)
    # imgs has shape (batch, height, width)
    out = np.empty((len(imgs), size[1], size[0]), dtype=np.float32)
    for i in xrange(len(imgs)):
        out[i, ...] = cv2.resize(imgs[i], size)
    return out


def make_tile(mat, rows, cols, flip):
    b = mat.shape[0]
    r = mat.shape[2] if flip else mat.shape[1]
    c = mat.shape[1] if flip else mat.shape[2]
    canvas = np.zeros((rows, cols, 3 if len(mat.shape) > 3 else 1), dtype=mat.dtype)
    step = int(max(1, math.floor(b * (r * c) / (rows * cols))))
    i = 0
    for x in xrange(int(math.floor(rows / r))):
        for y in xrange(int(math.floor(cols / c))):
            canvas[(x * r):((x + 1) * r), (y * c):((y + 1) * c), :] = np.transpose(mat[i, ...], (1, 0, 2)) if flip else mat[i, ...]
            i = (i + step) % b

    return canvas


def blend_images(mats):
    out = np.zeros_like(mats[0])
    for i in xrange(len(mats)):
        out = np.clip(out + mats[i], 0, 1)
    return out


def save_txt(mat, name):
    np.savetxt(name, mat, delimiter=",", fmt="%.2e")


def make_dimension_expander_kernel(output_size, input_size):
    t = np.empty((output_size, input_size), dtype=np.float32)
    for i in xrange(output_size):
        for j in xrange(input_size):
            t[i, j] = i - j + (input_size - 1) * 0.5
    return t


def create_dimension_expander_indicator(kernel, dx, three_sigmas):
    v = 2.0 * three_sigmas * three_sigmas / 9.0
    t = np.exp(- (kernel - dx)**2 / v)
    return t / np.sum(t, axis=0)


if __name__ == "__main__":
    np.set_printoptions(precision=2, linewidth=255)
    print "Run test util.py"
    ary = np.array([255, 150, 255], dtype=np.uint8)
    print ary
    ary_1 = convert_255_2_1(ary)
    print ary_1
    print convert_1_2_255(ary_1)
    print create_dimension_expander_indicator(make_dimension_expander_kernel(10, 5), 5, 1)
