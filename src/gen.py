import os
from random import randint
import random
import math
import util
import numpy as np
import cv2


data = []
template = []
for file in sorted(os.listdir("../data")):
    fullname = os.path.join("../data", file)
    if file.endswith(".png"):
        print "Reading:", fullname
        img = 1 - util.convert_255_2_1(cv2.imread(fullname, cv2.IMREAD_GRAYSCALE))
        if file.startswith("d"):
            data.append(img)
        else:
            template.append(img)


def view_data(index=randint(0, len(data) - 1)):
    cv2.imshow("data", data[index])
    cv2.waitKey(0)


def view_template(index=randint(0, len(template) - 1)):
    cv2.imshow("template", template[index])
    cv2.waitKey(0)


def random_thetas(shape=(1000, 1000), object_shape=(220, 220)):
    sx = random.uniform(0.75, 1.5)
    sy = random.uniform(0.75, 1.5)
    t = random.uniform(-math.pi, math.pi)
    diameter = max(object_shape[0], object_shape[1]) * math.sqrt(2)
    tx = random.uniform(diameter, shape[0] - diameter)
    ty = random.uniform(diameter, shape[1] - diameter)
    return np.array([[math.cos(t) * sx, -math.sin(t) * sy, tx],
                     [math.sin(t) * sx, math.cos(t) * sy, ty]], dtype=np.float32)


def gen_transform(shape, thetas, index=randint(0, len(data) - 1)):
    if thetas is None:
        thetas = random_thetas(shape, data[index].shape)
    return cv2.warpAffine(data[index], thetas, shape)


def view_gen(shape=(1000, 1000), thetas=None):
    imgs = []
    for i in xrange(3):
        imgs.append(gen_transform(shape, thetas, randint(0, len(data) - 1)))
    cv2.imshow("gen", util.blend_images(imgs))
    cv2.waitKey(0)


def gen_batch(batches, input_size=(40, 40), output_size=(200, 200)):
    t_ = []
    g_ = []
    for i in xrange(batches):
        index = randint(0, len(data) - 1)
        t_.append(template[index])
        g_.append(gen_transform((1000, 1000), None, index))
    return util.batch_resize(t_, input_size), util.batch_resize(g_, output_size)


if __name__ == "__main__":
    t, g = gen_batch(100)
    ts = util.make_tile(np.reshape(t, (t.shape[0], t.shape[1], t.shape[2], 1)), 400, 400, False)
    gs = util.make_tile(np.reshape(g, (g.shape[0], g.shape[1], g.shape[2], 1)), 800, 800, False)
    cv2.imshow("template", ts)
    cv2.imshow("generated", gs)
    cv2.waitKey(0)
