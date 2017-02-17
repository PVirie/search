import gen
import network as ann
import cv2
import util
import numpy as np


templates, examples, params = gen.gen_batch(100, input_size=(20, 20), output_size=(50, 50))

print "Figures >>", templates.shape
print "Examples >>", examples.shape

nn = ann.Network(100, (20, 20), (50, 50), 100)
nn.load_session("../artifacts/" + "test_weight")
drawn = nn.draw(templates, examples, params)

ts = util.make_tile(np.reshape(examples, (examples.shape[0], examples.shape[1], examples.shape[2], 1)), 800, 800, False)
gs = util.make_tile(np.reshape(drawn, (drawn.shape[0], drawn.shape[1], drawn.shape[2], 1)), 800, 800, False)
cv2.imshow("template", ts)
cv2.imshow("generated", gs)
cv2.waitKey(0)
