import gen
import network as ann
import cv2
import util
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cont", help="continue mode", action="store_true")
parser.add_argument("--rate", help="learning rate", type=float)
parser.add_argument("--pg", help="policy gradient", action="store_true")
parser.add_argument("--dg", help="disconnected gradient", action="store_true")
args = parser.parse_args()

templates, examples, params = gen.gen_batch(100, input_size=(20, 20), output_size=(50, 50))

print "Figures >>", templates.shape
print "Examples >>", examples.shape

nn = ann.Network(100, (20, 20), (50, 50), 100, {'learning_rate': args.rate, 'disconnected_gradient': args.dg, 'policy_gradient': args.pg})
nn.load_session("../artifacts/" + "test_weight")
drawn, out_params, matches = nn.draw(templates, examples, params)
print matches
print params, out_params

ts = util.make_tile(np.reshape(examples, (examples.shape[0], examples.shape[1], examples.shape[2], 1)), 800, 800, False)
gs = util.make_tile(np.reshape(drawn, (drawn.shape[0], drawn.shape[1], drawn.shape[2], 1)), 800, 800, False)
cv2.imshow("template", ts)
cv2.imshow("generated", gs)
cv2.waitKey(0)
