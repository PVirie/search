import gen
import network as ann
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cont", help="continue mode", action="store_true")
parser.add_argument("--rate", help="learning rate", type=float)
parser.add_argument("--pg", help="policy gradient", action="store_true")
parser.add_argument("--dg", help="disconnected gradient", action="store_true")
args = parser.parse_args()

template_size = (20, 20)
canvas_size = (50, 50)
batch_size = 100

templates, examples, true_values = gen.gen_batch(1000, input_size=template_size, output_size=canvas_size, blur=0.2)
nn = ann.Network(batch_size, template_size, canvas_size, 100, {'learning_rate': args.rate, 'disconnected_gradient': args.dg, 'policy_gradient': args.pg})
if args.cont:
    nn.load_session("../artifacts/" + "test_weight")

for step in xrange(100000):
    nn.train(templates, examples, true_values, batch_size=batch_size, blur=1.0)
    if step % 100 == 0:
        nn.save("../artifacts/" + "test_weight")
nn.save("../artifacts/" + "test_weight")
