import src.gen as gen
import src.network as ann
import cv2
import src.util as util
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--plot", help="run plot", action="store_true")
parser.add_argument("--test", help="run test", action="store_true")
parser.add_argument("--cont", help="continue mode", action="store_true")
parser.add_argument("--rate", help="learning rate", type=float)
parser.add_argument("--pg", help="policy gradient", action="store_true")
parser.add_argument("--dg", help="disconnected gradient", action="store_true")
args = parser.parse_args()

template_size = (20, 20)
canvas_size = (50, 50)
batch_size = 100
steps = 100

nn = ann.Network(batch_size, template_size, canvas_size, steps, {'learning_rate': args.rate, 'disconnected_gradient': args.dg, 'policy_gradient': args.pg})

if args.test:
    nn.load_session("./artifacts/" + "test_weight")

    templates, examples, params = gen.gen_batch(100, input_size=template_size, output_size=canvas_size, blur=0.2)
    drawn, out_params, matches = nn.draw(templates, examples, params)
    print matches
    print params, out_params[len(out_params) - 1]

    ts = util.make_tile(np.reshape(examples, (examples.shape[0], examples.shape[1], examples.shape[2], 1)), 800, 800, False)
    gs = util.make_tile(np.reshape(drawn, (drawn.shape[0], drawn.shape[1], drawn.shape[2], 1)), 800, 800, False)
    cv2.imshow("template", ts)
    cv2.imshow("generated", gs)
    cv2.waitKey(0)

elif args.plot:
    nn.load_session("./artifacts/" + "test_weight")

    templates, examples, params = gen.gen_batch(100, input_size=template_size, output_size=canvas_size, blur=0.2)
    drawns = nn.draw_sequence(templates, examples, params)
    drawns.append(1 - examples)

    ps = util.plot_progress(drawns, selected_ids=[0, 10, 20, 30, 40, 50, 60, 70, 80], display_steps=20, include_last=True)
    cv2.imshow("plotted", ps)

    cv2.waitKey(0)

else:
    if args.cont:
        nn.load_session("./artifacts/" + "test_weight")

    templates, examples, true_values = gen.gen_batch(1000, input_size=template_size, output_size=canvas_size, blur=0.2)

    for step in xrange(100000):
        nn.train(templates, examples, true_values, batch_size=batch_size, blur=1.0)
        if step % 100 == 0:
            nn.save("./artifacts/" + "test_weight")
    nn.save("./artifacts/" + "test_weight")
