import gen
import network as ann

template_size = (20, 20)
canvas_size = (50, 50)
templates, examples, true_values = gen.gen_batch(1000, input_size=template_size, output_size=canvas_size)

print "Figures >>", templates.shape
print "Examples >>", examples.shape
print "Thetas >>", true_values.shape

batch_size = 100
nn = ann.Network(batch_size, template_size, canvas_size, 100)
nn.train(templates, examples, true_values, "../artifacts/" + "test_weight", batch_size=batch_size, max_iteration=10000, continue_from_last=False)
