import gen
import network as ann
import cv2


templates, examples = gen.gen_batch(1, input_size=(20, 20), output_size=(50, 50))

print "Figures >>", templates.shape
print "Examples >>", examples.shape

nn = ann.Network(1, (20, 20), (50, 50), 25)
nn.load_session("../artifacts/" + "test_weight")
drawn, output = nn.draw(templates, examples)
print nn.debug(templates, examples)

print output[0]
cv2.imshow("template", templates[0])
cv2.imshow("example", examples[0])
cv2.imshow("drawn", drawn[0])
cv2.waitKey(0)
