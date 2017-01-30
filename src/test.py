import gen
import network as ann
import cv2


templates, examples = gen.gen_batch(2, input_size=(20, 20), output_size=(50, 50))

print "Figures >>", templates.shape
print "Examples >>", examples.shape

nn = ann.Network(2, (20, 20), (50, 50), 20)
nn.load_session("../artifacts/" + "test_weight")
drawn = nn.draw(templates, examples)

cv2.imshow("template", templates[0])
cv2.imshow("example", examples[0])
cv2.imshow("drawn", drawn[0])
cv2.waitKey(0)
