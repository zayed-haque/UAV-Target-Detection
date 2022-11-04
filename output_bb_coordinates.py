import cv2
import numpy as np
import pyyolo

names_filepath = "obj.names"
cfg_filepath = r"C:\Users\zayed\Downloads\bb_coordinate\yolov3_custom.cfg"
weights_filepath = r"C:\Users\zayed\Downloads\bb_coordinate\yolov3_custom_final.weights"

image_filepath = '1.jpg'

meta = pyyolo.load_names(names_filepath)
net = pyyolo.load_net(cfg_filepath, weights_filepath, 0)

im = cv2.imread(image_filepath)
yolo_img = pyyolo.array_to_image(im)
res = pyyolo.detect(net, meta, yolo_img)
colors = np.random.rand(meta.classes, 3) * 255

for r in res:
    cv2.rectangle(im, r.bbox.get_point(pyyolo.BBox.Location.TOP_LEFT, is_int=True),
                  r.bbox.get_point(pyyolo.BBox.Location.BOTTOM_RIGHT, is_int=True), tuple(colors[r.id].tolist()), 2)

cv2.imshow('Frame', im)
cv2.waitKey(0)

