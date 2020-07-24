# Try to remove sky from a bunch of images and dump result in folder

import cv2
import navbench as nb
from navbench import improc as ip

for i in range(0, 1000, 100):
    path = "databases/bottom_of_campus/straight_route2_fwd/frame%05i.jpg" % i
    print(path)

    im = nb.read_images(path)
    thresh = ip.remove_sky(im)
    assert cv2.imwrite("examples/no_sky/frame%05i.jpg" % i, thresh)
