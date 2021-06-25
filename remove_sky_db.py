#!/usr/bin/python3
# Remove sky from an image database and dump result in folder

import os
import sys

import cv2
import navbench as nb
from navbench import imgproc as ip

for dbpath in sys.argv[1:]:
    db = nb.Database(dbpath)
    head, tail = os.path.split(dbpath)
    if tail == '':
        _, tail = os.path.split(head)
    new_dpath = tail + '_nosky'
    if not os.path.exists(new_dpath):
        os.mkdir(new_dpath)

    for fpath in db.filepath:
        _, fname = os.path.split(fpath)
        im = cv2.imread(fpath)
        assert im.shape

        # Remove sky and histeq
        im_grey = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im_nosky = ip.remove_sky_and_histeq(im_grey)

        # Write frame
        assert cv2.imwrite(os.path.join(new_dpath, fname), im_nosky)
