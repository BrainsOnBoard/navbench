#!/usr/bin/python3
# Remove sky from an image database and dump result in folder

import os
import sys

import cv2
import navbench as nb
from navbench import improc as ip

for dbpath in sys.argv[1:]:
    db = nb.Database(dbpath)
    head, tail = os.path.split(dbpath)
    if tail == '':
        _, tail = os.path.split(head)
    new_dpath = tail + '_nosky'
    if not os.path.exists(new_dpath):
        os.mkdir(new_dpath)

    for fpath in db.entries['filepath']:
        _, fname = os.path.split(fpath)
        im = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        assert im.shape
        assert cv2.imwrite(os.path.join(new_dpath, fname), ip.remove_sky(im))
