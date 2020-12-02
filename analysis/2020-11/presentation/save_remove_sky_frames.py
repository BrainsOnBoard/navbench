#!/usr/bin/python3

import os
import sys
sys.path.append('../../..')

import cv2
import numpy as np

import navbench as nb
from navbench import improc as ip

dbpath = '../../../datasets/rc_car/Stanmer_park_dataset/0411/unwrapped_dataset1'

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
    im_nosky = cv2.cvtColor(ip.remove_sky_and_histeq(im_grey), cv2.COLOR_GRAY2BGR)

    # Put images one on top of the other
    combined = np.concatenate((im, im_nosky))

    # Write frame
    assert cv2.imwrite(os.path.join(new_dpath, fname), combined)
