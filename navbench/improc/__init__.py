import cv2
import numpy as np

def histeq(im):
    '''Histogram equalisation: Requires uint8s as input'''
    return cv2.equalizeHist(im)

def resize(width, height):
    '''Return a function to resize image to given size'''
    return lambda im: cv2.resize(im, (width, height))

def chain(*args):
    '''Return a function which chains the outputs of provided functions'''
    def chainedfun(im):
        for fun in args:
            im = fun(im)
        return im
    return chainedfun

def to_float(im):
    # Normalise values
    info = np.iinfo(im.dtype)
    return im.astype(np.float) / info.max
