import cv2
import numpy as np


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


def histeq(im):
    '''Histogram equalisation: Requires uint8s as input'''
    return cv2.equalizeHist(im)


def to_float(im):
    # Normalise values
    info = np.iinfo(im.dtype)
    return im.astype(np.float) / info.max


def fill_holes(im):
    '''Fills in any black objects'''
    assert im.dtype == np.uint8

    # Find polygons
    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return

    # Find the biggest polygon...
    biggest_idx = np.argmax([cv2.contourArea(cnt) for cnt in contours])

    #...drop it from list and colour it white
    biggest_cnt = contours.pop(biggest_idx)
    cv2.drawContours(im, [biggest_cnt], 0, 255, -1)

    # Colour remaining polygons black
    for cnt in contours:
        cv2.drawContours(im, [cnt], 0, 0, -1)

    return im


def threshold_otsu(im):
    _, thresh = cv2.threshold(
        im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def get_ground_mask(im):
    # First, roughly threshold image
    im = threshold_otsu(im)

    # Set bottom row to white so that any black bits below skyline will be
    # filled in
    im[-1, :] = 255

    # Fill any black holes
    im = fill_holes(im)

    # Fill white holes, by inverting, filling then inverting again
    im = cv2.bitwise_not(im)
    im = fill_holes(im)
    return cv2.bitwise_not(im)


def remove_sky(im):
    '''Sets the sky to 0'''
    mask = get_ground_mask(im)
    im = cv2.bitwise_and(im, mask)
    return im
