import cv2
import numpy as np
import os
import pandas as pd


def read_image_database(path):
    """Read info for image database entries from CSV file."""
    df = pd.read_csv(os.path.join(path, "database_entries.csv"))
    df = df.rename(columns=lambda x: x.strip())  # strip whitespace

    entries = {
        "x": df["X [mm]"] / 1000,
        "y": df["Y [mm]"] / 1000,
        "z": df["Z [mm]"] / 1000,
        "heading": df["Heading [degrees]"],
        "filepath": [os.path.join(path, fn.strip()) for fn in df["Filename"]]
    }

    return entries

def read_image(path, size=()):
    """Returns a greyscale image of type float."""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Convert to normalised matrix of floats
    info = np.iinfo(im.dtype)
    im = im.astype(np.float) / info.max

    return cv2.resize(im, size) if size else im



def read_images(paths, size=()):
    if not paths:
        return []

    im0 = read_image(paths[0], size)
    images = np.zeros([im0.shape[0], im0.shape[1], len(paths)], dtype=np.float)
    images[:, :, 0] = im0
    for i in range(1, len(paths)):
        images[:, :, i] = read_image(paths[i], size)
    return images


def mean_absdiff(x, y):
    """Return mean absolute difference between two images or sets of images
    (as 3D matrices). """

    # Check that the images are of the same size
    assert x.shape[0:2] == y.shape[0:2]

    # If the dimensions don't match up then we need to add an extra fake
    # dimension. Eww.
    if x.ndim == 2 and y.ndim == 3:
        x = x[:, :, np.newaxis]
    elif x.ndim == 3 and y.ndim == 2:
        y = y[:, :, np.newaxis]

    absdiffs = np.abs(x - y)
    return absdiffs.mean(axis=(0, 1))

def get_route_idf(images, snap):
    return mean_absdiff(images, snap)


def get_ridf(image, snap, step=1):
    assert step > 0

    nsteps = image.shape[1] // step
    cols = image.shape[2] if image.ndim == 3 else 1
    diffs = np.zeros([nsteps, cols], dtype=np.float)
    for i in range(nsteps):
        diffs[i, :] = mean_absdiff(image, snap)
        snap = np.roll(snap, -step, axis=1)

    return diffs


def get_route_ridf(images, snap, step=1):
    return np.amin(get_ridf(images, snap, step), axis=0)
