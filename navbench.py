import cv2
import numpy as np
import os
import pandas as pd


# Read info for image database entries from CSV file
def read_image_database(path):
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

# Returns a greyscale image of type uint8
def read_image(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    info = np.iinfo(im.dtype)
    return im.astype(np.float) / info.max


def image_diff(im1, im2):
    d = cv2.absdiff(im1, im2)
    return d.mean()


def get_route_idf(images, snap):
    return [image_diff(im, snap) for im in images]


def get_ridf(image, snap):
    diffs = []
    for _ in range(image.shape[1]):
        diffs.append(image_diff(image, snap))
        image = np.roll(image, 1, axis=1)
    return diffs


def get_route_ridf(images, snap):
    return [min(get_ridf(im, snap)) for im in images]

