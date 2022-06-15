import pandas as pd
import sqlite3
import rc_car_big
from time import perf_counter
import bob_robotics.navigation as bobnav
from bob_robotics.navigation import imgproc as ip
import os
import numpy as np
import matplotlib.pyplot as plt


IM_SIZE = (45, 180)
DB_PATH = os.path.join(os.path.dirname(__file__), 'navbench_cache.db')


def load_db(path):
    preprocess = ip.resize(*IM_SIZE)
    db = bobnav.Database(path)
    database_idx = range(0, len(db)) #, 80)
    images = db.read_images(entries=database_idx, preprocess=preprocess)
    return db, database_idx, images


def write_db(db, database_idx, images):
    with sqlite3.connect(DB_PATH) as con:
        df = pd.DataFrame(data={'database_idx': database_idx, 'image': images})
        df['database'] = db.name  # TODO: Make this the actual path
        df['im_height'] = IM_SIZE[0]
        df['im_width'] = IM_SIZE[1]
        df['preprocess'] = 'None'
        df.to_sql('images', con, if_exists='replace', index=False)

def unpack_image(row):
    return np.frombuffer(row.image, dtype=np.uint8).reshape(row.im_height, row.im_width)

def read_db(path):
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(f'SELECT im_height, im_width, image FROM images WHERE database="{os.path.basename(path)}" ORDER BY database_idx', con)
    return df.apply(unpack_image, axis=1).to_list()

if __name__ == '__main__':
    paths = rc_car_big.get_paths()

    t0 = perf_counter()
    write_db(paths[0])
    print(f'Elapsed: {perf_counter() - t0}')

    t0 = perf_counter()
    images = read_db(paths[0])
    print(f'Elapsed: {perf_counter() - t0}')

    plt.imshow(images[0])
    plt.show()
