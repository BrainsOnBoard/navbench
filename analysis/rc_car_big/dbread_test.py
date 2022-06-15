import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import os.path
import rc_car_big


# def get_images():
#     with sqlite3.connect(f"file:{os.path.dirname(__file__)}/database_cache.db?mode=ro", uri=True) as con:
#         train_df = pd.read_sql_query(
#             # 'SELECT * FROM train_data WHERE (database_idx % 80) = 0', conn)
#             'SELECT * FROM train_data WHERE (database_idx % 80) = 0', con)
#         # train_df = pd.read_sql_table('train_data', conn)
#         print(len(train_df))
#         # return list(zip(train_df['im_height'], train_df['im_width'])),
#         # train_df['image'].to_list()
#         return train_df


t0 = perf_counter()
# im_size, images = get_images()
paths = rc_car_big.get_paths()
db_cache = rc_car_big.DatabaseCache()
entries = db_cache.get_entries(paths[0], range(10), (180, 720), 'ip.histeq')
img = entries.iloc[0].image

# images = train_df[['im_height', 'im_width', 'image']].apply(
#     lambda row: np.frombuffer(row.image, dtype=np.uint8).reshape(row.im_height, row.im_width), axis=1)
# img = images[0]
print(f'Elapsed: {perf_counter() - t0}')

plt.imshow(img)
plt.show()
