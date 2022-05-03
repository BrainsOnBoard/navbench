import rc_car_big
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths)

labels = ('Invalid', 'GPS fix', 'dGPS fix', 'PPS fix', 'RTK', 'FRTK')  #, '(no data)')
for db in dbs:
    _, ax = plt.subplots()
    gps = db.entries['GPS quality'].apply(pd.to_numeric, errors='coerce')
    counts = [np.count_nonzero(gps == i) for i in range(6)]
    # counts.append(np.count_nonzero(np.isnan(gps)))
    ax.bar(labels, counts)
    ax.set_title(db.name)


plt.show()
