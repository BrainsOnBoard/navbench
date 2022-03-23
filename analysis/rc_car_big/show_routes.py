import os
ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
import sys
sys.path.append(ROOT)

import gm_plotting
import rc_car_big
import matplotlib.pyplot as plt


# **YUCK**: We forgot to save this in our CSV files
UTM_ZONE = (30, 'U')

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths)

client = gm_plotting.APIClient()

_, ax = plt.subplots()
for db in dbs:
    mlat, mlon = gm_plotting.utm_to_merc(db.x * 1000, db.y * 1000, *UTM_ZONE)
    ax.plot(mlon, mlat, alpha=0.7)
ax.axis('equal')
names = [db.name.replace("unwrapped_", "") for db in dbs]
ax.legend(names)

client.add_satellite_image_background(ax)

plt.show()
