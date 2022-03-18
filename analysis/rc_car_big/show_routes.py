import sys
sys.path.append('../..')

import navbench as nb
import os
from glob import glob
DBROOT = '../../datasets/rc_car/rc_car_big'
paths = glob(os.path.join(DBROOT, 'unwrapped_*'))
dbs = [nb.Database(path) for path in paths]

import matplotlib.pyplot as plt
for db in dbs:
    plt.plot(db.x, db.y, alpha=0.5)
plt.axis('equal')
names = [db.name.replace("unwrapped_", "") for db in dbs]
plt.legend(names)
plt.show()
