# %%
import rc_car_big
import matplotlib.pyplot as plt
import navbench as nb

TRAIN_SKIP = 40
TEST_SKIP = 40

paths = rc_car_big.get_paths()
dbs = rc_car_big.load_databases(paths[0:6], limits_metres=(0, 100))

train_route = dbs[0]
test_routes = dbs[1:]

analysis = rc_car_big.Analysis(train_route, TRAIN_SKIP)

_, ax0 = plt.subplots()
ax0.plot(train_route.x, train_route.y)
_, ax1 = plt.subplots()

for test_route in test_routes:
    test_entries, headings, nearest_train_entries, heading_error = analysis.get_headings(test_route, TEST_SKIP)

    label = test_route.name.replace('unwrapped_','')
    lines = ax0.plot(test_route.x, test_route.y, '--', label=label)
    colour = lines[0].get_color()
    ax0.plot(test_route.x[0], test_route.y[0], 'o', color=colour)

    nb.anglequiver(
        ax0,
        test_route.x[test_entries],
        test_route.y[test_entries],
        headings, color=colour, zorder=lines[0].zorder + 1, scale=300, scale_units='xy',
        alpha=0.8)

    ax1.plot(train_route.distance[nearest_train_entries], heading_error, label=label)
    ax1.set_xlabel("Distance along training route (m)")
    ax1.set_ylabel("Heading error (°)")

ax0.axis('equal')
ax0.legend()
ax1.legend()

plt.show()
